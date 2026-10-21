"""Quantized-weight variant of the fixed model: the original soft-attention GPT with every large weight
matrix and embedding table living on an int8 lattice.

The architecture is the record's (softmax attention, soft value-embedding gates, relu^2 MLPs, the softcap
head); the change is the parameters. Each QLinear/QEmbedding holds:
- `shadow`: a float latent parameter (the only trained quantity; the optimizer moves it),
- `weight_int` (int8) and `scale` (per output row): the deployed weights, W_eff = scale * weight_int,
- `weight`: the dequantized view used by the forward and read by the estimator's closed-form steps.

Weights enter the forward only through matmuls, so the loss is smooth in W_eff and in every activation:
the non-differentiability lives only in the shadow -> round map, which no estimator ever differentiates.
`apply_quant` re-quantizes (round-to-nearest or stochastic rounding) after each optimizer step.

Two compute styles (--mode):
- 'qat': the quantized weights are dequantized and the matmuls run in bf16 autocast as usual
  (fake quantization; the deployed function is exactly W_eff);
- 'int': integer arithmetic at every weight matmul: the input is dynamically quantized per token to
  int8 (absmax scale) and the product accumulates in fp32 (integer-exact for K <= 1023 and
  near-exact above; the discreteness of weights and activations — the thing under test — is exact).
  Norms, rotary, softmax and relu^2 stay in bf16/fp32, as in a W8A8 inference engine's non-GEMM ops.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from utils.model import apply_rotary_emb, attention, has_ve, norm


def quantize(w, stochastic):
    """Per-row symmetric int8: returns (int8 weights, fp32 scales). stochastic: round-to-nearest or
    stochastic rounding (floor + Bernoulli on the fraction; unbiased in expectation)."""
    scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    q = w / scale
    if stochastic:
        q = torch.floor(q) + (torch.rand_like(q) < q - torch.floor(q)).float()
    else:
        q = torch.round(q)
    return q.clamp_(-127, 127).to(torch.int8), scale.squeeze(-1)


class QLinear(nn.Module):
    """A bias-free linear layer whose weights live on a per-row int8 lattice."""

    def __init__(self, in_features, out_features, mode='qat'):
        super().__init__()
        self.in_features, self.out_features, self.mode = in_features, out_features, mode
        self.shadow = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_int', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('scale', torch.ones(out_features))
        self.register_buffer('weight', torch.zeros(out_features, in_features))
        self.register_buffer('w_cm', torch.zeros(in_features, out_features, dtype=torch.int8),
                             persistent=False)  # column-major view for torch._int_mm

    @torch.no_grad()
    def refresh(self, stochastic):
        self.weight_int, self.scale = quantize(self.shadow.detach().float(), stochastic)
        self.weight = self.weight_int.float() * self.scale.unsqueeze(-1)
        self.weight.requires_grad_(True)  # keep the backprop baseline's graph: W_eff stays a grad leaf
        self.w_cm = self.weight_int.t()  # (K, N) column-major view: what torch._int_mm takes

    def forward(self, x):
        if self.mode == 'int':
            with torch.autocast('cuda', enabled=False):
                shape = x.shape
                a = x.reshape(-1, shape[-1]).float()
                sx = a.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
                a = (a / sx).round().clamp_(-127, 127).to(torch.int8)
                if x.is_cuda and hasattr(torch, '_int_mm'):  # int8 tensor cores, int32 accumulation
                    y = torch._int_mm(a, self.w_cm).float()
                else:  # integer-exact in fp32 for K <= 1023, near-exact above
                    y = a.float() @ self.weight_int.float().T
                y = y * sx * self.scale
                return y.reshape(*shape[:-1], self.out_features).to(x.dtype)
        return F.linear(x, self.weight)


class QEmbedding(nn.Module):
    """An embedding table whose rows live on a per-row int8 lattice."""

    is_embedding = True

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.shadow = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.register_buffer('weight_int', torch.zeros(num_embeddings, embedding_dim, dtype=torch.int8))
        self.register_buffer('scale', torch.ones(num_embeddings))
        self.register_buffer('weight', torch.zeros(num_embeddings, embedding_dim))

    @torch.no_grad()
    def refresh(self, stochastic):
        self.weight_int, self.scale = quantize(self.shadow.detach().float(), stochastic)
        self.weight = self.weight_int.float() * self.scale.unsqueeze(-1)
        self.weight.requires_grad_(True)

    def forward(self, idx):
        w = self.weight
        if w.device.type == 'cuda':  # the record's embedding tables are bf16: keep F.embedding autocast-consistent
            w = w.to(torch.bfloat16)
        return F.embedding(idx, w)


@torch.no_grad()
def apply_quant(model, stochastic=False):
    """Re-quantize every quantized module from its shadow latents."""
    for m in model.modules():
        if isinstance(m, (QLinear, QEmbedding)):
            m.refresh(stochastic)


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 4096
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 8
    n_embd: int = 512
    # Per block, repeating: S = half-sequence window, L = full. The last block is always L.
    window_pattern: str = 'SSSL'
    mlp_mult: int = 4
    quant_mode: str = 'qat'   # 'qat': dequantized matmuls; 'int': int8 activations + integer GEMMs


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        mode = config.quant_mode
        self.c_q = QLinear(self.n_embd, self.n_head * self.head_dim, mode)
        self.c_k = QLinear(self.n_embd, self.n_kv_head * self.head_dim, mode)
        self.c_v = QLinear(self.n_embd, self.n_kv_head * self.head_dim, mode)
        self.c_proj = QLinear(self.n_embd, self.n_embd, mode)
        # The value embedding is scaled per head by 2 * sigmoid(gate), read from the block input's first
        # 32 channels. Tiny (8 x 32): kept in float.
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer) else None
        )

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = (apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin))
        q, k = (norm(q), norm(k))
        y = attention(q, k, v, window_size[0])
        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = getattr(config, 'mlp_mult', 4) * config.n_embd
        mode = config.quant_mode
        self.c_fc = QLinear(config.n_embd, width, mode)
        self.c_proj = QLinear(width, config.n_embd, mode)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        # Pre-norm block. attn.c_proj and mlp.c_proj are the "writers" into the residual stream.
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab = (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to * pad_vocab_size_to
        self.transformer = nn.ModuleDict({
            'wte': QEmbedding(padded_vocab, config.n_embd),
            'h': nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = QLinear(config.n_embd, padded_vocab, config.quant_mode)
        # The stream entering block i is resid_lambdas[i] * x + x0_lambdas[i] * x0, a learned mix of the
        # previous block's output and the normalized token embedding. Tiny: kept in float.
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): QEmbedding(padded_vocab, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.shadow, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.shadow, mean=0.0, std=0.001)
        s = 3 ** 0.5 * self.config.n_embd ** (-0.5)
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.shadow, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.shadow, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.shadow, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.shadow)
            torch.nn.init.uniform_(block.mlp.c_fc.shadow, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.shadow)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.shadow, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        (self.cos, self.sin) = (cos, sin)
        apply_quant(self)  # round-to-nearest at init

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.shadow.device
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = (freqs.cos().bfloat16(), freqs.sin().bfloat16())
        return (cos[None, :, None, :], sin[None, :, None, :])

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = (config.sequence_len, config.sequence_len // 2)
        char_to_w = {'L': (long_w, 0), 'S': (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)
        return sizes

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        x = norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)  # soft cap
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=loss_reduction,
            )
        return logits
