"""STE baseline: backprop with surrogate derivatives on the hard model of utils/model_hardk.py.

The reference for how far autograd-with-hacks gets on the same non-differentiable model:
- the value path is exact (autograd's own gather/mean backward is A^T);
- the hard gate gets a straight-through estimator (the step's backward is the identity);
- the hard top-k attention's scores get a softmax surrogate backward (the forward is the hard selection,
  the backward pretends the selection probabilities were the softmax at the same logits) — the standard
  straight-through choice for discrete attention.

Same model, init, data order, batch, optimizer and schedule as train/val/run_hardk.py; the only
difference is the learning signal (autograd + surrogates instead of measured credit).

    torchrun --standalone --nproc_per_node=1 train/val/run_hardk_ste.py --preset 38m
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.model_hardk import GPTConfig, hard_topk, ste_gate, window_allowed  # noqa: E402
from run import (  # noqa: E402
    CUDA, PRESETS, SEED, Recipe, evaluate, fake_data, load_split, make_model, make_optimizer,
)
from utils.prepare_data import TRAIN_ROWS, VAL_ROWS  # noqa: E402


class HardAttnSTE(torch.autograd.Function):
    """Hard top-k attention with a surrogate backward: the values take the exact A^T gradient (the
    selection is linear in the values), the scores take the softmax's backward at the same logits."""

    @staticmethod
    def forward(ctx, qp, kp, v, window, ktop):
        B, T, H, D = qp.shape
        allowed = window_allowed(T, window, qp.device)
        qh, kh, vh = (z.transpose(1, 2).float() for z in (qp, kp, v))
        logits = (qh @ kh.transpose(-1, -2)) / D ** 0.5
        indices, valid, count = hard_topk(logits, allowed, ktop)
        kk = indices.shape[-1]
        picked = vh.unsqueeze(3).expand(-1, -1, -1, kk, -1).gather(
            2, indices.unsqueeze(-1).expand(-1, -1, -1, -1, D))
        out = ((picked * valid.unsqueeze(-1)).sum(-2) / count.unsqueeze(-1)).transpose(1, 2)
        ctx.save_for_backward(qp, kp, v, indices, valid, count)
        ctx.window, ctx.ktop = window, ktop
        return out.to(v.dtype)

    @staticmethod
    def backward(ctx, d_out):
        qp, kp, v, indices, valid, count = ctx.saved_tensors
        B, T, H, D = qp.shape
        dq, dk, dv = (torch.zeros_like(z, dtype=torch.float32) for z in (qp, kp, v))
        allowed = window_allowed(T, ctx.window, qp.device)
        eye = None
        for r in range(B):  # the (h, T, T) workspaces are transient per sequence
            qh, kh, vh = (z[r].transpose(0, 1).float() for z in (qp, kp, v))       # (h, T, d)
            dh = d_out[r].transpose(0, 1).float()                                  # (h, T, d)
            idx, val, cnt = indices[r], valid[r], count[r].float()
            # the value gradient, exact: A^T times the output gradient
            A = torch.zeros(H, T, T, device=qp.device)
            A.scatter_(-1, idx, torch.where(val, cnt.reciprocal().unsqueeze(-1), torch.zeros((), device=qp.device)))
            dv[r] = torch.bmm(A.transpose(1, 2), dh).transpose(0, 1)
            # the score gradient, the softmax surrogate at the same logits
            logits = (qh @ kh.transpose(1, 2)) / D ** 0.5
            p = logits.masked_fill(~allowed, float('-inf')).softmax(-1)
            g = torch.bmm(dh, vh.transpose(1, 2))                                  # (h, T, T) = <d_out_s, v_t>
            dl = p * (g - (p * g).sum(-1, keepdim=True))                            # the softmax's backward
            dq[r] = torch.bmm(dl, kh).transpose(0, 1) / D ** 0.5
            dk[r] = torch.bmm(dl.transpose(1, 2), qh).transpose(0, 1) / D ** 0.5
        return dq.to(qp.dtype), dk.to(kp.dtype), dv.to(v.dtype), None, None


def ste_attention(q, k, v, window, ktop):
    return HardAttnSTE.apply(q, k, v, window, ktop)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/hardk_ste'))
    parser.add_argument('--preset', choices=sorted(PRESETS), default='38m')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--ktop', type=int, default=64)
    parser.add_argument('--train-file', default='train.pt')
    parser.add_argument('--train-rows', type=int, default=TRAIN_ROWS)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--eval-every', type=int, default=0)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--fake-data', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu') if args.cpu else torch.device('cuda', 0)
    if not args.cpu:
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    n_layer, width = PRESETS[args.preset]
    config = GPTConfig(n_layer=n_layer, n_head=width // 64, n_kv_head=width // 64, n_embd=width,
                       sequence_len=args.seq_len, ktop=args.ktop)
    recipe = Recipe()
    batch = args.batch

    if args.fake_data:
        training = fake_data(config, max(args.train_rows, batch * 4), args.seq_len, args.seed)
        val_rows = fake_data(config, 8, args.seq_len, args.seed + 1)
    else:
        training = load_split(args.data_dir / args.train_file, args.train_rows)
        val_rows = load_split(args.data_dir / 'val.pt', VAL_ROWS)
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(SEED + 1))]
    steps2 = len(training) // batch
    last = min(args.steps, steps2) if args.steps else steps2

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'config.json').write_text(json.dumps(dict(
        script=__file__, preset=args.preset, model={**vars(config)}, seed=args.seed, estimator='ste',
        batch=batch, train_rows=len(training), recipe=asdict(recipe), fake_data=args.fake_data,
        torch_version=torch.__version__,
    ), indent=2, default=str) + '\n')

    model = make_model(device, args.seed, config)
    model.requires_grad_(True)
    for block in model.transformer.h:  # the surrogates: hard forward, soft backward
        block.attn.attention_fn = ste_attention
        block.attn.gate_fn = ste_gate
    optimizer = make_optimizer(model, recipe, steps2)

    print(f'ste {args.preset}: {last} steps at batch {batch}, ktop {args.ktop}', flush=True)
    log = (args.output / 'metrics.jsonl').open('a')
    torch.cuda.synchronize() if not args.cpu else None
    start, paused, elapsed = time.perf_counter(), 0.0, 0.0

    def set_rate(step, steps):
        left = (steps - step + 1) / steps
        for g in optimizer.param_groups:
            g.setdefault('initial_lr', g['lr'])
            g['lr'] = g['initial_lr'] * min(1.0, left / recipe.warmdown)

    def after(step, loss):
        nonlocal paused, elapsed
        if not args.cpu:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start - paused
        record = dict(step=step, train_loss=loss, train_seconds=round(elapsed, 2))
        if args.eval_every and step % args.eval_every == 0 and step < last:
            pause = time.perf_counter()
            record['val_loss'] = evaluate(model, val_rows)
            paused += time.perf_counter() - pause
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + '\n')
        log.flush()

    for step in range(1, last + 1):
        rows = training[(step - 1) * batch:step * batch].to(device)
        x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
        set_rate(step, steps2)
        optimizer.zero_grad(set_to_none=True)
        if CUDA:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(x, y)
        else:
            loss = model(x, y)
        loss.backward()
        flat = torch.cat([p.grad.float().reshape(-1) for _, p in model.named_parameters() if p.grad is not None])
        if not bool(torch.isfinite(flat).all()):
            raise FloatingPointError('Non-finite gradient')
        optimizer.step()
        after(step, loss.item())

    result = dict(val_loss=evaluate(model, val_rows), train_seconds=round(elapsed, 2), steps=last,
                  batch=batch, ktop=args.ktop, tokens=last * batch * (args.seq_len - 1))
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.output / 'final.pt')
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
