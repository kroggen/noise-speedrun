"""Check Relay-hard's credits against autograd on the hard model.

Autograd through the hard model is exact where the model is smooth (the MLPs, the norms, the head) and
through the hard attention's value path (the gather/mean is linear in the values), and zero on the
score and gate paths (flat a.e.). So the cosines to check:
- c_v, the writers, the MLP matrices, the head, the embedding, the scalars: ~1 (measured vs exact);
- c_q, c_k, ve_gate: autograd is zero there; the estimator's swap regret and flip gain must be nonzero.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.val.run import PRESETS, Settings, make_model, Relay  # noqa: E402
from utils.model_hardk import GPTConfig  # noqa: E402

torch.manual_seed(0)
device = torch.device('cpu')
n_layer, width = PRESETS['2m']
config = GPTConfig(n_layer=n_layer, n_head=width // 64, n_kv_head=width // 64, n_embd=width,
                   sequence_len=512, ktop=64)
import train.val.run as rh  # noqa: E402
rh.WIDTH = width
rh.HIDDEN = 4 * width

g = torch.Generator().manual_seed(1)
x = torch.randint(0, config.vocab_size, (2, 512), generator=g)
y = torch.randint(0, config.vocab_size, (2, 512), generator=g)

# the estimator (measured credits)
model = make_model(device, 0, config)
settings = Settings(route_k=4 * width)
estimator = Relay(model, settings, 0, 0, 1)
gradients = estimator.estimate(x, y)
estimator.close()

# autograd (exact on the smooth and value paths, zero on scores and gates)
ref = make_model(device, 0, config)
ref.load_state_dict(model.state_dict())
ref.requires_grad_(True)
loss = ref(x, y, loss_reduction='sum')
loss.backward()

print(f'clean loss {loss.item() / y.numel():.4f}   (estimator saw {estimator.capture.clean_loss.mean():.4f})')
print(f'{"parameter":44s} {"cosine":>8s} {"|est|":>10s} {"|autograd|":>10s}')
for name, p in ref.named_parameters():
    if p.grad is None:
        continue
    est = gradients.get(name)
    ag = p.grad.float()
    if est is None or not torch.isfinite(est).all():
        print(f'{name:44s} {"None":>8s} {0.0:10.2e} {ag.norm().item():10.2e}')
        continue
    cos = torch.dot(est.flatten(), ag.flatten()).item() / (est.norm().item() * ag.norm().item() + 1e-30)
    print(f'{name:44s} {cos:8.3f} {est.norm().item():10.2e} {ag.norm().item():10.2e}')
