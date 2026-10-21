"""Check Relay-quant's credits against exact autograd on the int8-weight model.

Every path of the qat-mode model is smooth (the lattice is refreshed between steps, never
differentiated), so autograd computes dL/dW_eff exactly — the same object the estimator assembles by
measurement. The cosines should match the original Relay's soft-model check (~0.98 at init).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train.val.run as rq  # noqa: E402
from train.val.run import PRESETS, Settings, make_model, Relay  # noqa: E402
from utils.model_quant import GPTConfig  # noqa: E402

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_layer, width = PRESETS['38m']
rq.WIDTH = width
rq.HIDDEN = 4 * width
config = GPTConfig(n_layer=n_layer, n_head=width // 64, n_kv_head=width // 64, n_embd=width,
                   sequence_len=2048, quant_mode='qat')

g = torch.Generator().manual_seed(1)
x = torch.randint(0, config.vocab_size, (2, 2048), generator=g)
y = torch.randint(0, config.vocab_size, (2, 2048), generator=g)

model = make_model(device, 0, config)
estimator = Relay(model, Settings(route_k=4 * width), 0, 0, 1)
gradients = estimator.estimate(x.to(device), y.to(device))
estimator.close()

ref = make_model(device, 0, config)
ref.load_state_dict(model.state_dict())
ref.requires_grad_(True)
for m in ref.modules():
    if hasattr(m, 'shadow'):
        m.weight.requires_grad_(True)
if device.type == 'cuda':
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = ref(x.to(device), y.to(device), loss_reduction='sum')
else:
    loss = ref(x.to(device), y.to(device), loss_reduction='sum')
loss.backward()

print(f'clean loss {loss.item() / y.numel():.4f}   '
      f'(estimator saw {estimator.capture.clean_loss.mean():.4f})')
names = [(name + '.weight', m.weight) for name, m in ref.named_modules() if hasattr(m, 'shadow')]
names += [(n, p) for n, p in ref.named_parameters() if 'shadow' not in n]
print(f'{"parameter":44s} {"cosine":>8s} {"|est|":>10s} {"|autograd|":>10s}')
for key, p in names:
    est, ag = gradients.get(key), p.grad.float()
    if est is None or ag is None:
        print(f'{key:44s} {"None":>8s}')
        continue
    cos = torch.dot(est.flatten(), ag.flatten()).item() / (est.norm().item() * ag.norm().item() + 1e-30)
    print(f'{key:44s} {cos:8.3f} {est.norm().item():10.2e} {ag.norm().item():10.2e}')
