"""Backprop baseline on the int8-weight model: the exact gradient at the quantized point.

W_eff is a constant of the forward (the lattice is refreshed between steps, never differentiated), so
autograd computes dL/dW_eff exactly — the same object Relay measures, and the same object STE-QAT
approximates when differentiating the latent through round := identity. This run answers how much
Relay's measured credit (cosine ~0.99) costs against exact autograd on the same quantized function.

The shadows are updated with the autograd gradient and re-snapped after every step, exactly as in
train/val/run_quant.py. qat mode only (the integer-GEMM forward has no autograd path through the int8
weights, which is rather the point).

    torchrun --standalone --nproc_per_node=1 train/val/run_quant_bp.py
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import torch

if os.environ.get('NO_COMPILE'):  # smoke tests on CPU: eager mode (must precede utils.model's flex compile)
    torch.compile = lambda fn=None, **kw: fn if callable(fn) else (lambda f: f)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.model_quant import GPTConfig, apply_quant  # noqa: E402
from run import (  # noqa: E402
    CUDA, PRESETS, SEED, Recipe, Settings, evaluate, fake_data, load_split, make_model, make_optimizer,
)
from utils.prepare_data import TRAIN_ROWS, VAL_ROWS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/quant-bp'))
    parser.add_argument('--preset', choices=sorted(PRESETS), default='38m')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--sr', action='store_true')
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
                       sequence_len=args.seq_len, quant_mode='qat')
    recipe = Recipe()
    stochastic = args.sr
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
        script=__file__, preset=args.preset, model={**vars(config)}, seed=args.seed, estimator='backprop',
        sr=args.sr, batch=batch, train_rows=len(training), recipe=asdict(recipe), fake_data=args.fake_data,
        torch_version=torch.__version__,
    ), indent=2, default=str) + '\n')

    model = make_model(device, args.seed, config)
    model.requires_grad_(True)  # the float params (ve_gate, lambdas) learn by autograd directly
    quantized = [m for m in model.modules() if hasattr(m, 'shadow')]
    for m in quantized:  # W_eff enters the graph as a leaf: autograd returns dL/dW_eff exactly
        m.weight.requires_grad_(True)
    optimizer = make_optimizer(model, recipe, steps2)  # steps the shadows and the float params

    print(f'backprop {args.preset}: {last} steps at batch {batch} (gradient at the quantized point)',
          flush=True)
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
        for m in quantized:
            m.weight.grad = None
        if CUDA:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(x, y)
        else:
            loss = model(x, y)
        loss.backward()
        flat = torch.cat([m.weight.grad.float().reshape(-1) for m in quantized if m.weight.grad is not None]
                         + [p.grad.float().reshape(-1) for p in [model.resid_lambdas, model.x0_lambdas]
                            if p.grad is not None]
                         + [blk.attn.ve_gate.weight.grad.float().reshape(-1) for blk in model.transformer.h
                            if blk.attn.ve_gate is not None and blk.attn.ve_gate.weight.grad is not None])
        if not bool(torch.isfinite(flat).all()):
            raise FloatingPointError('Non-finite gradient')
        for m in quantized:  # the exact gradient at the quantized point, written into the shadow latents
            m.shadow.grad = m.weight.grad.detach().clone()
        optimizer.step()
        apply_quant(model, stochastic)
        for m in quantized:
            m.weight.grad = None
        after(step, loss.item())

    result = dict(val_loss=evaluate(model, val_rows), train_seconds=round(elapsed, 2), steps=last,
                  batch=batch, mode='qat-bp', sr=args.sr, tokens=last * batch * (args.seq_len - 1))
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.output / 'final.pt')
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
