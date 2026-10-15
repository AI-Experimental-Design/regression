#!/usr/bin/env python3
"""Train a one-feature classifier by gradient descent, written out by hand.

--model linear     fit the line wx + b to the 0/1 labels with MSE, then call
                   anything with wx + b >= 0.5 class 1
--model logistic   pass wx + b through a sigmoid to get a probability, fit
                   with binary cross-entropy, then call anything with
                   probability >= 0.5 class 1

The training loop is the same one train_line.py uses. Only the activation
and the loss change. The decision boundary is the x where the model output
crosses 0.5.

Requires: numpy

Usage:
    python train_logistic.py --data out/class.data.tsv --model logistic --out_prefix out/class_logistic
    python train_logistic.py --data out/class.data.tsv --model linear --out_prefix out/class_linear
"""
import argparse

import numpy as np

import regression as reg


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--x_col', help='x column name (default: first column)')
    p.add_argument('--y_col', help='label column name (default: second column)')
    p.add_argument('--model', choices=['linear', 'logistic'], default='logistic')
    p.add_argument('--w0', type=float, default=0.0)
    p.add_argument('--b0', type=float, default=0.0)
    p.add_argument('--lr', type=float,
                   help='learning rate (default 0.002 for linear, 0.1 for logistic)')
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--snapshot_epochs', type=int, nargs='+',
                   default=[1, 10, 50, 100, 200, 500, 1000, 2000, 5000])
    p.add_argument('--snapshot_every', type=int)
    p.add_argument('--out_prefix', required=True)
    return p.parse_args()


def boundary(w, b, model):
    """x where the output crosses 0.5. linear: wx+b=0.5. logistic: wx+b=0."""
    target = 0.5 if model == 'linear' else 0.0
    return (target - b) / w if w != 0 else float('nan')


def main():
    args = get_args()
    x, y = reg.read_xy(args.data, args.x_col, args.y_col)
    loss_name = 'mse' if args.model == 'linear' else 'bce'
    if args.lr is None:
        args.lr = 0.002 if args.model == 'linear' else 0.1

    snaps = reg.snapshot_epochs(args.epochs, args.snapshot_every, args.snapshot_epochs)
    rows = reg.train(x, y, args.w0, args.b0, args.lr, args.epochs,
                     loss_name, args.model, snaps)

    out = []
    for epoch, L, w, b, dw, db in rows:
        pred = (reg.predict(x, w, b, args.model) >= 0.5).astype(float)
        acc = float(np.mean(pred == y))
        xb = boundary(w, b, args.model)
        out.append((epoch, L, acc, w, b, xb))
        print(f'epoch {epoch:04d} {loss_name}={L:.4f} acc={acc:.3f} '
              f'w={w:+.3f} b={b:+.3f} boundary={xb:.3f}')

    params_file = args.out_prefix + '.params.tsv'
    reg.write_params(params_file, out, ['epoch', 'loss', 'acc', 'w', 'b', 'boundary'],
                     comment=f'model={args.model} loss={loss_name} lr={args.lr}')
    print(f'wrote {params_file}')


if __name__ == '__main__':
    main()
