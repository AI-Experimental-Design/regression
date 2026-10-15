#!/usr/bin/env python3
"""Run a trained one-feature classifier on new x values.

Prints, for each x, the weighted sum z = wx + b, the model output, and the
0/1 prediction. For the logistic model the output is a probability.

Usage:
    python logistic_inference.py --model logistic --w 1.9 --b -9.6 --x 3 4 5 6 7
"""
import argparse

import regression as reg


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['linear', 'logistic'], default='logistic')
    p.add_argument('--w', type=float, required=True)
    p.add_argument('--b', type=float, required=True)
    p.add_argument('--x', type=float, nargs='+', required=True)
    return p.parse_args()


def main():
    args = get_args()
    label = 'P(class 1)' if args.model == 'logistic' else 'wx + b'
    print(f'model={args.model}')
    print(f'{"x":>8} {"z=wx+b":>8} {label:>11} {"prediction":>10}')
    for xi in args.x:
        z = args.w * xi + args.b
        out = reg.predict(xi, args.w, args.b, args.model)
        print(f'{xi:8.2f} {z:8.3f} {out:11.3f} {int(out >= 0.5):>10d}')


if __name__ == '__main__':
    main()
