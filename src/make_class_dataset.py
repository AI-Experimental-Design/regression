#!/usr/bin/env python3
"""Generate a two-class dataset on a single feature x and save it to a file.

Points with x above --boundary are mostly class 1 and points below are
mostly class 0. --noise blurs the boundary so the classes overlap a little.
--outliers adds that many class 1 points far to the right. They are easy
to classify, but they pull a regression line toward them, which moves its
decision boundary.

Usage:
    python make_class_dataset.py --out out/class.data.tsv
    python make_class_dataset.py --outliers 10 --out out/class_outliers.data.tsv
"""
import argparse

import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--boundary', type=float, default=5.0)
    p.add_argument('--noise', type=float, default=0.75,
                   help='sd of the blur around the boundary')
    p.add_argument('--x_min', type=float, default=0.0)
    p.add_argument('--x_max', type=float, default=10.0)
    p.add_argument('--n_points', type=int, default=60)
    p.add_argument('--outliers', type=int, default=0,
                   help='number of extra class 1 points far to the right')
    p.add_argument('--outlier_x', type=float, nargs=2, default=[40.0, 50.0],
                   help='x range for the outliers')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    return p.parse_args()


def main():
    args = get_args()
    rng = np.random.default_rng(args.seed)
    x = rng.uniform(args.x_min, args.x_max, size=args.n_points)
    y = (x + rng.normal(0, args.noise, size=args.n_points) > args.boundary).astype(float)
    if args.outliers:
        xo = rng.uniform(*args.outlier_x, size=args.outliers)
        x = np.concatenate([x, xo])
        y = np.concatenate([y, np.ones(args.outliers)])
    np.savetxt(args.out, np.column_stack([x, y]), fmt='%.6f',
               delimiter='\t', header='x\ty', comments='')
    print(f'wrote {args.out}: {len(x)} points ({args.outliers} outliers), '
          f'{y.mean():.1%} class 1, baseline accuracy = {max(y.mean(), 1 - y.mean()):.3f}')


if __name__ == '__main__':
    main()
