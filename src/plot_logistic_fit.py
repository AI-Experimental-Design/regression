#!/usr/bin/env python
"""Plot binary class data with a fitted logistic (sigmoid) curve.

Reads a TSV with a header. The first column is x, the second is the 0/1 label.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--w", type=float, required=True)
    p.add_argument("--b", type=float, required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("-x", "--xlabel", default="x")
    p.add_argument("-y", "--ylabel", default="label")
    p.add_argument("--width", type=float, default=3)
    p.add_argument("--height", type=float, default=2.2)
    p.add_argument("--title", default="")
    args = p.parse_args()

    data = np.loadtxt(args.data, delimiter="\t", skiprows=1, ndmin=2)
    x, label = data[:, 0], data[:, 1]

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    colors = np.where(label >= 0.5, "tab:orange", "tab:blue")
    ax.scatter(x, label, c=colors, s=12, alpha=0.7)

    grid = np.linspace(x.min() - 0.5, x.max() + 0.5, 400)
    p_hat = 1 / (1 + np.exp(-(args.w * grid + args.b)))
    ax.plot(grid, p_hat, color="black", lw=1.5)

    ax.axhline(0.5, color="gray", ls=":", lw=1)
    if args.w != 0:
        ax.axvline(-args.b / args.w, color="gray", ls="--", lw=1)

    ax.set_ylim(-0.15, 1.15)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    if args.title:
        ax.set_title(args.title, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
