#!/usr/bin/env python3
"""Compare heat-grid CSV outputs without third-party packages."""
import argparse
import csv
import math
import sys

p = argparse.ArgumentParser()
p.add_argument('--reference', required=True)
p.add_argument('--candidate', required=True)
p.add_argument('--atol', type=float, default=1e-10)
a = p.parse_args()

def load(path):
    with open(path, newline='') as f:
        rows = [[float(v) for v in row] for row in csv.reader(f)]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f'Empty or ragged grid: {path}')
    return rows

try:
    ref, got = load(a.reference), load(a.candidate)
    if (len(ref), len(ref[0])) != (len(got), len(got[0])):
        raise ValueError(f'Shape mismatch: {len(ref)}x{len(ref[0])} vs {len(got)}x{len(got[0])}')
    errors = [abs(x-y) for rr, gg in zip(ref, got) for x, y in zip(rr, gg)]
    if not all(math.isfinite(v) for row in ref+got for v in row):
        raise ValueError('Non-finite value found')
    maximum = max(errors)
    print(f'Shape: {len(ref)} x {len(ref[0])}')
    print(f'Maximum Absolute Error: {maximum:.17g}')
    print('Validation:', 'PASS' if maximum <= a.atol else 'FAIL')
    sys.exit(0 if maximum <= a.atol else 1)
except (OSError, ValueError) as e:
    print('Validation ERROR:', e, file=sys.stderr)
    sys.exit(2)
