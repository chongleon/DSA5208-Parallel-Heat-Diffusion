#!/usr/bin/env python3
import csv, sys, statistics, collections
if len(sys.argv) != 3:
    raise SystemExit('Usage: python3 summarize_gpu_results.py input.csv output.csv')
with open(sys.argv[1], newline='') as f:
    rows = list(csv.DictReader(f))
groups = collections.defaultdict(list)
for row in rows:
    key = tuple(row[k] for k in ('mode','N','Steps','Ranks','GPUs'))
    groups[key].append(row)
with open(sys.argv[2], 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['mode','N','Steps','Ranks','GPUs','Repeats','RuntimeMean','RuntimeStd','CommunicationMean','ComputationMean'])
    for key, group in sorted(groups.items()):
        times = [float(r['Runtime']) for r in group]
        w.writerow([*key,len(group),f'{statistics.mean(times):.9f}',f'{statistics.stdev(times):.9f}' if len(times)>1 else '0',f'{statistics.mean(float(r["CommunicationTime"]) for r in group):.9f}',f'{statistics.mean(float(r["ComputationTime"]) for r in group):.9f}'])
