#!/usr/bin/env python3
import argparse, csv, pathlib, re
p = argparse.ArgumentParser()
for name in ('log','csv','mode'):
    p.add_argument('--'+name, required=True)
for name in ('n','steps','ranks','gpus','repeat'):
    p.add_argument('--'+name, required=True, type=int)
a = p.parse_args()
content = pathlib.Path(a.log).read_text(errors='replace')
values = {}
for k in ('GPUOffload','Runtime','CommunicationTime','ComputationTime'):
    m = re.search(r'^'+k+r'=([^\s]+)\s*$', content, re.M)
    if not m:
        raise SystemExit(f'Missing {k} in {a.log}; refusing incomplete CSV row')
    values[k] = m.group(1)
if values['GPUOffload'] != 'PASS':
    raise SystemExit(f'GPU offloading failed in {a.log}')
fields = ['mode','N','Steps','Ranks','GPUs','Repeat','Runtime','CommunicationTime','ComputationTime','GPUOffload','Log']
row = dict(zip(fields, [a.mode,a.n,a.steps,a.ranks,a.gpus,a.repeat,values['Runtime'],values['CommunicationTime'],values['ComputationTime'],values['GPUOffload'],a.log]))
path = pathlib.Path(a.csv)
with path.open('a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    if path.stat().st_size == 0:
        writer.writeheader()
    writer.writerow(row)
print('Recorded:', a.log)
