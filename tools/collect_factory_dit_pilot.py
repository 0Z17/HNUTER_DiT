"""Run the bounded, predeclared within-factory route pilot in subprocesses."""
import argparse,concurrent.futures,json,os,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--workers',type=int,default=2);p.add_argument('--sim-root',type=Path,default=Path('/home/z017/research/isaacsim-industrial-reactor'))
 p.add_argument('--collector-root',type=Path,default=Path('/home/z017/research/HNUTER_trajectory_collector'));a=p.parse_args()
 if not 1<=a.workers<=2:p.error('this pilot is limited to one or two workers')
 protocol=json.loads(a.protocol.read_text())
 if protocol['schema']!='within_factory_route_holdout_v1' or len(protocol['routes'])!=12:raise ValueError('expected fixed 12-route pilot')
 a.output.mkdir(parents=True,exist_ok=False);(a.output/'protocol.json').write_bytes(a.protocol.read_bytes())
 def run(i):
  points=protocol['routes'][i];mission=a.output/f'mission_{i:02}.json';mission.write_text(json.dumps(dict(start_position=points[0],waypoints=points[1:])))
  with (a.output/f'collect_{i:02}.log').open('w') as f:
   command=[sys.executable,'-m','hnuter_dit.factory_expert_data','collect','--mission',str(mission),'--output',str(a.output/f'route_{i:02}'),
      '--seed',str(protocol['seed']+i),'--max-duration','90','--sim-root',str(a.sim_root),'--collector-root',str(a.collector_root)]
   rc=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
  report=json.loads((a.output/f'route_{i:02}/rollout.json').read_text())
  return dict(route=i,role=protocol['roles'][str(i)],returncode=rc,accepted=report['accepted'],windows=report.get('windows',0),error=report.get('error'))
 rows=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  for job in concurrent.futures.as_completed([pool.submit(run,i) for i in range(12)]):
   rows.append(job.result());temp=a.output/'collection_status.tmp';temp.write_text(json.dumps(rows,indent=2)+'\n');temp.replace(a.output/'collection_status.json')
   print(json.dumps(rows[-1]),flush=True)
if __name__=='__main__':main()
