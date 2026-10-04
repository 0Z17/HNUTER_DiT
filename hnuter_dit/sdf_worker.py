"""Private worker for persistent GPU SDF + DiT; NumPy scans/pose splines only over IPC."""
from pathlib import Path
import json
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hnuter_dit.planner_worker import atomic_json


def run(directory):
    import numpy as np
    from hnuter_dit.sdf_runtime import SDFRuntime
    config=json.loads((directory/'config.json').read_text())
    runtime=SDFRuntime(config['checkpoint_path'])
    description=runtime.describe()
    if description['checkpoint_sha256']!=config['expected_checkpoint_sha256']:raise RuntimeError('checkpoint changed during startup')
    atomic_json(directory/'status.json',dict(state='ready',model=description))
    sequence=0
    while True:
        path=directory/'request.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                new=int(z['sequence']);values={k:z[k].copy() for k in z.files} if new>sequence else None
            if values is not None:
                sequence=new
                try:
                    if bool(values['reset']):
                        runtime.reset(values['origin'],values['start_rotation'],values['target'],
                            world_origin=values.get('map_world_origin'),world_frame=values.get('map_world_frame'),
                            global_start=values.get('map_global_start'))
                    controls,metadata=runtime.scan(values['points'],values['pose'],float(values['stamp']),
                        bool(values['plan']),int(values['seed']),float(values['elapsed']))
                except Exception as exc:
                    traceback.print_exc();controls=np.zeros((0,6));metadata=dict(error=f'{type(exc).__name__}: {exc}')
                if bool(values.get('record_visualization', False)) and 'error' not in metadata:
                    # Optional diagnostic snapshot, never a planner input. Keep
                    # its scan timestamp and report its additional readback cost.
                    began=time.perf_counter()
                    folder=directory/'visualization';folder.mkdir(exist_ok=True)
                    field=runtime.map.latest_features[[2,3],:,:,16].detach().cpu().numpy()
                    np.savez_compressed(folder/f'{sequence:06d}.npz', esdf_m=field[0]*3,
                        known=field[1].astype(bool), stamp=values['stamp'], origin=runtime.origin,
                        z_world=np.float64(runtime.origin[2]+.125), sequence=np.int64(sequence))
                    metadata['visualization_export_seconds']=time.perf_counter()-began
                metadata['sequence']=sequence
                with (directory/'response.tmp').open('wb') as stream:
                    np.savez(stream,sequence=np.int64(sequence),controls=controls,metadata=json.dumps(metadata,allow_nan=False))
                (directory/'response.tmp').replace(directory/'response.npz')
        stop=directory/'stop.json'
        if stop.exists() and sequence>=json.loads(stop.read_text()).get('sequence',0):
            atomic_json(directory/'status.json',dict(state='stopped',sequence=sequence,model=description));return
        time.sleep(.001)


if __name__=='__main__':
    directory=Path(sys.argv[1])
    try:run(directory)
    except BaseException as exc:
        traceback.print_exc();atomic_json(directory/'status.json',dict(state='error',error=f'{type(exc).__name__}: {exc}'));raise
