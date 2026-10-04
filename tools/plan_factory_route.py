"""Surveyed mission waypoints only; each local flight path must come from DiT.

Collision mesh queries here are offline route-envelope diagnostics, never neural
inputs or replacements for live Newton ray returns or in-flight collision checks.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import warp as wp

@wp.kernel
def query(mesh:wp.uint64,points:wp.array(dtype=wp.vec3),distances:wp.array(dtype=float)):
    i=wp.tid();p=points[i];q=wp.mesh_query_point(mesh,p,8.)
    value=float(8.)
    if q.result:value=wp.length(wp.mesh_eval_position(mesh,q.face,q.u,q.v)-p)
    distances[i]=value


def main():
    p=argparse.ArgumentParser();p.add_argument('--sim-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    points=np.array([[16.5,7,13],[16.5,11.4,13],[16.5,15.8,13],[16.5,20.2,13],[12.1,20.2,13],[7.7,20.2,13]],float)
    wp.config.kernel_cache_dir='/tmp/factory-route-warp';wp.init();path=a.sim_root/'_build/collision.npz';data=np.load(path)
    with wp.ScopedDevice('cpu'):
        mesh=wp.Mesh(points=wp.array(data['vertices'],dtype=wp.vec3),indices=wp.array(data['faces'].ravel(),dtype=wp.int32))
        dense=np.concatenate([np.linspace(x,y,101) for x,y in zip(points[:-1],points[1:])])
        output=wp.empty(len(dense),dtype=float);wp.launch(query,dim=len(dense),inputs=[mesh.id,wp.array(dense,dtype=wp.vec3)],outputs=[output]);dist=output.numpy()
    record=dict(schema='hnuter_factory_mission_v1',start_position=points[0].tolist(),waypoints=points[1:].tolist(),goal_yaws=[.3,.6,.9,1.2,1.5],
        nominal_distance_m=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),minimum_centerline_surface_distance_m=float(dist.min()),
        per_segment_minimum_centerline_distance_m=dist.reshape(-1,101).min(1).tolist(),maximum_verification_spacing_m=.044,
        mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope='Surveyed high-level waypoint mission; no local path or obstacle truth is supplied to DiT. Centerline clearance is not a learned-path collision certificate.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
