"""Real ray/triangle intersections against the unchanged factory collision mesh."""
import time
import numpy as np
import warp as wp
from .temporal_data import _module

@wp.kernel
def rays(mesh:wp.uint64,origin:wp.vec3,directions:wp.array(dtype=wp.vec3),out:wp.array(dtype=float)):
    i=wp.tid();hit=wp.mesh_query_ray(mesh,origin,directions[i],40.)
    value=float(-1.)
    if hit.result:value=hit.t
    out[i]=value

class MeshSensor:
    def __init__(self,vertices,faces,sim_root,count=20000):
        if count<1:raise ValueError('positive ray count required')
        wp.config.kernel_cache_dir='/tmp/hnuter-factory-expert-warp';wp.init()
        self.module=_module(sim_root/'perception/mid360.py','_factory_expert_mid360')
        self.count=count
        with wp.ScopedDevice('cpu'):
            self.mesh=wp.Mesh(points=wp.array(vertices,dtype=wp.vec3),indices=wp.array(faces.ravel(),dtype=wp.int32))
    def capture(self,state,stamp,index):
        m=self.module;directions=m.scan_directions(index,m.Mid360Config(rays_per_frame=self.count));pose=m.sensor_pose(state)
        with wp.ScopedDevice('cpu'):
            out=wp.empty(self.count,dtype=float)
            wp.launch(rays,dim=self.count,inputs=[self.mesh.id,wp.vec3(*pose[:3,3]),wp.array(directions@pose[:3,:3].T,dtype=wp.vec3)],outputs=[out])
            distances=out.numpy()
        valid=np.isfinite(distances)&(distances>=.1)&(distances<=40.)
        distances[~valid]=np.nan
        return m.LidarFrame(directions[valid]*distances[valid,None],pose,stamp,self.count,distances,directions,
                            np.full(valid.sum(),stamp),index)
