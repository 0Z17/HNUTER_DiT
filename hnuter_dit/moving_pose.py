"""Moving-start spline decoder; no collision repair, clipping or endpoint hold."""
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation
from .sdf_pose import angular_derivatives
KNOT_TIMES=np.arange(8)*.5+.6

class MovingReference:
    def __init__(self,controls,condition):
        controls=np.asarray(controls,float);c=np.asarray(condition,float)
        if controls.shape!=(8,6) or c.shape!=(17,) or not np.isfinite(controls).all() or not np.isfinite(c).all():
            raise ValueError('finite controls[8,6], condition[17] required')
        rpy=Rotation.from_quat(c[[10,11,12,9]]).as_euler('xyz')
        e=np.stack([angular_derivatives(rpy,axis,np.zeros(3))[0] for axis in np.eye(3)],axis=1)
        rates=np.linalg.solve(e,c[6:9])
        initial=np.r_[np.zeros(3),rpy];velocity=np.r_[c[3:6],rates];acc=np.r_[c[13:16],np.zeros(3)]
        terminal_rate=(controls[-1]-controls[-2])/.5
        self.spline=make_interp_spline(np.r_[0.,KNOT_TIMES],np.vstack([initial,controls]),k=5,
            bc_type=([(1,velocity),(2,acc)],[(1,terminal_rate),(2,np.zeros(6))]))
    def sample(self,t):
        scalar=np.ndim(t)==0;t=np.atleast_1d(t)
        if not np.isfinite(t).all() or np.any(t<0) or np.any(t>4.10000001):raise ValueError('invalid moving reference time')
        x,v,a=[self.spline(t,nu=i) for i in range(3)]
        omega,alpha=angular_derivatives(x[:,3:],v[:,3:],a[:,3:])
        d=dict(pos=x[:,:3],quaternion=Rotation.from_euler('xyz',x[:,3:]).as_quat()[:,[3,0,1,2]],
               vel=v[:,:3],acc=a[:,:3],angular_velocity=omega,angular_acceleration=alpha)
        return {k:v[0] if scalar else v for k,v in d.items()}
