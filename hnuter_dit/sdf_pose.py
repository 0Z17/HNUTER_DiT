"""Small static-scene SE(3) spline curriculum, independent of older checkpoints.

Angles are local XYZ Euler coordinates (bounded below 0.7 rad in this pilot).
All physical samples include analytic body angular velocity/acceleration.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation

HORIZON = 20.
CONTROL_COUNT = 12
FREE = slice(3, 9)
KNOTS = np.r_[np.zeros(4), np.arange(1, 9)*HORIZON/9, np.full(4, HORIZON)]
PROGRESS = np.array([.10, .25, .40, .60, .75, .90])
SCHEMA = 'hnuter_static_sdf_pose_v1'


def full_controls(free, target):
    free, target = np.asarray(free), np.asarray(target)
    if free.shape != (6, 6) or target.shape != (6,) or not np.isfinite(free).all() or not np.isfinite(target).all():
        raise ValueError('expected finite free[6,6], target[6]')
    return np.concatenate((np.zeros((3, 6)), free, np.repeat(target[None], 3, axis=0)))


def angular_derivatives(rpy, rates, accelerations):
    r,p,_ = np.moveaxis(rpy,-1,0); rd,pd,yd = np.moveaxis(rates,-1,0)
    rdd,pdd,ydd = np.moveaxis(accelerations,-1,0)
    cr,sr,cp,sp = np.cos(r),np.sin(r),np.cos(p),np.sin(p)
    omega = np.stack((rd-yd*sp, pd*cr+yd*sr*cp, -pd*sr+yd*cr*cp),axis=-1)
    alpha = np.stack((rdd-ydd*sp-yd*cp*pd,
        pdd*cr-pd*sr*rd+ydd*sr*cp+yd*cr*rd*cp-yd*sr*sp*pd,
        -pdd*sr-pd*cr*rd+ydd*cr*cp-yd*sr*rd*cp-yd*cr*sp*pd),axis=-1)
    return omega,alpha


class PoseSpline:
    def __init__(self, controls, origin=(0,0,0), world_from_local=None, start_rotation=None):
        self.controls=np.asarray(controls,dtype=float)
        if self.controls.shape!=(12,6) or not np.isfinite(self.controls).all():
            raise ValueError('pose controls must be finite [12,6]')
        self.origin=np.asarray(origin,dtype=float)
        self.frame=np.eye(3) if world_from_local is None else np.asarray(world_from_local)
        self.rotation=np.eye(3) if start_rotation is None else np.asarray(start_rotation)
        for matrix in (self.frame,self.rotation):
            if matrix.shape!=(3,3) or not np.allclose(matrix.T@matrix,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(matrix),1):
                raise ValueError('pose frames must be proper rotations')
        if self.origin.shape!=(3,) or not np.isfinite(self.origin).all():
            raise ValueError('invalid pose origin')
        self.spline=BSpline(KNOTS,self.controls,3,extrapolate=False)
        self.duration=HORIZON

    def sample(self,time):
        t=np.asarray(time,dtype=float)
        if not np.isfinite(t).all() or np.any(t < -1e-8) or np.any(t > HORIZON+1e-8):
            raise ValueError('sample outside fixed pose horizon')
        scalar=t.ndim==0; t=np.atleast_1d(np.clip(t,0,HORIZON))
        x,v,a=[self.spline(t,nu=i) for i in (0,1,2)]
        rot=self.rotation[None]@Rotation.from_euler('xyz',x[:,3:]).as_matrix()
        q=Rotation.from_matrix(rot).as_quat()[:,[3,0,1,2]]
        w,alpha=angular_derivatives(x[:,3:],v[:,3:],a[:,3:])
        result=dict(pos=x[:,:3]@self.frame.T+self.origin,vel=v[:,:3]@self.frame.T,
            acc=a[:,:3]@self.frame.T,quaternion=q,angular_velocity=w,angular_acceleration=alpha)
        return {k:val[0] if scalar else val for k,val in result.items()}

    def audit(self):
        # Dense audit supplemented by exact convex-hull derivative bounds.
        samples=self.sample(np.linspace(0,HORIZON,1001))
        v=self.spline.derivative(1).c[:CONTROL_COUNT-1,:3]
        a=self.spline.derivative(2).c[:CONTROL_COUNT-2,:3]
        return dict(speed_bound_mps=float(np.linalg.norm(v,axis=1).max()),
            acceleration_bound_mps2=float(np.linalg.norm(a,axis=1).max()),
            sampled_angular_speed_radps=float(np.linalg.norm(samples['angular_velocity'],axis=1).max()),
            sampled_angular_acceleration_radps2=float(np.linalg.norm(samples['angular_acceleration'],axis=1).max()),
            max_local_angle_rad=float(np.abs(self.controls[:,3:]).max()))


def proxy_clearance(path, boxes):
    # Evaluator/expert only; never an inference input.
    from .collision_loss import ROBOT_SPHERE_CENTERS,ROBOT_SPHERE_RADII
    poses=path.sample(np.linspace(0,HORIZON,1001))
    rotations=Rotation.from_quat(poses['quaternion'][:,[1,2,3,0]]).as_matrix()
    centers=poses['pos'][:,None]+np.einsum('tij,sj->tsi',rotations,np.array(ROBOT_SPHERE_CENTERS))
    boxes=np.asarray(boxes)
    d=np.abs(centers[:,:,None]-boxes[None,None,:,:3])-boxes[None,None,:,3:]/2
    dist=np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(d.max(axis=-1),0)-np.array(ROBOT_SPHERE_RADII)[None,:,None]
    return float(dist.min())
