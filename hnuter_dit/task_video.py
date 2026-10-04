"""Truthful schematic telemetry replay of one Surface and one Inspection run.

This tool renders saved measured states and rays; it does not create RGB camera
frames or rerun physics. Preflight-only LiDAR is explicitly marked as frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


def acceptance_display(report, task):
    """Keep the relaxed stored Surface flag separate from stricter diagnostics."""
    reported = bool(report['passed'])
    if task != 'surface':
        return dict(verdict='PASS' if reported else 'FAIL', saved_report_passed=reported,
                    details='Single-target geometric visibility and dwell; flight checks are separate')
    fraction = report.get('whole_operate_force_interval_fraction')
    threshold = report.get('acceptance', {}).get('force_interval_fraction_min', .9)
    whole = None if fraction is None else bool(fraction >= threshold)
    strict = report.get('actual_task_geometry', {}).get('passed')
    strict = None if strict is None else bool(strict)
    word = lambda value: 'UNKNOWN' if value is None else ('PASS' if value else 'FAIL')
    verdict = ('SOFT PASS' if whole is True and strict is True else 'PARTIAL') if reported else 'FAIL'
    return dict(verdict=verdict, saved_report_passed=reported,
                post_settle_soft_gate_passed=reported, full_window_force_passed=whole,
                strict_plane_geometry_passed=strict,
                whole_operate_force_interval_fraction=fraction,
                full_window_force_fraction_threshold=threshold,
                details=f'Post-settle soft gate {word(reported)}; full-window force {word(whole)}; strict plane {word(strict)}')


class Episode:
    def __init__(self, directory, index, method='Expert', *, evaluation_directory=None,
                 evaluation_method=None):
        self.directory, self.index, self.method = Path(directory), int(index), str(method)
        self.evaluation_directory = Path(evaluation_directory) if evaluation_directory is not None else None
        self.evaluation_method = evaluation_method
        if (self.evaluation_directory is None) != (evaluation_method is None):
            raise ValueError('evaluation_directory and evaluation_method must be supplied together')
        self.sources = [self.directory/'collection.json',
                        self.directory/f'report_{index:03}.json',
                        self.directory/f'trace_{index:03}.json',
                        self.directory/f'episode_{index:03}.npz']
        if self.evaluation_directory is not None:
            if not evaluation_method or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for c in evaluation_method):
                raise ValueError('evaluation_method must be an ASCII method name')
            initial_report = self.sources[1]
            self.sources[1] = self.evaluation_directory/f'{index:03}_{evaluation_method}.json'
            self.sources[2] = self.evaluation_directory/f'{index:03}_{evaluation_method}_trace.json'
            self.sources.append(initial_report)
            aggregate = self.evaluation_directory/'evaluation.json'
            if aggregate.exists():self.sources.append(aggregate)
        self.collection = json.loads(self.sources[0].read_text())
        self.report = json.loads(self.sources[1].read_text())
        self.trace = json.loads(self.sources[2].read_text())
        self.config = self.report.get('config', self.report.get('scenario'))
        if self.evaluation_directory is not None:
            if self.report.get('index') != index or self.report.get('method') != evaluation_method:
                raise ValueError('evaluation index/method does not match the requested selection')
            initial = json.loads(initial_report.read_text())
            initial_config = initial.get('config', initial.get('scenario'))
            if self.config != initial_config:
                raise ValueError('evaluation scenario must match the initial sensor-data scenario')
        self.task = self.config['task']
        if self.task not in ('surface', 'inspection') or not self.trace:
            raise ValueError('a nonempty real Surface or Inspection trace is required')
        self.acceptance = acceptance_display(self.report, self.task)
        self.times = np.array([row['t'] for row in self.trace], dtype=float)
        if not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError('trace timestamps must be finite and strictly increasing')
        self.positions = np.asarray([row['position'] for row in self.trace], dtype=float)
        self.reference = np.asarray([row['reference_position'] for row in self.trace], dtype=float)
        self.tool = np.asarray([row['tool_pole'] for row in self.trace], dtype=float)
        self.force = np.asarray([row['normal_force_n'] for row in self.trace], dtype=float)
        self.tracking = np.linalg.norm(self.positions-self.reference, axis=1)
        self.operate = self.config.get('operate_interval', [20/3, 40/3])
        self.lidar = []
        with np.load(self.sources[3], allow_pickle=False) as archive:
            ranges = archive['history_ray_ranges']
            directions = archive['ray_directions']
            poses = archive['history_sensor_poses']
            timestamps = archive['history_times']
            if not (len(ranges) == len(poses) == len(timestamps)):
                raise ValueError('LiDAR history arrays do not align')
            self.anchor_time = float(timestamps[-1])
            for i, stamp in enumerate(timestamps):
                rays = directions[i] if directions.ndim == 3 else directions
                valid = np.isfinite(ranges[i]) & (ranges[i] >= .1) & (ranges[i] <= 40.)
                points = rays[valid]*ranges[i][valid, None]
                points = points @ poses[i, :3, :3].T+poses[i, :3, 3]
                self.lidar.append((float(stamp)-self.anchor_time, points))
        self.lidar_scope = 'Measured MID-360 preflight scan, frozen; no in-flight LiDAR in this archive'
        # Optional real flight scans are a separate, documented collection
        # artifact. We do not synthesize them from the task geometry.
        flight = self.directory/f'lidar_{index:03}.npz'
        if flight.exists():
            self.sources.append(flight)
            with np.load(flight, allow_pickle=False) as archive:
                stamps = archive['times']
                poses = archive['sensor_poses']
                ranges = archive['ray_ranges']
                directions = archive['ray_directions']
                for i, stamp in enumerate(stamps):
                    rays = directions[i] if directions.ndim == 3 else directions
                    valid = np.isfinite(ranges[i]) & (ranges[i] >= .1) & (ranges[i] <= 40.)
                    points = rays[valid]*ranges[i][valid, None]
                    self.lidar.append((float(stamp), points@poses[i, :3, :3].T+poses[i, :3, 3]))
            self.lidar.sort(key=lambda item: item[0])
            self.lidar_scope = 'Measured MID-360 latest causal scan; no future scan interpolation'
        if self.task == 'inspection':
            self.angle = np.array([row['measurement']['axis_angle_deg'] for row in self.trace])
            self.distance = np.array([row['measurement']['range_m'] for row in self.trace])

    def sample(self, time):
        # Strict preceding-state selection; never interpolate through a failure.
        index = min(max(int(np.searchsorted(self.times, time, side='right')-1), 0), len(self.times)-1)
        eligible = [item for item in self.lidar if item[0] <= time+1e-9]
        scan_time, points = eligible[-1] if eligible else (-np.inf, np.empty((0, 3)))
        return index, scan_time, points

    def manifest(self):
        return dict(directory=str(self.directory.resolve()), index=self.index, method=self.method,
                    task=self.task, acceptance=self.acceptance, recorded_seconds=float(self.times[-1]),
                    evaluation_directory=str(self.evaluation_directory.resolve()) if self.evaluation_directory else None,
                    evaluation_method=self.evaluation_method,
                    trajectory_source='evaluation report/trace' if self.evaluation_directory else 'collected expert report/trace',
                    sensor_source='collection NPZ contains measured initial sensing; not used as the flight trajectory',
                    selection_scope='fixed condition index; this example is not an aggregate success estimate',
                    lidar_scope=self.lidar_scope,
                    source_sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in self.sources})


def _box_faces(center, half):
    center, half = np.array(center), np.array(half)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])*half+center
    return [corners[list(indices)] for indices in ((0,1,3,2),(4,5,7,6),(0,1,5,4),
                                                  (2,3,7,6),(0,2,6,4),(1,3,7,5))]


class Renderer:
    def __init__(self, episode):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from matplotlib.patches import Rectangle
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        self.episode = episode
        self.plt = plt
        plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                             'axes.facecolor':'#f4f7fb', 'axes.edgecolor':'#cad3df',
                             'axes.labelcolor':'#30465f', 'xtick.color':'#476079', 'ytick.color':'#476079'})
        self.fig = plt.figure(figsize=(12.8,7.2),dpi=100,facecolor='#ffffff')
        self.ax3 = self.fig.add_axes([.015,.285,.49,.51], projection='3d')
        self.ax2 = self.fig.add_axes([.56,.325,.405,.465])
        self.axa = self.fig.add_axes([.065,.105,.415,.13])
        self.axb = self.fig.add_axes([.56,.105,.405,.13])
        self.fig.text(.035,.956,'TASK-ADAPTIVE FLIGHT  /  PHYSICAL TELEMETRY',fontsize=17,
                      fontweight='bold',color='#172c46')
        verdict = episode.acceptance['verdict']
        self.fig.text(.965,.958,verdict,ha='right',fontsize=20,fontweight='bold',
                      color={'PASS':'#167548','SOFT PASS':'#167548','PARTIAL':'#b77812','FAIL':'#be303f'}[verdict])
        self.header = self.fig.text(.035,.921,'',fontsize=11,color='#30465f')
        self.fig.text(.035,.894,episode.acceptance['details'],fontsize=10,
                      color='#8f6418' if episode.task=='surface' else '#476079')
        scope='SCHEMATIC TELEMETRY REPLAY  |  Measured actuator-driven states, not RGB rendering'
        if episode.task=='surface':scope+='  |  Soft-contact model uncalibrated'
        self.fig.text(.035,.868,scope,
                      fontsize=9,color='#6c7787')
        self.status = self.fig.text(.035,.035,'',fontsize=10,color='#172c46')
        self.scan_label = self.fig.text(.035,.842,'',fontsize=8,color='#33669a')
        self.ax3.set_title('Measured body / tool / sensor geometry',pad=0)
        self.ax2.set_title('Plan view  |  task geometry and measured rays',loc='left')
        positions = np.vstack((episode.positions, episode.reference, episode.tool))
        if episode.task == 'inspection':
            positions = np.vstack((positions, episode.config['inspection_target']))
        self.low, self.high = positions.min(axis=0)-[.6,.65,.25], positions.max(axis=0)+[.6,.65,.4]
        self.low[2] = max(0., self.low[2])
        self.ax3.set(xlim=(self.low[0],self.high[0]), ylim=(self.low[1],self.high[1]),
                     zlim=(self.low[2],self.high[2]),xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]')
        self.ax3.set_box_aspect(np.maximum(self.high-self.low,[1,1,1]))
        self.ax3.view_init(elev=27, azim=-64)
        self.ax2.set(xlim=(self.low[0],self.high[0]), ylim=(self.low[1],self.high[1]),
                     xlabel='X [m]',ylabel='Y [m]')
        self.ax2.set_aspect('equal',adjustable='box'); self.ax2.grid(alpha=.22)
        for box in episode.config.get('boxes',[]):
            center, half = np.array(box[:3]),np.array(box[3:])/2
            if np.any(center[:2]+half[:2]<self.low[:2]) or np.any(center[:2]-half[:2]>self.high[:2]):
                continue
            if center[2]+half[2] < self.low[2]:
                continue
            self.ax3.add_collection3d(Poly3DCollection(_box_faces(center,half),facecolor='#8b98a8',
                                                     alpha=.18,edgecolor='#718098',linewidth=.4))
            self.ax2.add_patch(Rectangle(center[:2]-half[:2],2*half[0],2*half[1],
                                        facecolor='#b9c3d0',alpha=.5))
        if episode.task=='surface':
            center=np.array(episode.config['pad_center']); half=np.array(episode.config['pad_half_extents'])
            self.ax3.add_collection3d(Poly3DCollection(_box_faces(center,half),facecolor='#b0ceb7',
                                                     alpha=.5,edgecolor='#679578',linewidth=.5))
            self.ax2.add_patch(Rectangle(center[:2]-half[:2],2*half[0],2*half[1],
                                        facecolor='#dcebe0',edgecolor='#679578',alpha=.65))
            patch=episode.config.get('keepout_in_pad_xy')
            if patch is not None:
                x0,x1,y0,y1=patch
                self.ax2.add_patch(Rectangle(center[:2]+[x0,y0],x1-x0,y1-y0,
                                            facecolor='#f2bfc3',edgecolor='#b24754',alpha=.8,hatch='//'))
                vertices=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])+center[:2]
                vertices=np.column_stack((vertices,np.full(4,center[2]+half[2]+.002)))
                self.ax3.add_collection3d(Poly3DCollection([vertices],facecolor='#d25464',alpha=.55))
        else:
            target=np.array(episode.config['inspection_target'])
            self.ax3.scatter(*target,c='#dba01a',marker='*',s=120,zorder=20)
            self.ax2.scatter(*target[:2],c='#dba01a',marker='*',s=120,label='inspection target')
        self.ax3.plot(*episode.reference.T,color='#808c9a',ls='--',lw=1,label='commanded reference')
        self.ax2.plot(*episode.reference[:,:2].T,color='#808c9a',ls='--',lw=1,label='commanded reference')
        self.path3,=self.ax3.plot([],[],[],color='#126ca6',lw=2.1)
        self.path2,=self.ax2.plot([],[],color='#126ca6',lw=2.1,label='measured body')
        self.tool3,=self.ax3.plot([],[],[],color='#b55b28',lw=1.4)
        self.body3,=self.ax3.plot([],[],[],color='#172c46',lw=3,marker='o',markersize=3)
        self.body2,=self.ax2.plot([],[],color='#172c46',marker='o',lw=3,markersize=3)
        self.sensor3,=self.ax3.plot([],[],[],color='#9b43a5',lw=2)
        self.sensor2,=self.ax2.plot([],[],color='#9b43a5',lw=2)
        self.sight3,=self.ax3.plot([],[],[],color='#168856',lw=1.4,ls=':')
        self.sight2,=self.ax2.plot([],[],color='#168856',lw=1.4,ls=':')
        self.force3,=self.ax3.plot([],[],[],color='#d93345',lw=3)
        self.cloud3=self.ax3.scatter([],[],[],s=1,alpha=.22,color='#44a5bf')
        self.cloud2=self.ax2.scatter([],[],s=1,alpha=.3,color='#44a5bf',label='measured MID-360')
        self.ax2.legend(loc='upper left',fontsize=7,framealpha=.8)
        self.metric_a,=self.axa.plot([],[],color='#185f9a',lw=1.5)
        self.metric_b,=self.axb.plot([],[],color='#185f9a',lw=1.5)
        self.cursor_a=self.axa.axvline(0,color='#c7394d',lw=1)
        self.cursor_b=self.axb.axvline(0,color='#c7394d',lw=1)
        for ax in (self.axa,self.axb):
            ax.set_xlim(0,20); ax.axvspan(*episode.operate,color='#b6d2ba',alpha=.32)
            ax.grid(alpha=.2); ax.set_xlabel('episode time [s]',fontsize=8)
        if episode.task=='surface':
            self.axa.set_ylim(0,max(2.,float(episode.force.max())*1.1))
            self.axa.axhspan(.5,1.5,color='#99c79d',alpha=.3)
            self.axa.set_title('Measured normal contact force [N]  |  band: 0.5-1.5',fontsize=9,loc='left')
            self.axb.set_ylim(0,max(.16,float(episode.tracking.max())*1.1))
            self.axb.axhline(.15,color='#b74651',ls='--',lw=.8)
            self.axb.set_title('Measured body tracking error [m]',fontsize=9,loc='left')
        else:
            self.axa.set_ylim(0,max(15.,min(180.,float(episode.angle.max())*1.05)))
            self.axa.axhspan(0,10,color='#99c79d',alpha=.3)
            self.axa.set_title('Measured FPV optical-axis error [deg]  |  limit: 10',fontsize=9,loc='left')
            self.axb.set_ylim(0,max(3.,float(episode.distance.max())*1.05))
            self.axb.axhspan(*episode.config.get('inspection_range',[1.7,2.5]),color='#99c79d',alpha=.3)
            self.axb.set_title('Measured camera-to-target distance [m]',fontsize=9,loc='left')
        self.last_scan=None

    def frame(self, time):
        from scipy.spatial.transform import Rotation
        episode=self.episode
        i, scan_time, points=episode.sample(time)
        row=episode.trace[i]; current=float(episode.times[i])
        active='OPERATE' if episode.operate[0]<=current<=episode.operate[1] else row.get('phase','transit').upper()
        stopped=time>episode.times[-1]+.02
        self.header.set_text(f'{episode.task.upper()}  |  {episode.method}  |  episode {episode.index:03d}  |  '
                             f't={current:05.2f} / 20.00 s  |  {active}')
        self.path3.set_data_3d(*episode.positions[:i+1].T)
        self.path2.set_data(*episode.positions[:i+1,:2].T)
        rotation=Rotation.from_quat(np.array(row['quaternion'])[[1,2,3,0]]).as_matrix()
        position=episode.positions[i]
        body=position[None]+np.array([[0,-.42,0],[0,.42,0],[0,0,0],[.35,0,0]])@rotation.T
        self.body3.set_data_3d(*body.T);self.body2.set_data(*body[:,:2].T)
        tool=np.vstack((position,episode.tool[i]));self.tool3.set_data_3d(*tool.T)
        if episode.task=='surface':
            force=np.vstack((episode.tool[i],episode.tool[i]+[0,0,.25*episode.force[i]]))
            self.force3.set_data_3d(*force.T)
            camera=position+rotation@np.array([.24,0,.025]); axis=rotation[:,0]
            self.status.set_text(f'Force {episode.force[i]:.2f} N   |   tool contact: {bool(row["allowed_contact"])}   |   '
                                 f'non-tool collisions: {episode.report.get("forbidden_contact_steps",0)} steps   |   '
                                 f'tracking {1000*episode.tracking[i]:.1f} mm')
            self.metric_a.set_data(episode.times[:i+1],episode.force[:i+1])
            self.metric_b.set_data(episode.times[:i+1],episode.tracking[:i+1])
        else:
            view=row['measurement'];camera=np.array(view['camera_position']);axis=np.array(view['optical_axis'])
            sight=np.vstack((camera,view['target_position']))
            color='#168856' if view['visible'] else '#b77a23'
            self.sight3.set_data_3d(*sight.T);self.sight3.set_color(color)
            self.sight2.set_data(*sight[:,:2].T);self.sight2.set_color(color)
            self.status.set_text(f'Actual ray first hit: {view["first_hit_geom"]}   |   visible: {view["visible"]}   |   '
                                 f'effective: {view["effective"]}   |   single-point LOS; no image recognition')
            self.metric_a.set_data(episode.times[:i+1],episode.angle[:i+1])
            self.metric_b.set_data(episode.times[:i+1],episode.distance[:i+1])
        sensor=np.vstack((camera,camera+.7*axis))
        self.sensor3.set_data_3d(*sensor.T);self.sensor2.set_data(*sensor[:,:2].T)
        for cursor in (self.cursor_a,self.cursor_b):cursor.set_xdata([current,current])
        if self.last_scan != scan_time:
            visible=np.all((points>=self.low)&(points<=self.high),axis=1)
            points=points[visible]
            points=points[::max(1,int(np.ceil(len(points)/1800)))]
            self.cloud3._offsets3d=tuple(points.T)
            self.cloud2.set_offsets(points[:,:2])
            self.last_scan=scan_time
        self.scan_label.set_text(f'MID-360 scan t={scan_time:.2f} s relative to flight start  |  '+episode.lidar_scope)
        if stopped:
            self.status.set_text(f'EXECUTION STOPPED at {current:.2f} s. Last recorded state shown; no further motion synthesized.')
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:,:,:3].copy()

    def close(self):
        self.plt.close(self.fig)


def export(surface, inspection, output, *, ffmpeg='ffmpeg', fps=10):
    """Export both explicitly selected task episodes, retaining failed verdicts."""
    from PIL import Image
    if surface.task!='surface' or inspection.task!='inspection':
        raise ValueError('video must contain one explicit Surface and one Inspection selection')
    if fps<=0 or fps>30:raise ValueError('fps must be in 1..30')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    program=shutil.which(str(ffmpeg))
    if program is None:raise FileNotFoundError(ffmpeg)
    command=[program,'-hide_banner','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
             '-pix_fmt','rgb24','-s','1280x720','-r',str(fps),'-i','-','-an','-c:v','libx264',
             '-preset','fast','-crf','20','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(output)]
    posters=[];frame_count=0
    with subprocess.Popen(command,stdin=subprocess.PIPE) as encoder:
        try:
            for episode in (surface,inspection):
                renderer=Renderer(episode)
                try:
                    poster=Image.fromarray(renderer.frame(10.));posters.append(poster)
                    poster.save(output.with_name(output.stem+f'.{episode.task}.png'))
                    for frame in range(20*fps):
                        encoder.stdin.write(renderer.frame((frame+1)/fps).tobytes());frame_count+=1
                    print(json.dumps(dict(task=episode.task,frames=20*fps,acceptance=episode.acceptance)),flush=True)
                finally:renderer.close()
            encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError('ffmpeg encoding failed')
        except BaseException:
            encoder.kill();encoder.wait();raise
    poster=Image.new('RGB',(1280,1440),'white')
    poster.paste(posters[0],(0,0));poster.paste(posters[1],(0,720))
    poster_path=output.with_suffix('.poster.png');poster.save(poster_path)
    manifest=dict(schema='hnuter_task_telemetry_video_v1',format='schematic telemetry replay; no RGB rendering',
                  fps=fps,frames=frame_count,duration_s=frame_count/fps,resolution=[1280,720],
                  episodes=[surface.manifest(),inspection.manifest()],poster=str(poster_path.resolve()),
                  video_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for task in ('surface','inspection'):
        parser.add_argument(f'--{task}-directory',type=Path,required=True)
        parser.add_argument(f'--{task}-index',type=int,required=True)
        parser.add_argument(f'--{task}-method',default='Expert')
        parser.add_argument(f'--{task}-evaluation-method',default='adapter_raw')
    parser.add_argument('--evaluation-directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fps',type=int,default=10)
    parser.add_argument('--ffmpeg',default='ffmpeg')
    args=parser.parse_args()
    surface=Episode(args.surface_directory,args.surface_index,args.surface_method,
                    evaluation_directory=args.evaluation_directory,
                    evaluation_method=args.surface_evaluation_method if args.evaluation_directory else None)
    inspection=Episode(args.inspection_directory,args.inspection_index,args.inspection_method,
                       evaluation_directory=args.evaluation_directory,
                       evaluation_method=args.inspection_evaluation_method if args.evaluation_directory else None)
    print(json.dumps(export(surface,inspection,args.output,ffmpeg=args.ffmpeg,fps=args.fps),indent=2))


if __name__=='__main__':main()
