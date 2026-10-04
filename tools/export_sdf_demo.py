#!/usr/bin/env python3
"""Compose captured RGB, measured LiDAR and timestamp-matched nvblox diagnostics.

No simulation or scene geometry is used during export. All frames are retained,
including history warmup and terminal hold. Playback follows the simulation clock.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_flight_video import validate_manifest, validate_flight_identity, locate_ffmpeg, digest

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
BG=(10,17,28);FG=(228,237,246);MUTED=(147,166,184)


def text(draw,xy,value,size=20,color=FG,bold=False):
    draw.text(xy,value,font=ImageFont.truetype(BOLD if bold else FONT,size),fill=color)


def axes(draw,rect,bounds):
    x0,y0,w,h=rect;xmin,xmax,ymin,ymax=bounds
    for x in np.arange(np.ceil(xmin),xmax+1,1):
        px=x0+(x-xmin)/(xmax-xmin)*w
        draw.line((px,y0,px,y0+h),fill=(37,50,66))
        if x%2==0:text(draw,(px-8,y0+h+4),f'{x:g}',14,MUTED)
    for y in np.arange(np.ceil(ymin),ymax+1,1):
        py=y0+h-(y-ymin)/(ymax-ymin)*h
        draw.line((x0,py,x0+w,py),fill=(37,50,66))
        if y%2==0:text(draw,(x0-30,py-8),f'{y:g}',14,MUTED)
    def pixel(points):
        p=np.asarray(points)
        return np.column_stack((x0+(p[:,0]-xmin)/(xmax-xmin)*w,y0+h-(p[:,1]-ymin)/(ymax-ymin)*h))
    return pixel


def path_overlay(draw,pixel,positions,goal,planned):
    if planned is not None:
        draw.line([tuple(p) for p in pixel(planned)],fill=(85,202,245),width=2)
    if len(positions)>1:draw.line([tuple(p) for p in pixel(positions)],fill=(245,250,254),width=3)
    p=pixel(positions[-1:])[0];draw.ellipse((p[0]-6,p[1]-6,p[0]+6,p[1]+6),fill=(255,255,255))
    g=pixel(np.array(goal)[None])[0]
    draw.line((g[0]-8,g[1],g[0]+8,g[1]),fill=(255,197,67),width=3)
    draw.line((g[0],g[1]-8,g[0],g[1]+8),fill=(255,197,67),width=3)


def export(run,output,ffmpeg=None):
    run=Path(run).resolve();output=Path(output).resolve()
    if output.exists() or output.with_suffix('.json').exists():raise ValueError('preserve existing export')
    main=validate_manifest(run/'video/manifest.json')
    flight=validate_flight_identity(run/'video/manifest.json',main,run/'report.json')
    side_path=run/'perception_video/manifest.json';side=json.loads(side_path.read_text())
    if digest(side_path)!=flight['perception_video']['manifest_sha256']:raise ValueError('perception manifest mismatch')
    other=side['camera']['camera'];secondary=validate_manifest(run/f'perception_video/{other}/manifest.json')
    if digest(run/f'perception_video/{other}/manifest.json')!=side['camera']['manifest_sha256']:raise ValueError('secondary identity mismatch')
    if len(main['frames'])!=len(side['frames']) or len(main['frames'])!=len(secondary['frames']):raise ValueError('unequal capture lengths')
    origin=np.array(flight['initial_position']);goal=np.array(flight['goal_position'])-origin
    positions=np.array([f['position'] for f in main['frames']])-origin
    plan=next((p for p in flight['planner']['plans'] if p['status']=='accepted'),None)
    planned=None
    if plan:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from hnuter_dit.sdf_pose import PoseSpline
        from scipy.spatial.transform import Rotation
        rotation=Rotation.from_quat(np.array(main['frames'][0]['quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
        planned=PoseSpline(np.array(plan['controls']),origin,start_rotation=rotation).sample(np.linspace(0,20,100))['pos']-origin
    executable=locate_ffmpeg(ffmpeg)
    output.parent.mkdir(parents=True,exist_ok=True)
    command=[executable,'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1200',
        '-r',str(main['fps']),'-i','-','-an','-c:v','libx264','-preset','medium','-threads','2','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    map_hashes={};unknown_frames=0
    try:
        for i,(frame,second,record) in enumerate(zip(main['frames'],secondary['frames'],side['frames'])):
            t=frame['simulation_time']
            if second['physics_step']!=frame['physics_step'] or record['physics_step']!=frame['physics_step']:raise ValueError('unsynchronized capture')
            canvas=Image.new('RGB',(1920,1200),BG);draw=ImageDraw.Draw(canvas)
            text(draw,(24,12),'HNUTER | GPU distance-field history + DiT | static obstacle flight',28,bold=True)
            text(draw,(1500,18),f'SIM {t:05.1f} / 24.0 s',24,(99,220,186),True)
            for x,label,path in ((0,main['camera'].upper(),run/'video'/frame['filename']),
                                  (960,other.upper(),run/'perception_video'/other/second['filename'])):
                canvas.paste(Image.open(path).convert('RGB'),(x,56))
                draw=ImageDraw.Draw(canvas);draw.rectangle((x+12,68,x+195,101),fill=BG)
                text(draw,(x+24,72),label+' CAMERA',18,bold=True)
            text(draw,(24,613),'MID-360 measured returns | local XY, height color',23,bold=True)
            text(draw,(984,613),'nvblox GPU ESDF | observed-distance slice',23,bold=True)
            left=(58,680,490,360);right=(1025,665,390,390)
            pixel=axes(draw,left,(-3,8,-5,5));map_pixel=axes(draw,right,(-2,6,-4,4))
            scan=record['scan']
            if scan:
                path=side_path.parent/scan['filename']
                if digest(path)!=scan['sha256']:raise ValueError('scan hash mismatch')
                with np.load(path) as z:
                    if abs(float(z['scan_time'])-scan['time'])>1e-8 or scan['time']>t+1e-8:raise ValueError('scan timestamp mismatch')
                    points=z['points_world']-origin
                valid=(points[:,0]>-3)&(points[:,0]<8)&(points[:,1]>-5)&(points[:,1]<5)
                points=points[valid];pix=np.rint(pixel(points)).astype(int)
                colors=(colormaps['turbo'](np.clip((points[:,2]+origin[2])/8,0,1))[:,:3]*255).astype(np.uint8)
                # Rasterize only measured returns. No obstacle geometry is loaded.
                pixels=np.asarray(canvas).copy()
                pixels[pix[:,1],pix[:,0]]=colors
                pixels[pix[:,1]+1,pix[:,0]]=colors
                canvas=Image.fromarray(pixels);draw=ImageDraw.Draw(canvas)
                text(draw,(580,686),f"Scan: {scan['time']:.3f} s",20)
                text(draw,(580,720),f"Age: {(t-scan['time'])*1000:.0f} ms (sim)",20)
                text(draw,(580,754),f"Returns: {scan['hits']:,}",20)
            else:text(draw,(580,686),'Waiting for first scan',20)
            seq=record['map_sequence']
            if seq is not None:
                path=run/'planner_worker'/Path(flight['planner']['checkpoint']['worker_directory']).name/'visualization'/f'{seq:06d}.npz'
                map_hashes[str(path.relative_to(run))]=digest(path)
                with np.load(path) as z:
                    stamp=float(z['stamp']);esdf=z['esdf_m'];known=z['known'];height=float(z['z_world'])
                    if abs(stamp-record['map_scan_time'])>1e-8 or stamp>t+1e-8:raise ValueError('map is newer than displayed state')
                    if not np.allclose(z['origin'],origin):raise ValueError('map anchor mismatch')
                rgb=(colormaps['viridis'](np.clip(esdf/3,0,1))[:,:,:3]*255).astype(np.uint8)
                rgb[~known]=(26,35,47)
                canvas.paste(Image.fromarray(rgb.transpose(1,0,2)[::-1]).resize((390,390),Image.Resampling.NEAREST),(1025,665));draw=ImageDraw.Draw(canvas)
                text(draw,(1460,686),f'Map scan: {stamp:.3f} s',20)
                text(draw,(1460,720),f'Horizontal slice z = {height:.2f} m',20)
                text(draw,(1460,754),f'Observed cells: {100*known.mean():.1f}%',20)
                text(draw,(1460,795),'Dark cells = UNKNOWN',21,(252,190,95),True)
                text(draw,(1460,828),'Unknown is not certified free.',18,MUTED)
                text(draw,(1460,875),f"Mapping: {record['map_seconds']*1000:.1f} ms",20)
                text(draw,(1460,909),f"Sensor response: {record['sensor_response_seconds']*1000:.1f} ms",20)
            else:
                unknown_frames+=1;text(draw,(1460,686),'Waiting for mapped observation',20)
            active=planned if plan and t>=plan['activation_time'] else None
            path_overlay(draw,pixel,positions[:i+1],goal,active)
            path_overlay(draw,map_pixel,positions[:i+1],goal,active)
            for j in range(280):
                c=tuple(int(v) for v in np.array(colormaps['turbo'](j/279)[:3])*255)
                draw.line((580+j,840,580+j,856),fill=c)
                c=tuple(int(v) for v in np.array(colormaps['viridis'](j/279)[:3])*255)
                draw.line((1460+j,962,1460+j,978),fill=c)
            text(draw,(580,865),'Height: 0 m                      8 m',17,MUTED)
            text(draw,(1460,984),'Distance: 0 m                    3 m+',17,MUTED)
            text(draw,(580,920),'White: measured flight',19)
            text(draw,(580,951),'Blue: accepted DiT plan',19,(85,202,245))
            text(draw,(580,982),'Gold: requested pose location',19,(255,197,67))
            draw.rectangle((0,1085,1920,1200),fill=(18,29,43))
            text(draw,(24,1100),f"Goal error {frame['goal_distance_m']:.3f} m | contacts {'DETECTED' if frame['collision_detected'] else 'none'} | {frame['planner_status']} | accepted plans {record['accepted_plans']}",23,bold=True)
            text(draw,(24,1140),f"Actual physics + sensor records | 1x SIM playback | {flight['simulation_seconds']:.1f} sim s recorded in {flight['loop_wall_seconds']:.1f} wall s (startup excluded).",20,MUTED)
            if i in (0,10,80,150,len(main['frames'])-1):canvas.save(output.parent/f'preview_{i:04d}.png')
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        if process.wait():raise RuntimeError('ffmpeg failed')
    except BaseException:
        if process.poll() is None:process.terminate();process.wait()
        raise
    evidence=dict(schema='hnuter_sdf_demo_mp4_v1',mp4_sha256=digest(output),frame_count=len(main['frames']),fps=main['fps'],
        encoded_seconds=len(main['frames'])/main['fps'],simulation_seconds=main['last_simulation_time'],resolution=[1920,1200],
        recording_loop_wall_seconds=flight['loop_wall_seconds'],physics_report_sha256=digest(run/'report.json'),camera_manifest_sha256=digest(run/'video/manifest.json'),
        perception_manifest_sha256=digest(side_path),map_snapshot_sha256=map_hashes,unknown_initial_frames=unknown_frames,
        checkpoint_sha256=flight['planner']['checkpoint']['checkpoint_sha256'],passed=flight['passed'],
        final_goal_distance_m=flight['final_goal_distance_m'],collision=flight['collision'],
        source='Actual in-run rendered cameras, timestamped measured raycasts and actual nvblox ESDF; no pose replay, interpolation or synthetic obstacle cloud.',
        playback='Every recorded frame from 0 to 24 simulation seconds retained; 1x simulation clock, not a wall-clock real-time claim.')
    output.with_suffix('.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({k:v for k,v in evidence.items() if k!='map_snapshot_sha256'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output',required=True);parser.add_argument('--ffmpeg')
    args=parser.parse_args();export(args.run,args.output,args.ffmpeg)
