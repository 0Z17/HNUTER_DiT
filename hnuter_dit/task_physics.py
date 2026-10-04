"""Native 1 kHz task execution; explicit tool and measured solver contacts.

The force controller is a separately reported position-reference outer loop.
It does not alter the inherited aircraft controller, gains, actuator allocation,
or integrated body pose. Task-plane geometry is an external task command.
"""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import numpy as np

from .temporal_data import NativeWorld, DEFAULT_SIM_ROOT, _module, DT


TOOL_POLE = np.array([0., 0., -.75])
TIP_RADIUS = .025
TIP_CENTER = np.array([0., 0., -.725])
DEFAULT_FORCE_CONTROL = dict(target_n=.8, proportional_m_per_n=.002,
    integral_m_per_ns=.004, maximum_offset_m=.035, maximum_rate_mps=.025,
    filter_seconds=.03, release_seconds=.35)


def _numbers(values):
    return ' '.join(str(float(value)) for value in values)


class TaskWorld(NativeWorld):
    """Unmodified aircraft plus a separately inertialized downward tool.

    Pad/obstacles are fixed physical collision geometry. Only the tip/pad pair,
    on the pad top face during engage/operate/release, is permitted. Surface
    task pose is supplied by the task; this does not imply MID-360 observes the
    platform directly below the aircraft.
    """
    def __init__(self, config, sim_root=DEFAULT_SIM_ROOT):
        import mujoco
        self.mj, self.scenario, self.sim_root = mujoco, copy.deepcopy(config), Path(sim_root)
        self.task = config.get('task','surface')
        if self.task not in ('surface','inspection'):
            raise ValueError('task must be surface or inspection')
        self.scenario.setdefault('origin', [0.,0.,2.8])
        self.scenario.setdefault('pad_center', [2.2,0.,1.75])
        self.scenario.setdefault('pad_half_extents', [3.,1.5,.05])
        self.scenario.setdefault('tool_mass_kg', .02)
        self.force_cfg = dict(DEFAULT_FORCE_CONTROL, **config.get('force_control', {}))
        # Frozen after development case contact_v11. This highly compliant
        # simulation material is explicitly uncalibrated; report penetration.
        self.contact_material = dict(friction=[.20,.005,.0001],solref=[5.,1],solimp=[.9,.95,.001])
        self.contact_material.update(config.get('contact_material',{}))
        self.pad_center = np.asarray(self.scenario['pad_center'], dtype=float)
        self.pad_half = np.asarray(self.scenario['pad_half_extents'], dtype=float)
        if (self.pad_center.shape != (3,) or self.pad_half.shape != (3,) or
                not np.isfinite(np.r_[self.pad_center, self.pad_half]).all() or
                np.any(self.pad_half <= 0)):
            raise ValueError('finite positive pad geometry required')
        mass = float(self.scenario['tool_mass_kg'])
        if not 0 < mass <= .1:
            raise ValueError('pilot tool mass must be explicit and in (0,.1] kg')
        if any(not np.isfinite(v) or v <= 0 for v in self.force_cfg.values()):
            raise ValueError('positive finite force outer-loop parameters required')
        source = self.sim_root/'robots/hnuter/source/hnuter.xml'
        root = ET.parse(source).getroot()
        root.find('compiler').set('meshdir', str(source.parent/'meshes'))
        drone = root.find("worldbody/body[@name='drone']")
        # The reference point is the lower sphere pole, not its centre.
        shaft_start, shaft_end, shaft_radius = -.30, -.70, .010
        shaft_mass, ball_mass = .75*mass, .25*mass
        shaft_z = .5*(shaft_start+shaft_end)
        com_z = (shaft_mass*shaft_z+ball_mass*TIP_CENTER[2])/mass
        length = shaft_start-shaft_end
        i_xy = (shaft_mass*(3*shaft_radius**2+length**2)/12 +
                shaft_mass*(shaft_z-com_z)**2 + .4*ball_mass*TIP_RADIUS**2 +
                ball_mass*(TIP_CENTER[2]-com_z)**2)
        i_z = .5*shaft_mass*shaft_radius**2 + .4*ball_mass*TIP_RADIUS**2
        tool = ET.SubElement(drone, 'body', name='task_probe')
        ET.SubElement(tool, 'inertial', pos=f'0 0 {com_z}', mass=str(mass),
                      diaginertia=_numbers([i_xy, i_xy, i_z]))
        common = dict(contype='2', conaffinity='1', group='5',
                      **{key:_numbers(value) for key,value in self.contact_material.items()})
        ET.SubElement(tool, 'geom', name='probe_shaft', type='capsule',
                      fromto=f'0 0 {shaft_start} 0 0 {shaft_end}', size=str(shaft_radius), **common)
        ET.SubElement(tool, 'geom', name='probe_tip', type='sphere',
                      pos=_numbers(TIP_CENTER), size=str(TIP_RADIUS), **common)
        ET.SubElement(tool, 'site', name='probe_contact_pole', pos=_numbers(TOOL_POLE), size='.003')
        if self.task == 'surface':
            ET.SubElement(root.find('worldbody'), 'geom', name='task_pad', type='box',
                          pos=_numbers(self.pad_center), size=_numbers(self.pad_half),
                          contype='1', conaffinity='1', group='0',
                          **{key:_numbers(value) for key,value in self.contact_material.items()})
        if config.get('inspection_target') is not None:
            ET.SubElement(root.find('worldbody'), 'geom', name='inspection_target', type='sphere',
                          pos=_numbers(config['inspection_target']), size=str(config.get('target_radius',.05)),
                          contype='1',conaffinity='1',group='0')
        for index, box in enumerate(config.get('boxes', [])):
            ET.SubElement(root.find('worldbody'), 'geom', name=f'task_obstacle_{index}', type='box',
                          pos=_numbers(box[:3]), size=_numbers(np.asarray(box[3:])/2),
                          contype='1', conaffinity='1', group='0')
        self._directory = tempfile.TemporaryDirectory(prefix='hnuter-contact-task-')
        path = Path(self._directory.name)/'world.xml'
        self.xml_source = ET.tostring(root, encoding='unicode')
        path.write_text(self.xml_source)
        cls = _module(self.sim_root/'tests/reference/hnuter_control.py', '_task_native_hnuter').HnuterController
        self.controller = cls(path)
        self.model, self.data = self.controller.model, self.controller.data
        if not np.isclose(self.model.opt.timestep, DT):
            raise RuntimeError('Task requires original 1 ms integration')
        # Initialization only; all subsequent motion is actuator driven.
        self.controller.set_freejoint_pose(self.scenario['origin'], config.get('initial_quaternion', [1,0,0,0]))
        self.robot_geoms = self.model.body_rootid[self.model.geom_bodyid] == self.controller._body_id
        self.model.geom_group[self.robot_geoms] = 5
        self.tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'probe_tip')
        self.pad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'task_pad')
        mujoco.mj_forward(self.model, self.data)
        self.offset = self.integral = self.filtered_force = 0.
        self.last_measurement = None
        self.steps = self.allowed_steps = self.forbidden_steps = self.saturated_steps = 0
        self.saturation_streak = self.maximum_saturation_streak = 0
        self.first_forbidden = None
        self.normal_forces, self.tangent_forces = [], []
        self.maximum_any_contact_force = self.maximum_penetration = 0.

    @property
    def pad_top(self):
        return float(self.pad_center[2]+self.pad_half[2])

    def _command(self, desired, force_enabled):
        command = {key: np.array(value, copy=True) for key,value in desired.items()}
        actual_force = 0. if self.last_measurement is None else self.last_measurement['normal_force_n']
        cfg = self.force_cfg
        self.filtered_force += DT/(cfg['filter_seconds']+DT)*(actual_force-self.filtered_force)
        if force_enabled:
            error = cfg['target_n']-self.filtered_force
            proposed_integral = self.integral+cfg['integral_m_per_ns']*error*DT
            target = np.clip(proposed_integral+cfg['proportional_m_per_n']*error,
                             -cfg['maximum_offset_m'], cfg['maximum_offset_m'])
            # Conditional integration prevents windup at the bounded displacement.
            if ((-cfg['maximum_offset_m'] < target < cfg['maximum_offset_m']) or
                    (target <= -cfg['maximum_offset_m'] and error > 0) or
                    (target >= cfg['maximum_offset_m'] and error < 0)):
                self.integral = proposed_integral
        else:
            self.integral = 0.
            target = self.offset-float(np.clip(self.offset,-DT*cfg['maximum_offset_m']/cfg['release_seconds'],
                                               DT*cfg['maximum_offset_m']/cfg['release_seconds']))
        self.offset += float(np.clip(target-self.offset,
                              -cfg['maximum_rate_mps']*DT, cfg['maximum_rate_mps']*DT))
        command['pos'][2] -= self.offset
        return command

    def _contacts(self, phase):
        allowed = []
        forbidden = []
        self.raw_cross_contacts = []
        normal = tangent = 0.
        for contact_id, contact in enumerate(self.data.contact[:self.data.ncon]):
            a, b = int(contact.geom1), int(contact.geom2)
            if a < 0 or b < 0 or self.robot_geoms[a] == self.robot_geoms[b]:
                continue
            force = np.zeros(6)
            self.mj.mj_contactForce(self.model, self.data, contact_id, force)
            self.raw_cross_contacts.append(dict(geom_ids=[a,b],distance_m=float(contact.dist),
                efc_address=int(contact.efc_address),force_contact_frame=force.tolist()))
            # Positive solver margins are not geometric touching; retain all
            # such records and their solver force for auditing this decision.
            if float(contact.dist) > 0:
                continue
            position = np.asarray(contact.pos).copy()
            normal_axis = np.asarray(contact.frame).reshape(3,3)[0]
            on_top = (abs(position[2]-self.pad_top) <= .01 and
                      np.all(np.abs(position[:2]-self.pad_center[:2]) <= self.pad_half[:2]-.005) and
                      abs(normal_axis[2]) > .995)
            entry = dict(pair=[self.mj.mj_id2name(self.model,self.mj.mjtObj.mjOBJ_GEOM,g) for g in (a,b)],
                         distance_m=float(contact.dist), position=position.tolist(),
                         force_contact_frame=force.tolist())
            if ({a,b} == {self.tip_id,self.pad_id} and on_top and phase in ('engage','operate','release')):
                allowed.append(entry)
                normal += max(0., float(force[0]))
                tangent += float(np.linalg.norm(force[1:3]))
            else:
                forbidden.append(entry)
        return allowed, forbidden, normal, tangent

    def step_task(self, desired, phase='free', force_enabled=False):
        if phase not in ('free','approach','engage','operate','release','retreat'):
            raise ValueError('unknown task execution phase')
        if force_enabled and self.task != 'surface':
            raise ValueError('force feedback requires a surface task')
        command = self._command(desired, force_enabled)
        self.controller.set_desired_state(command)
        self.controller.update_control()
        ctrl = self.data.ctrl.copy()
        limited = self.model.actuator_ctrllimited.astype(bool)
        ranges = self.model.actuator_ctrlrange
        saturated = limited & ((ctrl <= ranges[:,0]+1e-6) | (ctrl >= ranges[:,1]-1e-6))
        self.mj.mj_step(self.model,self.data)
        self.mj.mj_forward(self.model,self.data)
        state = self.state()
        if not all(np.isfinite(value).all() for value in state.values()):
            raise RuntimeError('non-finite task physical state')
        allowed, forbidden, normal, tangent = self._contacts(phase)
        raw_peak = max([max(0.,item['force_contact_frame'][0]) for item in self.raw_cross_contacts],default=0.)
        penetration = max(0., max([-item['distance_m'] for item in self.raw_cross_contacts],default=0.))
        self.maximum_any_contact_force = max(self.maximum_any_contact_force,raw_peak)
        self.maximum_penetration = max(self.maximum_penetration,penetration)
        pole = state['position']+state['rotation_matrix']@TOOL_POLE
        tip_center = self.data.geom_xpos[self.tip_id].copy()
        self.steps += 1
        self.allowed_steps += bool(allowed)
        self.forbidden_steps += bool(forbidden)
        self.saturated_steps += bool(saturated.any())
        self.saturation_streak = self.saturation_streak+1 if saturated.any() else 0
        self.maximum_saturation_streak = max(self.maximum_saturation_streak,self.saturation_streak)
        if forbidden and self.first_forbidden is None:
            self.first_forbidden = dict(time=float(self.data.time),contacts=forbidden)
        self.normal_forces.append(normal)
        self.tangent_forces.append(tangent)
        result = dict(time=float(self.data.time), phase=phase, position=state['position'].tolist(),
            quaternion=state['quaternion'].tolist(), velocity=state['velocity'].tolist(),
            tool_pole=pole.tolist(), ball_center=tip_center.tolist(),
            ball_lower_surface_world=(tip_center-np.array([0.,0.,TIP_RADIUS])).tolist(),
            reference_position=np.asarray(desired['pos']).tolist(), command_position=command['pos'].tolist(),
            normal_reference_offset_m=self.offset, filtered_force_n=self.filtered_force,
            normal_force_n=normal,tangent_force_n=tangent,
            peak_any_contact_normal_force_n=raw_peak,max_penetration_m=penetration,
            allowed_contact=bool(allowed), forbidden_contact=bool(forbidden),
            allowed_contacts=allowed,forbidden_contacts=forbidden,
            raw_cross_contacts=self.raw_cross_contacts,
            saturated_actuators=np.flatnonzero(saturated).tolist(),ctrl=ctrl.tolist(),
            tool_plane_error_m=float(pole[2]-self.pad_top) if self.task=='surface' else None)
        self.last_measurement = result
        return result

    def step(self, desired):
        return self.step_task(desired)['forbidden_contact']

    def metadata(self):
        return dict(backend=f'native MuJoCo {self.mj.__version__}',time_step=DT,
            controller='unmodified tests/reference/hnuter_control.py',controller_mass_kg=self.controller.mass,
            physical_total_mass_kg=float(self.model.body_mass.sum()),tool_mass_kg=self.scenario['tool_mass_kg'],
            tool_pole_body=TOOL_POLE.tolist(),tip_center_body=TIP_CENTER.tolist(),tip_radius_m=TIP_RADIUS,
            pad_center=self.pad_center.tolist() if self.task=='surface' else None,
            pad_half_extents=self.pad_half.tolist() if self.task=='surface' else None,
            contact_material=self.contact_material,force_controller=dict(self.force_cfg),
            allowed_contact='probe_tip/task_pad top face only during engage, operate or release',
            force_source='MuJoCo mj_contactForce; normal first and tangent next two components in contact frame',
            force_feedback_scope='bounded outer position-reference offset, separately reported; no body pose writes',
            force_offset_sign='positive offset shifts commanded body height down; negative shifts it up',
            material_calibration='soft numerical contact; not calibrated to a physical tool or rigid surface; report solver penetration and full-operation force metrics',
            perception_scope='real native MID-360 rays; task pad geometry is external task command, not asserted observed')


def execute_reference(world, reference, duration=20., operate_interval=(20/3,40/3),
                      contact_interval=(4.,16.), record_stride=10):
    """Development/expert ceiling; predeclared gates are not training claims."""
    begin, end = map(float,operate_interval)
    engage, release = map(float,contact_interval)
    count = round(duration/DT)
    time_origin = float(world.data.time)
    traces = []
    operating = []
    full_operating = []
    longest_loss = loss = 0
    full_longest_loss = full_loss = 0
    max_tracking = 0.
    first_contact = None
    for step in range(count):
        time = step*DT
        if time < engage: phase='free'
        elif time < begin: phase='engage'
        elif time <= end: phase='operate'
        elif time < release: phase='release'
        else: phase='retreat'
        desired = reference.sample(time)
        # Engage position reference approaches the surface analytically; the
        # force servo begins only once nominal tool height reaches the surface.
        near_surface = float(desired['pos'][2]+TOOL_POLE[2]-world.pad_top) <= .025
        result = world.step_task(desired,phase=phase,
                                 force_enabled=world.task=='surface' and phase in ('engage','operate','release') and near_surface)
        max_tracking = max(max_tracking,float(np.linalg.norm(np.array(result['position'])-desired['pos'])))
        if result['allowed_contact'] and first_contact is None:
            first_contact = result['time']
        if begin+1.0 <= time <= end:
            operating.append(result)
            loss = 0 if result['allowed_contact'] else loss+1
            longest_loss = max(longest_loss,loss)
        if begin <= time <= end:
            full_operating.append(result)
            full_loss = 0 if result['allowed_contact'] else full_loss+1
            full_longest_loss = max(full_longest_loss,full_loss)
        if step % record_stride == 0 or step == count-1:
            traces.append(result)
        if result['forbidden_contact']:
            break
    force = np.array([item['normal_force_n'] for item in operating])
    contact = np.array([item['allowed_contact'] for item in operating],dtype=bool)
    final = world.state(); goal = reference.sample(duration)
    distance = float(np.linalg.norm(final['position']-goal['pos']))
    operating_force_ok = ((force >= .5)&(force <= 1.5)) if len(force) else np.array([],bool)
    complete = step+1 == count
    report = dict(schema='hnuter_task_native_execution_v1',completed=complete,steps=step+1,
        simulation_seconds=(step+1)*DT,metadata=world.metadata(),scenario=world.scenario,
        reference_time_origin=time_origin,
        forbidden_contact_steps=world.forbidden_steps,first_forbidden=world.first_forbidden,
        allowed_contact_steps=world.allowed_steps,first_contact_time=first_contact,
        saturated_steps=world.saturated_steps,saturation_fraction=world.saturated_steps/(step+1),
        maximum_continuous_saturation_s=world.maximum_saturation_streak*DT,
        max_position_tracking_error_m=max_tracking,final_goal_error_m=distance,
        final_speed_mps=float(np.linalg.norm(final['velocity'])),
        operate_interval=[begin,end],force_measurement_interval=[begin+1.,end],
        operate_interval_simulation_time=[time_origin+begin,time_origin+end],
        measured_operating_steps=len(operating),operating_contact_fraction=float(contact.mean()) if len(contact) else 0.,
        whole_operate_contact_fraction=float(np.mean([v['allowed_contact'] for v in full_operating])) if full_operating else 0.,
        whole_operate_force_interval_fraction=float(np.mean([.5<=v['normal_force_n']<=1.5 for v in full_operating])) if full_operating else 0.,
        whole_operate_longest_contact_loss_s=full_longest_loss*DT,
        whole_operate_force_quantiles_n=np.quantile([v['normal_force_n'] for v in full_operating],[0,.05,.5,.95,1]).tolist() if full_operating else [],
        actual_tool_tangential_progress_m=float(np.linalg.norm(np.array(full_operating[-1]['tool_pole'])[:2]-np.array(full_operating[0]['tool_pole'])[:2])) if full_operating else 0.,
        longest_operating_contact_loss_s=longest_loss*DT,
        operating_force_interval_fraction=float(operating_force_ok.mean()) if len(force) else 0.,
        normal_force_quantiles_n=np.quantile(force,[0,.05,.5,.95,1]).tolist() if len(force) else [],
        operating_tool_plane_absolute_error_quantiles_m=np.quantile([abs(v['tool_plane_error_m']) for v in operating],[0,.5,.95,1]).tolist() if operating and world.task=='surface' else [],
        all_phase_peak_force_n=float(max(world.normal_forces,default=0)),
        peak_any_robot_contact_force_n=world.maximum_any_contact_force,
        maximum_actual_penetration_m=world.maximum_penetration,
        max_outer_reference_offset_m=max(abs(item['normal_reference_offset_m']) for item in traces),
        outer_reference_offset_range_m=[min(item['normal_reference_offset_m'] for item in traces),
                                        max(item['normal_reference_offset_m'] for item in traces)],
        acceptance=dict(contact_fraction_min=.95,force_interval_n=[.5,1.5],force_interval_fraction_min=.90,
            longest_contact_loss_max_s=.15,peak_force_max_n=3.,goal_error_max_m=.12,tracking_max_m=.15,
            saturation_fraction_max=.01,operate_force_initial_settle_s=1.))
    report['passed'] = bool(complete and world.forbidden_steps==0 and distance<.12 and max_tracking<.15 and
        report['operating_contact_fraction']>=.95 and report['operating_force_interval_fraction']>=.90 and
        report['longest_operating_contact_loss_s']<=.15 and report['all_phase_peak_force_n']<=3. and
        report['saturation_fraction']<=.01)
    return report,traces
