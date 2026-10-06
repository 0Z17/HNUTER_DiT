"""Whitelisted training inputs for factory causal navigation (NumPy/DataLoader).

No scene geometry, OMPL states, expert quality or future sensing enters inputs.
All runs of an identical mesh must share one split across supplied manifests.
"""
from pathlib import Path
import json
import numpy as np
from .sdf_navigation_data import sha

INPUT_KEYS=('fields','ages','current_velocity','current_angular_velocity','current_quaternion',
            'current_reference_acceleration','goal','execution_delay_s')

class FactoryWindowDataset:
    def __init__(self, manifests, split=0):
        if split not in (0,1,2):raise ValueError('invalid split')
        self.entries=[];groups={}
        for manifest in manifests:
            manifest=Path(manifest);m=json.loads(manifest.read_text())
            if m['schema']!='factory_causal_navigation_v1':raise ValueError('wrong schema')
            group=m['split_group'];assigned=m['split']
            if group in groups and groups[group]!=assigned:raise ValueError('same factory mesh leaks across splits')
            groups[group]=assigned
            if assigned==split:
                self.entries.extend((manifest.parent/r['file'],r['sha256']) for r in m['records'])
        if not self.entries:raise ValueError('no records for requested split')

    def __len__(self):return len(self.entries)

    def __getitem__(self,index):
        path,digest=self.entries[index]
        if sha(path)!=digest:raise ValueError('window hash mismatch')
        with np.load(path,allow_pickle=False) as d:
            inputs={k:d[k].astype(np.float32,copy=True) for k in INPUT_KEYS}
            labels={'future':d['future'].astype(np.float32,copy=True),'future_times':d['future_times'].astype(np.float32,copy=True)}
        if not all(np.isfinite(x).all() for x in (*inputs.values(),*labels.values())):raise ValueError('nonfinite training sample')
        if inputs['fields'].shape!=(4,6,32,32,32) or labels['future'].shape!=(81,19):raise ValueError('incompatible tensor shape')
        return dict(inputs=inputs,labels=labels)
