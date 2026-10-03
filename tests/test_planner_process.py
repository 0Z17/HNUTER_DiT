import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np

from hnuter_dit.planner_process import PlannerProcess, _worker_environment


def test_client_import_does_not_import_torch():
    source = "import sys,json; import hnuter_dit.planner_process; print(json.dumps('torch' in sys.modules))"
    result = subprocess.run([sys.executable, '-c', source], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) is False


def test_worker_environment_strips_isaac_python_and_libraries(monkeypatch):
    for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'LD_LIBRARY_PATH'):
        monkeypatch.setenv(key, '/poisoned/isaac/torch')
    monkeypatch.delenv('HNUTER_DIT_LIBRARY_PATH', raising=False)
    env = _worker_environment()
    assert not {'PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD'} & env.keys()
    assert env['LD_LIBRARY_PATH'] == ''
    assert env['PYTHONNOUSERSITE'] == '1'


def test_missing_worker_python_fails_with_actionable_message(tmp_path):
    with pytest.raises(RuntimeError, match='HNUTER_DIT_PYTHON'):
        PlannerProcess(tmp_path, python=str(tmp_path/'not-installed'))


def test_existing_package_exports_remain_lazy_and_compatible():
    import hnuter_dit
    from hnuter_dit.model import DiffusionSchedule
    assert hnuter_dit.DiffusionSchedule is DiffusionSchedule
    with pytest.raises(AttributeError):
        hnuter_dit.not_a_public_symbol


@pytest.fixture
def protocol_worker(tmp_path, monkeypatch):
    script = tmp_path/'protocol_worker.py'
    script.write_text('''import json,sys,time
from pathlib import Path
import numpy as np
d=Path(sys.argv[1]); sequence=0
def status(value):
    p=d/'status.tmp';p.write_text(json.dumps(value));p.replace(d/'status.json')
status(dict(state='ready',model={'checkpoint':'protocol_test_double'}))
while True:
    if (d/'request.npz').exists():
        with np.load(d/'request.npz') as r:
            next_sequence=int(r['sequence']);mode=str(r['mode']);count=int(r['count']);seed=int(r['seed'])
        if next_sequence>sequence:
            sequence=next_sequence
            if seed==99:time.sleep(30)
            metadata={'error':'test rejection'} if mode=='online' else {'sequence':sequence,'inference_seconds':0.}
            paths=np.zeros((count,128,7),np.float32);paths[:,:,3]=1
            with (d/'response.tmp').open('wb') as f:np.savez(f,sequence=sequence,paths=paths,metadata=json.dumps(metadata))
            (d/'response.tmp').replace(d/'response.npz')
    if (d/'stop.json').exists():
        status(dict(state='stopped',sequence=sequence));break
    time.sleep(.005)
''')
    monkeypatch.setattr(PlannerProcess, '_worker_script', script)
    return tmp_path


def test_atomic_protocol_multiple_requests_error_recovery_and_close(protocol_worker):
    client = PlannerProcess(protocol_worker, python=sys.executable, device='cpu')
    assert client.output_dir.stat().st_mode & 0o777 == 0o755
    tokens=np.zeros((32,10));mask=np.zeros(32,bool);pose=[0,0,2,1,0,0,0]
    paths, metadata = client.sample('baseline',tokens,mask,pose,pose,7,count=2)
    assert paths.shape == (2,128,7) and metadata['sequence'] == 1
    with pytest.raises(RuntimeError, match='test rejection'):
        client.sample('online',tokens,mask,pose,pose,7,count=1)
    _, metadata = client.sample('baseline',tokens,mask,pose,pose,8,count=1)
    assert metadata['sequence'] == 3
    assert client.close()['state'] == 'stopped'
    assert client.close()['sequence'] == 3
    assert client._process.poll() == 0


def test_request_timeout_terminates_worker_and_closed_client(protocol_worker):
    client = PlannerProcess(protocol_worker, python=sys.executable, device='cpu', request_timeout=.1)
    tokens=np.zeros((32,10));mask=np.zeros(32,bool);pose=[0,0,2,1,0,0,0]
    with pytest.raises(RuntimeError, match='timed out'):
        client.sample('baseline',tokens,mask,pose,pose,99,count=1)
    assert client._process.poll() is not None
    with pytest.raises(RuntimeError, match='closed'):
        client.sample('baseline',tokens,mask,pose,pose,7,count=1)
    client.close()


def test_unexpected_worker_death_is_not_mistaken_for_a_pending_request(protocol_worker):
    client = PlannerProcess(protocol_worker, python=sys.executable, device='cpu')
    client._process.kill()
    client._process.wait(timeout=5)
    with pytest.raises(RuntimeError, match='exited'):
        client.sample('baseline',np.zeros((32,10)),np.zeros(32,bool),[0,0,2,1,0,0,0],
                      [1,0,2,1,0,0,0],7,count=1)
    with pytest.raises(RuntimeError, match='exited'):
        client.close()
    assert client._closed
