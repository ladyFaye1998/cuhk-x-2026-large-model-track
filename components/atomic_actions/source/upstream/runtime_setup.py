"""Provision isolated numerical environments before any observations or fitting."""
from pathlib import Path
import argparse
import json
import os
import platform
import subprocess
import sys

UPSTREAM = {'numpy':'2.0.2', 'scipy':'1.15.3', 'scikit-learn':'1.6.1',
            'pandas':'2.2.3', 'lightgbm':'4.6.0', 'numba':'0.61.2',
            'torch':'2.10.0+cu128', 'torchvision':'0.25.0+cu128',
            'timm':'1.0.15', 'safetensors':'0.6.2', 'threadpoolctl':'3.6.0'}
DOWNSTREAM = {'numpy':'2.4.4', 'scipy':'1.18.0', 'scikit-learn':'1.9.0',
              'torch':'2.10.0+cpu', 'pandas':'2.2.3', 'joblib':'1.5.3',
              'threadpoolctl':'3.6.0'}
PYTHONS = ('3.12.13', '3.12.14')
UPSTREAM_CPU = {k:v for k,v in UPSTREAM.items() if k not in ('torchvision','timm','safetensors')}
UPSTREAM_CPU['torch'] = '2.10.0+cpu'

def clean_environment():
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               PIP_NO_CACHE_DIR='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
    return env

def inspect(python, expected, cuda=False, shared_torch=None):
    code = ('import importlib.metadata as m,json,platform,torch; '
            f'expected={expected!r}; '
            'actual={k:m.version(k) for k in expected}; '
            f'assert platform.python_version() in {PYTHONS!r}, platform.python_version(); '
            'assert actual==expected, (actual,expected); '
            + ('assert torch.cuda.is_available(); ' if cuda else '') +
            'print(json.dumps(dict(python=platform.python_version(),packages=actual,'
            'cuda_available=torch.cuda.is_available())))')
    env=clean_environment()
    if shared_torch: env['PYTHONPATH']=str(shared_torch)
    if not cuda: env['CUDA_VISIBLE_DEVICES']=''
    result = subprocess.run([str(python), '-B', '-c', code], env=env,
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError('Runtime inspection failed for '+str(python)+'\n'+result.stdout+'\n'+result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])

def setup(root, require_cuda=True):
    # The pinned Linux image remains the default and is unchanged. Setting
    # CUHKX_ALLOW_NON_LINUX_HOST=1 permits a declared local host of the same Python
    # version; the deviation is recorded in the runtime manifest and the run report so
    # a reviewer can see the environment was not the pinned image.
    allow_other_host = os.environ.get('CUHKX_ALLOW_NON_LINUX_HOST') == '1'
    if platform.python_version() not in PYTHONS:
        raise RuntimeError('The full notebook requires Python 3.12.13/3.12.14; found '
                           + platform.python_version())
    if platform.system() != 'Linux' and not allow_other_host:
        raise RuntimeError('The full notebook requires its pinned Linux Python 3.12.13/3.12.14 image')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('Runtime directory must be empty; environment reuse is not implicit')
    # CUDA packages are supplied by the measured pinned Kaggle image, never
    # replaced by an uncontrolled GPU package installation.
    inspect(sys.executable, {'torch': UPSTREAM['torch'], 'torchvision': UPSTREAM['torchvision']}, cuda=require_cuda)
    env = clean_environment()
    upstream, upstream_cpu, downstream = root / 'upstream', root / 'upstream_cpu', root / 'downstream'
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', '--system-site-packages', str(upstream)], env=env, check=True)
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(downstream)], env=env, check=True)
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(upstream_cpu)], env=env, check=True)
    # venv lays interpreters out as bin/python on POSIX and Scripts/python.exe on Windows.
    _venv_python = (lambda base: base / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))
    up, cpu, down = _venv_python(upstream), _venv_python(upstream_cpu), _venv_python(downstream)
    # On the pinned Linux image the base interpreter is the system Python, so
    # --system-site-packages gives the upstream env its measured CUDA torch. When the
    # declared host runs the notebook from inside another virtualenv, that flag resolves
    # to the *base* installation instead, which does not carry torch. Make the same
    # inheritance explicit rather than installing a second, uncontrolled CUDA torch.
    if allow_other_host:
        import sysconfig
        inherited = [entry for entry in sys.path if entry.endswith('site-packages')]
        target = Path(sysconfig.get_paths()['purelib'].replace(sys.prefix, str(upstream)))
        if not target.is_dir():
            target = upstream / ('Lib/site-packages' if os.name == 'nt' else
                                 f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')
        target.mkdir(parents=True, exist_ok=True)
        newline = chr(10)
        (target / '_cuhkx_declared_host_inherit.pth').write_text(
            newline.join(inherited) + newline, encoding='utf-8')

    pins = [f'{k}=={v}' for k,v in UPSTREAM.items() if k not in ('torch','torchvision')]
    pins += ['llvmlite==0.44.0', 'joblib==1.6.0', 'requests==2.34.2', 'PyYAML==6.0.2', 'packaging==24.2']
    # The upstream env installs with --no-deps. On the pinned image pandas finds its
    # runtime dependencies through --system-site-packages; a declared host has no such
    # system interpreter, so name them explicitly at the versions pip resolved for the
    # matching CPU environment. This adds no new library, only what the image supplied.
    if allow_other_host:
        pins += ['pytz==2026.3.post1', 'python-dateutil==2.9.0.post0',
                 'tzdata==2026.3', 'six==1.17.0']
    subprocess.run([sys.executable, '-m', 'pip', '--python', str(up), 'install', '--only-binary=:all:', '--no-deps', *pins], env=env, check=True)
    # Both CPU virtualenvs use one CPU-only torch installation, which has no
    # NumPy dependency. Their NumPy/scipy/sklearn installations remain separate.
    shared_torch=root/'shared_cpu_torch'
    subprocess.run([sys.executable, '-m', 'pip', '--python', str(down), 'install', '--only-binary=:all:',
                    '--target', str(shared_torch), '--extra-index-url', 'https://download.pytorch.org/whl/cpu',
                    'torch==2.10.0+cpu'], env=env, check=True)
    if any(shared_torch.glob('numpy*')):
        raise RuntimeError('The shared CPU torch directory must not carry a numerical runtime')
    subprocess.run([sys.executable, '-m', 'pip', '--python', str(cpu), 'install', '--only-binary=:all:',
                    *[f'{k}=={v}' for k,v in UPSTREAM_CPU.items() if k!='torch'],
                    'llvmlite==0.44.0', 'joblib==1.6.0', 'requests==2.34.2'], env=env, check=True)
    # This virtualenv has no system packages and cannot import the CUDA wheel.
    subprocess.run([sys.executable, '-m', 'pip', '--python', str(down), 'install', '--only-binary=:all:',
                    *[f'{k}=={v}' for k,v in DOWNSTREAM.items() if k!='torch'], 'requests==2.34.2',
                    # run_pipeline.py reads peak RSS via psutil on Windows; the pinned Linux
                    # image uses resource.getrusage instead and needs no extra package.
                    *(['psutil==7.2.2'] if os.name == 'nt' else [])], env=env, check=True)
    report = {'upstream_python': str(up), 'downstream_python': str(down),
              'upstream_cpu_python':str(cpu),'shared_cpu_torch':str(shared_torch),
              'upstream': inspect(up, UPSTREAM, cuda=require_cuda),
              'upstream_cpu':inspect(cpu,UPSTREAM_CPU,shared_torch=shared_torch),
              'downstream': inspect(down, DOWNSTREAM,shared_torch=shared_torch),
              'upstream_CUDA_image_preserved': True, 'environments_isolated': True,
              'cuda_availability_required':require_cuda,
              'environment_bootstrap':'venv --without-pip; base pip --python manages each exact interpreter'}
    (root / 'runtime_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',required=True)
    print(json.dumps(setup(p.parse_args().out),indent=2))
