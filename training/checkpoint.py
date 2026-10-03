from importlib.metadata import version
from pathlib import Path
import torch
from environment import contract
from policies import ActorCritic


def save_checkpoint(path, model, optimizer, config, step):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(dict(format_version=1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    config=config, step=step, contract=contract(),
                    versions={p: version(p) for p in ('torch', 'rlgym', 'rlgym-api',
                              'rlgym-rocket-league', 'rocketsim', 'numpy')}), temporary)
    temporary.replace(path)


def load_checkpoint(path, device='cpu'):
    data = torch.load(path, map_location=device, weights_only=True)
    if data['format_version'] != 1 or data['contract'] != contract():
        raise ValueError('Checkpoint observation/action contract does not match this environment')
    for package, expected in data['versions'].items():
        if package != 'torch' and version(package) != expected:
            raise ValueError(f'{package} version differs from checkpoint: expected {expected}')
    model = ActorCritic(data['config']['hidden_size'],
                        ground_only=data['config'].get('ground_only', False)).to(device)
    model.load_state_dict(data['model'])
    model.eval()
    return model, data
