"""Torch-only diagnostic process. No RLBot, transport, training, or dataset reads."""
import ast
import hashlib
import math
import sys
import time
import torch
from torch import nn
from common import CHECKPOINT,CHECKPOINT_PIN,MANIFEST_PIN,V13,MODES,read,sha,POLICY_SOURCE_PIN

def emit(obj):
    import json
    print(json.dumps(obj,allow_nan=False),flush=True)

def main():
    import json
    torch.set_num_threads(1)
    if sha(CHECKPOINT) != CHECKPOINT_PIN: raise RuntimeError('Checkpoint mismatch')
    if sha(V13/'bootstrap.py') != POLICY_SOURCE_PIN: raise RuntimeError('Policy class source mismatch')
    # Execute only the accepted model class, never bootstrap main/readers.
    tree = ast.parse((V13/'bootstrap.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Policy')
    env = dict(torch=torch,nn=nn)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'accepted_v13_policy','exec'),env)
    saved = torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    if saved['dataset_manifest_sha256'] != MANIFEST_PIN: raise RuntimeError('Dataset identity mismatch')
    model = env['Policy']().eval()
    model.load_state_dict(saved['state_dict'],strict=True)
    h = None
    last = 0
    emit(dict(status='ready',checkpoint_sha256=CHECKPOINT_PIN,parameters=26181,
        torch=torch.__version__,device='cpu',hidden_resets=1,role='diagnostic_only'))
    for line in sys.stdin:
        request = json.loads(line)
        if request == {'command':'close'}: return
        if set(request) != {'callback','observation13'} or request['callback'] != last+1:
            raise ValueError('One chronological callback stream only; no reset or private fields accepted')
        obs = request['observation13']
        if len(obs)!=13 or any(type(v) not in (int,float) or not math.isfinite(v) for v in obs):
            raise ValueError('Expected 13 finite numeric features')
        x = torch.tensor([obs],dtype=torch.float32)
        before = None if h is None else hashlib.sha256(h.numpy().tobytes()).hexdigest()
        start = time.perf_counter()
        with torch.inference_mode(): logits,steer,h = model(x,h)
        if not all(torch.isfinite(v).all() for v in (logits,steer,h)): raise ValueError('Nonfinite model output')
        i = int(logits.argmax(1).item())
        a = [0.,0.,0.,0.,0.,False,False,False]
        if i==1: a[0]=1.; a[1]=float(steer.item())
        elif i==2: a[5]=True
        elif i==3: a[2]=-1.; a[5]=True
        last = request['callback']
        emit(dict(callback=last,mode=MODES[i],controls=a,logits=logits[0].tolist(),
            continuous_steer=float(steer.item()),hidden_before_sha256=before,
            hidden_after_sha256=hashlib.sha256(h.numpy().tobytes()).hexdigest(),
            hidden_resets=1,inference_seconds=time.perf_counter()-start))

if __name__=='__main__': main()
