"""Phase 1 only: audit unchanged V12 code/checkpoints, no fitting/live launch.

Writes new research-gate evidence and the two explicitly requested audit reports.
Numerical probes inspect the existing evaluation function's input mutation only;
no V10 test predictions are generated and no frozen artifact is rewritten.
"""
import ast
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys
import types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
V12=HERE.parent/'v12'
PIN='080943fd8f9a81ada800e0a2f9ff3593b67f66d32272211d67030c03cf433a83'
VERSION=HERE.parent/'datasets/v10_live_1v1_18d_20261005_v1'
REPORT=ROOT/'training/reports/v13_context_audit_20261005'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def load(path): return json.loads(path.read_text(encoding='utf-8'))

def write(path,obj): path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    if REPORT.with_suffix('.json').exists() or REPORT.with_suffix('.md').exists():
        raise RuntimeError('Phase 1 report exists; preserve it and stop for review')
    assert sha(VERSION/'manifest.json')==PIN
    manifest=load(VERSION/'manifest.json')
    gate=load(V12/'test_suite_started.json'); run=Path(gate['run_path'])
    metadata=load(run/'run_metadata.json'); selections=load(run/'selection_complete_before_test.json')
    protected={}
    for path,h in metadata['code_sha256'].items():
        p=Path(path); assert sha(p)==h, 'V12 implementation changed: '+path
        protected[path]=h
    assert sha(V12/'experiment.json')=='58ce29ff9a7005262c8449bcc0f0ed7696b0c80fa3722eb4737a356d23f4ed74'
    config=load(V12/'experiment.json')
    source=V12/'bakeoff.py'; tree=ast.parse(source.read_text(encoding='utf-8'))
    wanted={'Policy','decode','predict','loss_parts','metrics'}
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted]
    env=dict(np=np,torch=torch,nn=nn,F=F,CHANNELS=tuple(manifest['native_channels']))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
    torch.set_num_threads(1)
    checkpoint_shapes={}
    for kind in ('A','B'):
        path=Path(selections[kind]['checkpoint'])
        assert sha(path)==selections[kind]['checkpoint_sha256']==gate['selected_checkpoint_sha256'][kind]
        protected[str(path)]=sha(path)
        saved=torch.load(path,map_location='cpu',weights_only=True)
        assert saved['experiment']==config and saved['dataset_manifest_sha256']==PIN
        model=env['Policy'](kind); model.load_state_dict(saved['state_dict']); model.eval()
        with torch.inference_mode():
            logits,steer,h=model(torch.zeros(7,18,dtype=torch.float32))
        checkpoint_shapes[kind]=dict(parameters=sum(p.numel() for p in model.parameters()),
            input=[7,18],logits_shape=list(logits.shape),steering_shape=list(steer.shape),
            input_output_dtype='float32',hidden_shape=None if h is None else list(h.shape),
            weights={k:dict(shape=list(t.shape),dtype=str(t.dtype)) for k,t in saved['state_dict'].items()},
            checkpoint_sha256=sha(path),epoch=saved['epoch'])
    # Scripted finite numerical outputs expose exactly what V12 predict passes in.
    class Capture:
        def __init__(self): self.inputs=[]; self.cursor=0
        def __call__(self,x,h):
            self.inputs.append(x.detach().numpy().copy())
            modes=(1,2,0,3,1,0); out=torch.zeros(len(x),4)
            for j in range(len(x)): out[j,modes[self.cursor+j]]=10
            self.cursor+=len(x)
            return out,torch.full((len(x),),.25),h
    obs=np.arange(6*18,dtype=np.float32).reshape(6,18)/100
    match=dict(obs=obs,mode=np.zeros(6,np.int64),metadata={'match_id':'numeric_mutation_probe'})
    tf=Capture(); env['predict'](tf,match,'teacher_forced',3)
    sf=Capture(); pred=env['predict'](sf,match,'student_forced_offline',3)
    actual_tf=np.concatenate(tf.inputs); actual_sf=np.concatenate(sf.inputs)
    assert np.array_equal(actual_tf,obs) and np.array_equal(actual_sf[:,:13],obs[:,:13])
    assert np.array_equal(match['obs'],obs), 'Evaluation mutated recorded inputs'
    for i in range(len(obs)):
        last_mode=0 if i==0 else int(pred['modes'][i-1]); expected=np.zeros(4,np.float32); expected[last_mode]=1
        last_steer=0 if i==0 else pred['native'][i-1,1]
        assert np.array_equal(actual_sf[i,13:17],expected) and actual_sf[i,17]==last_steer
    assert all(sha(Path(p))==h for p,h in protected.items()) and sha(VERSION/'manifest.json')==PIN
    conclusions={
        'input_order':manifest['features'],
        'tensor_contract':{'callback_input':'[L,18] float32, L <=512 during chunks; [1,18] during autoregressive evaluation',
                           'gru_internal_input':'[1,L,64] float32, batch_first=True',
                           'gru_hidden':'[1,1,64] float32; one layer, one direction, one match at a time',
                           'output':'mode logits [L,4] float32 and tanh continuous steering [L] float32; decoded native controls [L,8] float32',
                           'target':'mode int64 for CE; analog float32; frozen buttons uint8 converted to float32 only for native metric arrays'},
        'hidden_reset':'None at each match start in train/validation/final evaluation and each independent context pass. No goal/replay/kickoff/missing-ball resets. Training h.detach() at chunk boundaries truncates gradients without resetting values.',
        'chronology':'Fixed frozen match order, consecutive slices start:start+512; tail retained. Every train callback once per epoch. Optimizer updates each chunk; hidden values computed under earlier weights carry into the next chunk.',
        'teacher_forced':'Recorded 18D arrays passed unchanged; GRU hidden carries across consecutive chunks.',
        'student_forced':'Recorded input copied callback-by-callback; replace indices 13:17 with previous argmax predicted mode one-hot and index17 with previous decoded steering (zero outside Chase). Initial previous action Neutral/zero. Recorded indices0:13 remain bitwise unchanged. No ground-truth prior actions used after initialization.',
        'mutation_probe':{'passed':True,'only_indices_changed':[13,14,15,16,17],'source_arrays_unchanged':True,'probe_kind':'six numerical rows, no trajectories/fitting or dataset test predictions'},
        'teacher_information':'Previous-action fields are intentionally teacher-forced in standard inputs. No private sequence ID/index/phase is read by Policy/train/teacher_loss. Final diagnostics use private labels only as retrospective metric masks. Current prediction is a service forecast supplied at the callback, not a later recorded ball outcome.',
        'reproducibility':'Code/config/checkpoint hashes match recorded V12 run; safe weights_only state-dict loading succeeds. Seed42/runtime/device/selection history recorded. Exact rerun bit identity is not guaranteed because GPU determinism was not enforced. Optimizer state and per-chunk recurrent state were not saved, so checkpoints do not support exact interrupted-run continuation.',
        'diagnosis':'No context mutation bug found in inspected predict implementation/numerical probe. Evidence supports sensitivity to previous-action context. Physical-state distribution shift and the precise learned shortcut mechanism are not established.'}
    result=dict(gate_version='v13_context_audit_v1',phase=1,status='passed_with_scope',
                created_at=datetime.now(timezone(timedelta(hours=5,minutes=30))).isoformat(),
                manifest_sha256=PIN,v12_run=str(run),v12_verified_hashes=protected,
                checkpoint_tensor_shapes=checkpoint_shapes,audit=conclusions,
                unchanged={'v12':True,'frozen_manifest':True,'dataset_records':'Not read or written during this implementation audit',
                           'teacher_and_bot_directories':'Not imported or modified'},
                new_training_started=False,live_launched=False,
                source_references={node.name:{'file':str(source),'line':node.lineno} for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef))})
    evidence=HERE/'phase1_context_audit_v1'; evidence.mkdir(exist_ok=False)
    write(evidence/'audit.json',result); write(REPORT.with_suffix('.json'),result)
    lines=['# V13 context audit','', 'Phase 1: **passed with scope**. Unchanged V12 inspected; no fixes, fitting or live launch.', '',
           '## Exact 18D ordering','', ' | '.join(f'{i}: {name}' for i,name in enumerate(manifest['features'])), '']
    for key in ('tensor_contract','hidden_reset','chronology','teacher_forced','student_forced','teacher_information','reproducibility','diagnosis'):
        value=conclusions[key]
        lines+=['## '+key.replace('_',' ').capitalize(),'',json.dumps(value,indent=2) if isinstance(value,dict) else value,'']
    lines+=['## Evidence and limits','',
            'Numerical capture of the original V12 evaluation function confirms teacher-forced input identity, unchanged physical/prediction/dt indices0:13, exactly previous predicted mode/steering in indices13:18, and no in-place modification. This is an implementation probe, not another V10 test evaluation.', '',
            'Checkpoint/config/code SHA256s and every checkpoint tensor shape/dtype are in the JSON companion. MLP has 11,013 parameters; GRU has 26,501. Hidden shape is [1,1,64].', '',
            'The V11 frozen row audit remains the evidence for prior-action provenance in the recorded data. Phase1 does not repeat the dataset sweep or claim a comprehensive root-cause explanation.', '',
            'Original code and saved artifacts unchanged. Stop before subsequent gates until their specifications are resolved.']
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('V13 PHASE1 COMPLETE:',REPORT.with_suffix('.md'),flush=True)

if __name__=='__main__': main()
