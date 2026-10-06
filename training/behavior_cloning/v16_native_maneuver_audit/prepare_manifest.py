"""Seal Stage-1 sources once; never reads frozen dataset samples or launches a game."""
from datetime import datetime,timezone
from contracts import HERE,ROOT,read,write,sha,require,protected

def run():
    require(not (HERE/'source_manifest.json').exists(),'Source seal already exists; do not silently reseal')
    existing=read(ROOT/'training/behavior_cloning/v15_representation_investigation/source_seal.json')
    files=dict(existing['files'])
    for name,expected in files.items(): require(sha(ROOT/name)==expected,'Existing protected/pinned source changed: '+name)
    # Verify completed V14 artifacts by its existing integrity seal, never refit/load model tensors.
    old=ROOT/'training/behavior_cloning/v14_temporal_objective/run_v2'
    for name,expected in read(old/'integrity.json')['files'].items():
        path=old/name; require(path.resolve().is_relative_to(old.resolve()),'Old artifact path escape')
        require(sha(path)==expected,'Completed V14 artifact changed: '+name)
        files[path.relative_to(ROOT).as_posix()]=expected
    paths=list(HERE.rglob('*.py'))+[HERE/'protocol.md',HERE/'README.md',HERE/'collection_plan.json',HERE/'source_audit.json',HERE/'source_manifest_preflight_attempt1.json',
          ROOT/'training/behavior_cloning/v13_dagger/phase3_shadow_v2/diagnostics.py',
          ROOT/'python-example/venv/Lib/site-packages/rlbot/managers/bot.py',
          ROOT/'python-example/venv/Lib/site-packages/rlbot/interface.py']
    for path in paths: files[path.relative_to(ROOT).as_posix()]=sha(path)
    hashes=protected(); files.update(hashes)
    write(HERE/'source_manifest.json',dict(version='v16_stage1_seal_v2',created_utc=datetime.now(timezone.utc).isoformat(),files=files,
          protected_count=43,sample_reads=False,old_model_use='Hash verification only; no checkpoint load/inference',
          V10_access='Manifest/source checksum only; no train/validation/test samples',
          source_audit_sha256=sha(HERE/'source_audit.json'),protocol_sha256=sha(HERE/'protocol.md')))
    print('V16 SOURCE SEAL:',sha(HERE/'source_manifest.json'),'Protected:',len(hashes),flush=True)

if __name__=='__main__': run()
