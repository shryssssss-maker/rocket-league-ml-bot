"""Seal the accepted V10 artifacts without changing any collection input.

--check is a short non-live preflight. Creation and --verify read all artifacts.
No collection, tensor export, preprocessing, resampling, or training occurs.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collection as c

VERSION = 'v10_live_1v1_18d_20261005_v1'
INDIA_TIME = timezone(timedelta(hours=5, minutes=30), name='Asia/Kolkata')
OUT = HERE.parent / 'datasets' / VERSION
REVIEW = c.ROOT / 'training/reports/v10_collection_review_20261005.json'
REPORT = c.ROOT / 'training/reports/v10_dataset_freeze_20261005'
EXPECTED_SPLITS = {
    'train': ['pilot_01', 'v10_001', 'v10_004', 'v10_006'],
    'validation': ['pilot_02', 'v10_002'],
    'test': ['pilot_03', 'v10_003', 'v10_005'],
}
LIMITATIONS = [
    'Splits are match-disjoint, NOT fully opponent/session-disjoint: human_a spans train/validation/test. human_b has two train matches only (v10_004 and v10_006), sharing human_b_v10_session_01. human_c has one test match (v10_005). No records or split assignments were changed.',
    'Opponent identities were declared by the user; pilot play-session identities were not separately recorded and remain unknown.',
    'Coverage targets are provisional heuristics, not proof of recurrent learning or generalization; natural collection does not guarantee extreme states or long pending-sequence interruptions.',
    'Counts describe actual processed RLBot callbacks, not lossless network delivery or undelivered physics ticks.',
    'Hidden teacher sequence state is diagnostic only and excluded from the unchanged 18D observations.',
    'This is an immutable hash-identified reference artifact, not an OS write lock on the original files. Referenced files must remain available and pass verification; changes require a new dataset version.',
    'Historical code hashes are retained from collection metadata. Current code snapshots are labeled separately and do not claim to reconstruct unavailable historical revisions.',
]

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def rel(path):
    return path.relative_to(c.ROOT).as_posix()

def descriptor(path):
    return dict(path=rel(path), bytes=path.stat().st_size, sha256=c.sha(path))

def preflight():
    # Windows does not necessarily have an IANA timezone database installed.
    # Exercise both timestamp paths before reading the large artifacts.
    datetime.now(INDIA_TIME).isoformat()
    datetime.fromtimestamp(REVIEW.stat().st_mtime, timezone.utc).isoformat()
    review = read(REVIEW)
    if (review['status'] != 'coverage_targets_met_awaiting_review'
            or review['eligible_callbacks'] != 239205 or review['failures']
            or review['invalid_eligible_rows'] or review['dropped_processed_eligible_rows']
            or review['exact_match_split'] != EXPECTED_SPLITS):
        raise RuntimeError('Accepted review differs from the approved dataset')
    c.preserve.verify_sources()
    # Full pilot preservation is checked during the user-run sealing operation.
    for m in review['matches']:
        folder = folder_for(m)
        manifest = read(folder / 'processed/manifest.json')
        if manifest['features'] != list(c.FEATURES) or manifest['channels'] != list(c.CHANNELS):
            raise RuntimeError('Observation/action ordering mismatch')
        if manifest['split'] != m['split']:
            raise RuntimeError('Split mismatch')
        for name, spec in manifest['tensors'].items():
            if spec['shape'][0] != m['callbacks']:
                raise RuntimeError('Tensor count mismatch: ' + name)
    return review

def folder_for(m):
    return (c.PILOT if m['is_pilot'] else c.MATCHES) / m['match_id']

def seal():
    if OUT.exists() or REPORT.with_suffix('.json').exists() or REPORT.with_suffix('.md').exists():
        raise RuntimeError('Freeze version/report already exists; verify it, never overwrite it')
    review = preflight()
    before = c.preserve.verify_sources()
    pilot = c.anchor()
    matches = []
    artifacts = {}
    for n, m in enumerate(review['matches'], 1):
        folder = folder_for(m)
        print(f"FREEZE {n}/9: verifying {m['match_id']} raw + tensor artifacts", flush=True)
        entries = {p.relative_to(folder).as_posix(): descriptor(p)
                   for p in sorted(folder.rglob('*')) if p.is_file()}
        integrity = read(folder / 'integrity.json')
        manifest = read(folder / 'processed/manifest.json')
        for name, h in integrity.items():
            if entries[name]['sha256'] != h:
                raise RuntimeError('Raw integrity mismatch: ' + str(folder / name))
        for name, spec in manifest['tensors'].items():
            entry = entries['processed/' + spec['file']]
            if entry['sha256'] != spec['sha256'] or entry['bytes'] != spec['bytes']:
                raise RuntimeError('Tensor integrity mismatch: ' + name)
        if entries['processed/manifest.json']['sha256'] != m['integrity']['manifest_sha256']:
            raise RuntimeError('Processed manifest changed since accepted audit')
        run = read(folder / 'run_summary.json')
        summary = read(folder / 'agent_summary.json')
        if run['status'] != 'natural_match_ended' or run['error'] or summary['callbacks'] != m['callbacks']:
            raise RuntimeError('Match completion/count mismatch')
        metadata = read(folder / 'metadata.json')
        item = dict(m, dataset_directory=rel(folder), session_id=metadata['session_id'],
                    tensors=manifest['tensors'], tensor_manifest=entries['processed/manifest.json'],
                    files=entries, collection_metadata=metadata,
                    collection_timestamp_evidence={
                        'metadata_file_mtime_utc': datetime.fromtimestamp((folder/'metadata.json').stat().st_mtime, timezone.utc).isoformat(),
                        'meaning': 'Filesystem timestamp evidence only; not an independently recorded wall-clock start time'},
                    historical_code_hashes=metadata.get('code_sha256', metadata.get('code_hashes')),
                    historical_config_hashes=metadata.get('config_sha256', metadata.get('config_hashes')))
        matches.append(item)
        artifacts.update({v['path']: v for v in entries.values()})
    # Inputs are never written. Freeze outputs are exclusively created only after checks pass.
    OUT.mkdir(parents=True, exist_ok=False)
    snapshots = []
    sources = set(c.BASE.glob('*.py')) | set(HERE.glob('*.py')) | {
        c.PLAN, c.ANCHOR, REVIEW, REVIEW.with_suffix('.md'),
        c.ROOT/'training/v2_observation_test/observation_contract.py',
        c.ROOT/'training/live_reference_test/launch.py',
        c.ROOT/'training/live_reference_test/source_hashes.json',
    }
    for source in sorted(sources):
        if not source.is_file(): raise RuntimeError('Snapshot source missing: ' + str(source))
        destination = OUT / 'snapshots' / rel(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        snapshots.append(dict(original=descriptor(source), frozen_copy=descriptor(destination)))
    if c.preserve.verify_sources() != before or c.anchor() != pilot:
        raise RuntimeError('Protected sources/pilot changed during freeze')
    manifest = dict(dataset_version=VERSION, status='frozen_user_approved',
        frozen_at=datetime.now(INDIA_TIME).isoformat(),
        root_path=str(c.ROOT), path_resolution='All paths are repository-root relative; no concatenation or rewritten inputs',
        approval='User approved V10 dataset freeze on 2026-10-05; no training authorization',
        exact_match_split=EXPECTED_SPLITS, matches=matches, artifacts=artifacts,
        current_code_config_report_snapshots=snapshots, protected_source_hashes=before,
        pilot_preservation=pilot, statistics=review,
        features=list(c.FEATURES), native_channels=list(c.CHANNELS),
        analog_order=list(c.CHANNELS[:5]), button_order=list(c.CHANNELS[5:]),
        mode_mapping={'0':'Neutral','1':'Chase','2':'Jump','3':'Front dodge'},
        endianness='little', native_controls='Raw exact ControllerState; tensors retain approved float32 analog/uint8 button representation',
        observation_contract={'dimension':18,'dtype':'float32','source':'snapshots/training/v2_observation_test/observation_contract.py',
            'relative_coordinates':'Native car-local forward/right/up; no team inversion',
            'scales':{'relative_xyz_and_distance':6000,'speed':2300,'horizon_seconds':2,'first_offset_seconds':1/120,'dt_seconds':1/60},
            'first_callback_dt':'Raw null; processed dt tensor 0; observation normalized dt 0',
            'previous_action':'Previously transmitted teacher controls; Neutral initialization; not hidden sequence labels'},
        selector='int((elapsed + 2 - first_slice_time) * 120)',
        limitations=LIMITATIONS, resampled=False, rebalanced=False, augmented=False, training_started=False)
    c.write_json(OUT/'manifest.json', manifest)
    manifest_hash = c.sha(OUT/'manifest.json')
    summary = dict(dataset_version=VERSION, manifest=descriptor(OUT/'manifest.json'),
        eligible_callbacks=review['eligible_callbacks'], splits=review['grouped_coverage']['split'],
        action_counts=review['aggregate_counts']['modes'],
        action_percentages={k:100*v/review['eligible_callbacks'] for k,v in review['aggregate_counts']['modes'].items()},
        sequence_starts=review['sequence_starts'], sequence_completions=review['sequence_completions'],
        near_far=review['aggregate_counts']['near_far'], callback_dt_seconds=review['distributions']['callback_dt'],
        all_statistics=review, limitations=LIMITATIONS,
        integrity={'artifact_files_hashed':len(artifacts),'all_raw_and_tensor_hashes_verified':True,
                   'raw_tensor_roundtrip':'Exact in accepted audit, linked by unchanged manifest hashes',
                   'protected_hashes_unchanged':len(before),'pilot_files_unchanged':len(pilot)},
        dataset_frozen=True, training_started=False)
    c.write_json(OUT/'report.json', summary)
    lines = ['# V10 frozen dataset', '', f'Dataset version: **{VERSION}**. User-approved freeze; no training.', '',
             f'Manifest SHA256: `{manifest_hash}`.', '',
             f'Manifest: `{OUT / "manifest.json"}`. All per-file paths, SHA256 hashes, tensor shapes/dtypes, identities and collection provenance are recorded there.', '',
             '## Split sizes', '', '| Split | Matches | Callbacks | Starts | Completions |', '|---|---:|---:|---:|---:|']
    for split,g in summary['splits'].items():
        lines.append(f"| {split} | {g['matches']} | {g['callbacks']:,} | {g['sequence_starts']} | {g['sequence_completions']} |")
    lines += ['', '## Action and sequence coverage', '', '| Mode | Callbacks | Percent |', '|---|---:|---:|']
    for k,v in summary['action_counts'].items(): lines.append(f"| {k} | {v:,} | {summary['action_percentages'][k]:.4f}% |")
    lines += ['', f"Total callbacks: {review['eligible_callbacks']:,}. Sequence starts/completions: {summary['sequence_starts']}/{summary['sequence_completions']}. All-four-phase completions: {review['complete_sequences_with_all_four_phases']}. Jump and Front-dodge counts exceed approved 1,000/3,000 targets.", '',
              '## Ball availability and timing', '', f"Near/far/missing-ball counts: {json.dumps(summary['near_far'])}.", '',
              f"Callback dt in seconds (first callback of each match excluded): `{json.dumps(summary['callback_dt_seconds'])}`.", '',
              'All processed callbacks retained in original order. No resampling, balancing, chunks, augmentation, synthetic trajectories or changes to the 18D/native-control contracts.', '',
              '## Integrity and paths', '', f"Verified {len(artifacts)} artifact files against collection/tensor manifests. Zero invalid/dropped processed rows in accepted audit; all {len(before)} protected hashes and {len(pilot)} pilot files unchanged.", '',
              'Raw and tensor artifacts remain at their original paths. Code/config/report snapshots are under the versioned `snapshots/` folder. File SHA256 values and tensor shape/dtype/order are enumerated in manifest.json. The accepted collection review remains historical and is not rewritten.', '',
              '## Limitations', ''] + ['- '+x for x in LIMITATIONS] + ['', 'Frozen; stop for review before preprocessing or training.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for suffix in ('.json','.md'):
        shutil.copyfile(OUT/('report'+suffix), REPORT.with_suffix(suffix))
    seal_files = {p.relative_to(OUT).as_posix():c.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    c.write_json(OUT/'SHA256.json', seal_files)
    verify()
    print('FROZEN:', VERSION, '\nManifest SHA256:', manifest_hash, '\nReport:', REPORT.with_suffix('.md'), flush=True)

def verify():
    manifest = read(OUT/'manifest.json')
    for name,h in read(OUT/'SHA256.json').items():
        if c.sha(OUT/name) != h: raise RuntimeError('Freeze output changed: '+name)
    for i,(path,entry) in enumerate(manifest['artifacts'].items(),1):
        actual = c.ROOT/path
        if actual.stat().st_size != entry['bytes'] or c.sha(actual) != entry['sha256']:
            raise RuntimeError('Frozen dataset input changed: '+path)
        if i%20==0: print(f'VERIFY {i}/{len(manifest["artifacts"])} artifact files',flush=True)
    if c.preserve.verify_sources() != manifest['protected_source_hashes'] or c.anchor() != manifest['pilot_preservation']:
        raise RuntimeError('Protection/pilot anchor mismatch')
    print('FREEZE INTEGRITY PASS; no training or input writes',flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--check',action='store_true'); parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if Path(sys.executable).resolve()!=c.preserve.INTERPRETER.resolve(): raise RuntimeError('Use approved live interpreter')
    if args.check:
        preflight(); print('FREEZE PREFLIGHT PASS: nine approved matches, splits and tensor contracts; no freeze/hash sweep yet',flush=True)
    elif args.verify: verify()
    else: seal()
