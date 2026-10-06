"""Immutable summary snapshots. Never replace a file the supervisor may have open."""
import json
import os
from pathlib import Path

class SummaryPublisher:
    def __init__(self,folder):
        self.folder=Path(folder)/'summary_snapshots'
        self.sequence=0
        self.errors=[]

    def publish(self,value):
        self.sequence+=1
        try:
            self.folder.mkdir(exist_ok=True)
            # New path for every publication; readers only open complete .json.
            pending=self.folder/f'{self.sequence:08d}.tmp'
            target=self.folder/f'{self.sequence:08d}.json'
            if target.exists() or pending.exists():raise FileExistsError('Summary name already exists')
            pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            os.rename(pending,target)  # No overwrite, no reader can have target open yet.
            return target
        except OSError as e:
            self.errors.append(dict(sequence=self.sequence,type=type(e).__name__,error=repr(e)))
            return None  # A diagnostic failure must not suppress teacher controls.

class SummaryReader:
    def __init__(self):self.snapshot={};self.latest=None
    def read(self,folder):
        paths=sorted((Path(folder)/'summary_snapshots').glob('*.json'))
        if not paths:return self.snapshot
        latest=paths[-1]
        if latest==self.latest:return self.snapshot
        try:
            value=json.loads(latest.read_text(encoding='utf-8'))
        except (OSError,ValueError):return self.snapshot
        self.snapshot=value;self.latest=latest
        return value

def publication_view(status,callback,last_submitted,errors):
    last=0 if last_submitted is None else last_submitted['callback']
    if errors:return 'diagnostic_publication_error'
    if status=='coverage_complete_awaiting_review' and last!=callback:
        return 'coverage_ready_pending_submission'
    return status

def callback_accounting(rows):
    grouped={};malformed=[]
    for row in rows:
        key=row.get('callback')
        if type(key) is not int:malformed.append(row);continue
        grouped.setdefault(key,[]).append(row)
    # Keep duplicate evidence, but never feed duplicates into the hidden chain.
    unique=[next((r for r in group if 'student' in r),group[0]) for _,group in sorted(grouped.items())]
    return unique,{str(k):len(v) for k,v in grouped.items() if len(v)>1},malformed

def final_status(supervisor_status,summary,closed):
    agent=closed.get('status',summary.get('agent_internal_status',summary.get('status')))
    if summary.get('summary_publication_errors') or summary.get('status')=='diagnostic_publication_error':
        agent='diagnostic_publication_error'
    effective=agent if agent in ('error','comparison_failure','diagnostic_publication_error') else supervisor_status
    return agent,effective
