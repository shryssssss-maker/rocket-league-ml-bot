"""Diagnostic-only regression checks. No game packet, model fitting or live launch."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from common import HERE,V13,sha
from diagnostics import SummaryPublisher,SummaryReader,publication_view,callback_accounting,final_status

def run(folder):
    location=folder/'diagnostic_checks'
    location.mkdir(exist_ok=False)
    publisher=SummaryPublisher(location)
    first=publisher.publish(dict(status='running',callbacks=1))
    assert first is not None
    reader=SummaryReader()
    assert reader.read(location)['callbacks']==1
    before=sha(first)
    # Keep the first final JSON open (the Windows scenario that broke v1).
    with first.open('rb') as held:
        second=publisher.publish(dict(status='running',callbacks=2))
        assert second is not None and second!=first and held.read()
        assert reader.read(location)['callbacks']==2
    assert sha(first)==before
    with patch('diagnostics.os.rename',side_effect=PermissionError(13,'Injected diagnostic denial')):
        assert publisher.publish(dict(status='running',callbacks=3)) is None
    assert len(publisher.errors)==1
    assert reader.read(location)['callbacks']==2
    assert publisher.publish(dict(status='diagnostic_publication_error',callbacks=3)) is not None
    assert reader.read(location)['status']=='diagnostic_publication_error'
    assert publication_view('coverage_complete_awaiting_review',9,{'callback':8},[])=='coverage_ready_pending_submission'
    assert publication_view('coverage_complete_awaiting_review',9,{'callback':9},[])=='coverage_complete_awaiting_review'
    assert publication_view('coverage_complete_awaiting_review',9,{'callback':9},publisher.errors)=='diagnostic_publication_error'
    # Exercise the actual harness summary method with diagnostic attributes only.
    # No teacher call, GamePacket or BallPrediction is constructed.
    from live_agent import LiveHarness
    harness=object.__new__(LiveHarness)
    harness.folder=location
    harness.callback_count=1;harness.status='running';harness.last_submitted=None
    harness.original_flips=0;harness.coverage={};harness.failures=[];harness.max_analog_error=0.
    harness.packet_receipts=0;harness.prediction_receipts=0;harness.index=0;harness.team=1
    harness.planner_stage='near';harness.divergence_sequences={};harness.request_count=0;harness.pause_count=0
    harness.shadow=SimpleNamespace(flips=0,sink=SimpleNamespace(forbidden=0,discarded={}))
    harness.summary_publisher=SummaryPublisher(location/'harness')
    (location/'harness').mkdir()
    harness.diagnostic_errors=[]
    with (location/'diagnostic_errors.jsonl').open('w',encoding='utf-8') as error_log:
        harness.diagnostic_error_log=error_log
        with patch('diagnostics.os.rename',side_effect=PermissionError(13,'Injected diagnostic denial')):
            harness.summary()  # Must not escape into get_output/SDK.
        assert len(harness.diagnostic_errors)==1
        harness.summary()
    assert SummaryReader().read(location/'harness')['status']=='diagnostic_publication_error'
    # The old run is read-only evidence for error accounting, not a replay test.
    old=V13/'phase3_shadow_v1/sessions/20261005_042947_003076/callbacks.jsonl'
    old_pin=sha(old)
    records=[json.loads(line) for line in old.read_text(encoding='utf-8').splitlines()]
    unique,duplicates,malformed=callback_accounting(records)
    assert len(records)==1116 and len(unique)==1115 and duplicates=={'1115':2} and not malformed
    assert sha(old)==old_pin
    agent,effective=final_status('coverage_complete_awaiting_review',{'status':'error'}, {'status':'error'})
    assert agent==effective=='error'
    # Identity checks for every behavioral/probe function and unchanged files.
    old_dir=V13/'phase3_shadow_v1'
    old_tree=ast.parse((old_dir/'live_agent.py').read_text(encoding='utf-8-sig'))
    new_tree=ast.parse((HERE/'live_agent.py').read_text(encoding='utf-8-sig'))
    def methods(tree):
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LiveHarness')
        return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}
    old_methods,new_methods=methods(old_tree),methods(new_tree)
    unchanged=('observe_original','cover','count','transition_sequence_ids','complete','request','plan',
        '_handle_packet','_handle_ball_prediction')
    for name in unchanged:assert old_methods[name]==new_methods[name],name+' changed'
    for name in ('reference.py','student_client.py','student_worker.py'):
        assert sha(old_dir/name)==sha(HERE/name),name+' changed'
    # get_output changes affect only publication/error bookkeeping, not the
    # contiguous native teacher/shadow/observation/student decision block.
    def decision_block(tree):
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LiveHarness')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_output')
        block=next(n for n in method.body if isinstance(n,ast.Try)).body
        end=next(i for i,n in enumerate(block) if isinstance(n,ast.If)
            and isinstance(n.test,ast.Name) and n.test.id=='problems')
        return [ast.dump(n,include_attributes=False) for n in block[:end]]
    assert decision_block(old_tree)==decision_block(new_tree)
    return dict(status='passed',immutable_summary_while_old_handle_open=True,
        injected_publication_denial_nonfatal=True,actual_harness_summary_denial_nonfatal=True,
        completion_requires_final_submission=True,final_error_overrides_cached_completion=True,
        duplicate_accounting=dict(raw_lines=1116,unique_callbacks=1115,duplicate_ids=duplicates),
        behavior_probe_function_ast_identity=list(unchanged),
        teacher_shadow_observation_student_decision_block_ast_identical=True,
        byte_identical_shadow_model_worker_and_client=True,old_evidence_sha256=old_pin,
        no_live_connection=True,no_game_packet_or_prediction_constructed=True)
