"""Installed source/descriptor audit; no game or native packet instantiation."""
import ast
import importlib.metadata
import sys
from contracts import HERE, ROOT, FEATURES, CHANNELS, NATIVE_FIELDS, sha, write, require, protected

def run():
    from rlbot import flat
    stub = ROOT/'python-example/venv/Lib/site-packages/rlbot_flatbuffers/__init__.pyi'
    source=stub.read_text(encoding='utf-8'); tree=ast.parse(source)
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='PlayerInfo')
    fields={}
    for i,node in enumerate(cls.body):
        if isinstance(node,ast.AnnAssign):
            following=cls.body[i+1] if i+1<len(cls.body) else None
            doc=following.value.value if isinstance(following,ast.Expr) and isinstance(following.value,ast.Constant) and isinstance(following.value.value,str) else ''
            fields[node.target.id]=dict(source=stub.relative_to(ROOT).as_posix(),line=node.lineno,
                                        datatype=ast.unparse(node.annotation),installed_documentation=doc.strip(),
                                        runtime_descriptor_available=hasattr(flat.PlayerInfo,node.target.id),
                                        lifecycle='Copied from packet.players[index] before original.MyBot.get_output; engine sampling phase unresolved')
    for name in NATIVE_FIELDS: require(fields[name]['runtime_descriptor_available'],'Required SDK field absent: '+name)
    versions={name:importlib.metadata.version(name) for name in ('rlbot','rlbot-flatbuffers','psutil')}
    require(versions['rlbot']=='2.0.0b55' and versions['rlbot-flatbuffers']=='0.19.0','Live SDK version changed')
    enums={name:int(getattr(flat.AirState,name)) for name in ('OnGround','Jumping','DoubleJumping','Dodging','InAir')}
    require(list(enums.values())==list(range(5)),'Unexpected enum mapping')
    audit=dict(status='source_and_descriptor_audit_complete_runtime_semantics_pending',versions=versions,python=sys.version,
               fields={k:fields[k] for k in (*NATIVE_FIELDS,'boost','demolished_timeout','is_supersonic','hitbox','hitbox_offset','latest_touch')},
               air_state_enum=enums,features=list(FEATURES),channels=list(CHANNELS),
               lifecycle_sources=[dict(path='python-example/venv/Lib/site-packages/rlbot/managers/bot.py',
                   functions=['_handle_packet','_handle_ball_prediction','_packet_processor','_run'],
                   findings='Packet and prediction handlers assign separate latest objects. Queue draining coalesces packets. _packet_processor assigns latest prediction, calls get_output, then sends PlayerInput. No atomic packet/prediction pair guarantee.'),
                   dict(path='python-example/venv/Lib/site-packages/rlbot/interface.py',functions=['handle_incoming_message','send_msg','send_bytes'],
                        findings='CorePacket.unpack dispatches handlers; send_msg packs InterfacePacket and socket.sendall. Return confirms local transport only, not engine application.')],
               causal_classification='All recorded native values are pre-current-action received packet snapshots. Current/historical semantics vary by field; engine tick boundary/reset implementation not proven.',
               last_input=dict(source_claim='Last controller input from this player',
                   unresolved=['Engine producer/build commit','Exact applied-versus-received tick phase','Equality to previous teacher submission','Proof of physical maneuver execution'],
                   runtime_plan='Compare native inputs with prior confirmed submissions at lags 1..8, and current-label coincidence separately; isolate changed-action callbacks. Never align/shift captured values.'),
               reset_caveats=['Stub documents double-jump/dodge false on ground and dodge_elapsed zero on landing; verify observationally, do not reset them ourselves.',
                              'Jumping documentation contains inconsistent 0.2-second/240-tick parenthetical; not timing authority.',
                              'Dodge direction reference frame and reset-on-ground semantics are not specified; record native components without inventing a frame.',
                              'No source guarantee that boost, demolition, or contact flags reset on every match phase transition.'],
               upstream_schema_corrob='https://github.com/RLBot/flatbuffers-schema/blob/main/schema/gamedata.fbs (not installed engine provenance)',
               protected_hashes_checked=len(protected()),synthetic_native_instances=False,live_connection=False)
    write(HERE/'source_audit.json',audit)
    print('V16 SOURCE AUDIT: installed descriptors and lifecycle verified; last_input engine semantics pending live evidence.',flush=True)

if __name__=='__main__': run()
