"""Bounded serial IPC; student has no access to the game's controller socket."""
import json
import queue
import subprocess
import threading
import time
from common import HERE,TRAIN_PYTHON,CHECKPOINT_PIN

class StudentClient:
    def __init__(self,folder):
        self.errors = (folder/'student_stderr.log').open('w',encoding='utf-8')
        self.process = subprocess.Popen([str(TRAIN_PYTHON),'-u','-B',str(HERE/'student_worker.py')],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.errors,text=True,
            encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.messages = queue.Queue()
        def receive():
            try:
                for line in self.process.stdout:
                    self.messages.put(json.loads(line))
            except Exception as e: self.messages.put(dict(error=repr(e)))
            finally: self.messages.put(dict(error='Worker exited'))
        threading.Thread(target=receive,daemon=True).start()
        try:
            self.ready = self.take(15.)
            if self.ready.get('status')!='ready' or self.ready.get('checkpoint_sha256')!=CHECKPOINT_PIN:
                raise RuntimeError('Worker handshake mismatch')
        except BaseException:
            self.close()
            raise

    def take(self,seconds):
        try: value = self.messages.get(timeout=seconds)
        except queue.Empty: raise TimeoutError('Diagnostic student worker timeout')
        if 'error' in value: raise RuntimeError(value['error'])
        return value

    def infer(self,callback,obs):
        start = time.perf_counter()
        self.process.stdin.write(json.dumps(dict(callback=callback,observation13=obs),allow_nan=False)+'\n')
        self.process.stdin.flush()
        result = self.take(2.)
        if result['callback'] != callback: raise RuntimeError('Worker callback mismatch')
        result['ipc_seconds'] = time.perf_counter()-start
        return result

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"command":"close"}\n'); self.process.stdin.flush()
                self.process.wait(timeout=2.)
            except (OSError,subprocess.TimeoutExpired):
                self.process.terminate(); self.process.wait(timeout=2.)
        self.errors.close()
