# Forward child-process output through Python so Colab renders it in this cell.
import codecs
import os
import signal
import subprocess
import sys


def stream_command(command, *, cwd=None, env=None):
    command = list(map(str, command))
    child_env = dict(os.environ if env is None else env)
    child_env.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    with subprocess.Popen(command, cwd=cwd, env=child_env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          start_new_session=True) as process:
        try:
            # Read available bytes rather than lines, including progress bars and
            # messages that have no trailing newline yet.
            while block := os.read(process.stdout.fileno(), 4096):
                sys.stdout.write(decoder.decode(block))
                sys.stdout.flush()
            sys.stdout.write(decoder.decode(b'', final=True))
            sys.stdout.flush()
            returncode = process.wait()
        except BaseException:
            # An interrupted cell must not leave its evaluator children running.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)

# Rebind the current notebook's helper; existing script() calls use this automatically.
def run(*args):
    stream_command(args, cwd=REPO, env=ENV)

print('Live subprocess logging enabled for subsequent commands.', flush=True)
