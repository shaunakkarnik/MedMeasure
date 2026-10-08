"""Verify the actual notebook helper streams output before a child exits."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]


def notebook_helper():
    notebook = json.loads((ROOT/'notebooks/medvision_v0_kits23_baseline.ipynb').read_text())
    cell = next(c for c in notebook['cells'] if 'def stream_command(' in c['source'])
    namespace = {}
    exec(cell['source'], namespace)
    return namespace['stream_command']


class ColabLogging(unittest.TestCase):
    def test_streams_stdout_without_newline_before_completion_and_stderr(self):
        stream_command = notebook_helper()
        ready = threading.Event()
        errors = []

        class Output(io.StringIO):
            def write(self, text):
                result = super().write(text)
                if 'ready' in self.getvalue():
                    ready.set()
                return result

        output = Output()
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory)/'release'
            child = ('import os,sys,time\n'
                     'assert os.environ["PYTHONUNBUFFERED"] == "1"\n'
                     'sys.stdout.write("ready")\n'
                     f'while not os.path.exists({str(release)!r}): time.sleep(.01)\n'
                     'sys.stderr.write(" stderr-visible\\n")\n')

            def execute():
                try:
                    stream_command([sys.executable, '-c', child])
                except BaseException as error:
                    errors.append(error)

            with contextlib.redirect_stdout(output):
                thread = threading.Thread(target=execute)
                thread.start()
                received_while_running = ready.wait(timeout=5)
                still_running = thread.is_alive()
                release.touch()
                thread.join(timeout=5)
            self.assertTrue(received_while_running, output.getvalue())
            self.assertTrue(still_running)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertIn('stderr-visible', output.getvalue())

    def test_nonzero_exit_keeps_error_output_and_raises(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                notebook_helper()([sys.executable, '-c', 'import sys; print("failed"); sys.exit(7)'])
        self.assertEqual(raised.exception.returncode, 7)
        self.assertIn('failed', output.getvalue())

    def test_hotfix_matches_notebook_streaming_helper(self):
        notebook = json.loads((ROOT/'notebooks/medvision_v0_kits23_baseline.ipynb').read_text())
        source = next(c['source'] for c in notebook['cells'] if 'def stream_command(' in c['source'])
        self.assertTrue((ROOT/'notebooks/colab_logging_hotfix.py').read_text().startswith(source))
