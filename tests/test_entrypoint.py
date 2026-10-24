from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class EntrypointTests(unittest.TestCase):
    def test_launch_from_outside_repository(self):
        app_path = Path(__file__).resolve().parents[1] / 'app.py'
        script = (
            'from streamlit.testing.v1 import AppTest\n'
            f'app = AppTest.from_file({str(app_path)!r}, default_timeout=30).run()\n'
            'assert not app.exception, list(app.exception)\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, '-c', script], cwd=directory,
                text=True, capture_output=True, timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
