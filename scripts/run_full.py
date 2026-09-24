import subprocess
import sys

subprocess.run([
    sys.executable, "scripts/run_pilot.py",
    "--config", "configs/default.yaml",
    "--max-windows", "1000",
    "--models", "naive", "linear", "ttm", "timesfm",
    "--device", "cpu",
    "--batch-size", "64",
], check=True)
