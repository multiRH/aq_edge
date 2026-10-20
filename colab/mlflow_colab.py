# Start MLflow server before running Optuna
import os
import sys
import time
import shutil
import sqlite3
import subprocess
import urllib.request

PROJECT = "/content/drive/MyDrive/repositories/aq_edge"
PORT = 5000
DB = "/content/mlflow.db"
ARTIFACTS = "/content/mlartifacts"
BACKUP = f"{PROJECT}/mlflow_backup"

os.chdir(PROJECT)

# Install MLflow
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "mlflow"],
    check=True,
)

os.makedirs(ARTIFACTS, exist_ok=True)

# Restore previous tracking data, if available
if not os.path.exists(DB) and os.path.exists(f"{BACKUP}/mlflow.db"):
    shutil.copy2(f"{BACKUP}/mlflow.db", DB)
    if os.path.isdir(f"{BACKUP}/mlartifacts"):
        shutil.copytree(
            f"{BACKUP}/mlartifacts", ARTIFACTS, dirs_exist_ok=True
        )

def mlflow_up():
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/health", timeout=2
        ):
            return True
    except Exception:
        return False

# Start the server if it is not already running
if not mlflow_up():
    log = open("/content/mlflow_server.log", "w")
    subprocess.Popen(
        [
            sys.executable, "-m", "mlflow", "server",
            "--host", "127.0.0.1",
            "--port", str(PORT),
            "--backend-store-uri", f"sqlite:///{DB}",
            "--artifacts-destination", ARTIFACTS,
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )

    for _ in range(60):
        if mlflow_up():
            break
        time.sleep(1)
    else:
        log.close()
        raise RuntimeError(
            "MLflow server failed to start. Check /content/mlflow_server.log"
        )

os.environ["MLFLOW_TRACKING_URI"] = f"http://127.0.0.1:{PORT}"
print(f"MLflow server ready: {os.environ['MLFLOW_TRACKING_URI']}")