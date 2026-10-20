"""
Run aq_edge on a Colab VM (code is read from the mounted Google Drive).

Usage (from the repo, in Ubuntu/WSL):
    colab new -s work --gpu T4
    colab drivemount -s work
    colab exec -s work -f colab/go.py
"""
import os, sys, time, shutil, sqlite3, threading, subprocess, importlib, runpy
import urllib.request

# ---- settings ---------------------------------------------------------------
PROJECT = "/content/drive/MyDrive/repositories/aq_edge"
ENTRY = "main.py"                                   # entry script inside PROJECT
PIP_PACKAGES = ["neuralhydrology", "mlflow"]        # packages Colab does not have
MLFLOW_PORT = 5000                                  # must match your code/config
MLFLOW_DB = "/content/mlflow.db"                    # local disk (SQLite is unreliable on Drive)
MLFLOW_ARTIFACTS = "/content/mlartifacts"
BACKUP = PROJECT + "/mlflow_backup"                 # copy kept in Drive
BACKUP_EVERY = 300                                  # seconds between background backups
# -----------------------------------------------------------------------------

os.chdir(PROJECT)
sys.path[:0] = [PROJECT, PROJECT + "/src"]

if PIP_PACKAGES:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PIP_PACKAGES], check=True)


def backup():
    """Copy MLflow data to Drive (safe copy of the SQLite file)."""
    os.makedirs(BACKUP, exist_ok=True)
    if os.path.exists(MLFLOW_DB):
        snap = "/content/mlflow_snapshot.db"
        src, dst = sqlite3.connect(MLFLOW_DB), sqlite3.connect(snap)
        with dst:
            src.backup(dst)
        src.close(); dst.close()
        shutil.copy(snap, BACKUP + "/mlflow.db")
    if os.path.isdir(MLFLOW_ARTIFACTS):
        shutil.copytree(MLFLOW_ARTIFACTS, BACKUP + "/mlartifacts", dirs_exist_ok=True)


def mlflow_up():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{MLFLOW_PORT}/health", timeout=2)
        return True
    except Exception:
        return False


# Restore earlier experiments from Drive on a fresh VM
if not os.path.exists(MLFLOW_DB) and os.path.exists(BACKUP + "/mlflow.db"):
    shutil.copy(BACKUP + "/mlflow.db", MLFLOW_DB)
    if os.path.isdir(BACKUP + "/mlartifacts"):
        shutil.copytree(BACKUP + "/mlartifacts", MLFLOW_ARTIFACTS, dirs_exist_ok=True)
    print("[go] restored MLflow data from Drive")

# Start the MLflow server in the background (skipped if already running)
if not mlflow_up():
    log = open("/content/mlflow_server.log", "w")
    subprocess.Popen(
        [sys.executable, "-m", "mlflow", "server",
         "--host", "127.0.0.1", "--port", str(MLFLOW_PORT),
         "--backend-store-uri", f"sqlite:///{MLFLOW_DB}",
         "--artifacts-destination", MLFLOW_ARTIFACTS],
        stdout=log, stderr=subprocess.STDOUT,
    )
    for _ in range(90):
        if mlflow_up():
            break
        time.sleep(1)
    else:
        print(open("/content/mlflow_server.log").read())
        raise RuntimeError("MLflow server did not start")
print(f"[go] MLflow server up on port {MLFLOW_PORT}")
os.environ.setdefault("MLFLOW_TRACKING_URI", f"http://127.0.0.1:{MLFLOW_PORT}")

# Background backup to Drive (started once per session)
if not getattr(sys, "_aq_backup_started", False):
    def _loop():
        while True:
            time.sleep(BACKUP_EVERY)
            try:
                backup()
            except Exception as e:
                print("[go] backup failed:", e)
    threading.Thread(target=_loop, daemon=True).start()
    sys._aq_backup_started = True

# Force Python to re-read your code (the session keeps imported modules in memory)
for name in list(sys.modules):
    if name == "aq_edge" or name.startswith("aq_edge."):
        del sys.modules[name]
importlib.invalidate_caches()

# Show which version of the code Colab sees (helps to detect unsynced Drive files)
for rel in (ENTRY, "aq_edge/datautils/preprocessing.py"):
    p = os.path.join(PROJECT, rel)
    if os.path.exists(p):
        print(f"[go] {rel} last modified:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p))))

# Run the project
sys.argv = [ENTRY]
try:
    runpy.run_path(ENTRY, run_name="__main__")
finally:
    backup()
    print("[go] MLflow data saved to", BACKUP)