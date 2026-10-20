import os, sys, time, subprocess, runpy, shutil, urllib.request

PROJECT = "/content/drive/MyDrive/repositories/aq_edge"
MLFLOW_PORT = 5000
MLFLOW_DB = "/content/mlflow.db"          # local disk: SQLite is unreliable on Drive
MLFLOW_ARTIFACTS = "/content/mlartifacts"
BACKUP = PROJECT + "/mlflow_backup"        # copied to Drive at the end

os.chdir(PROJECT)
sys.path[:0] = [PROJECT, PROJECT + "/src"]

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "mlflow"], check=True)

def mlflow_up():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{MLFLOW_PORT}/health", timeout=2)
        return True
    except Exception:
        return False

# Start the MLflow server in the background (skip if already running)
if not mlflow_up():
    log = open("/content/mlflow_server.log", "w")
    subprocess.Popen(
        [sys.executable, "-m", "mlflow", "server",
         "--host", "127.0.0.1", "--port", str(MLFLOW_PORT),
         "--backend-store-uri", f"sqlite:///{MLFLOW_DB}",
         "--artifacts-destination", MLFLOW_ARTIFACTS],
        stdout=log, stderr=subprocess.STDOUT,
    )
    for _ in range(60):
        if mlflow_up():
            break
        time.sleep(1)
    else:
        print(open("/content/mlflow_server.log").read())
        raise RuntimeError("MLflow server did not start")

os.environ.setdefault("MLFLOW_TRACKING_URI", f"http://127.0.0.1:{MLFLOW_PORT}")

try:
    runpy.run_path("main.py", run_name="__main__")   # your entry script
finally:
    # Save the experiment data to Drive before the VM disappears
    os.makedirs(BACKUP, exist_ok=True)
    if os.path.exists(MLFLOW_DB):
        shutil.copy(MLFLOW_DB, BACKUP + "/mlflow.db")
    if os.path.exists(MLFLOW_ARTIFACTS):
        shutil.copytree(MLFLOW_ARTIFACTS, BACKUP + "/mlartifacts", dirs_exist_ok=True)