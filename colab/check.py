"""
Quick status check of the Colab session.

Usage:
    colab exec -s work -f colab/check.py
"""
import os, time, subprocess, urllib.request

PROJECT = "/content/drive/MyDrive/repositories/aq_edge"
MLFLOW_PORT = 5000

print("Drive mounted:", os.path.isdir("/content/drive/MyDrive"))
print("Project found:", os.path.isdir(PROJECT))

for rel in ("main.py", "aq_edge/datautils/preprocessing.py"):
    p = os.path.join(PROJECT, rel)
    if os.path.exists(p):
        print(f"{rel}: modified", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p))))
        if rel.endswith("preprocessing.py"):
            print("  timezone fix present:", "_match_tz" in open(p).read())
    else:
        print(f"{rel}: NOT FOUND")

for d in ("artifacts", "mlflow_backup"):
    p = os.path.join(PROJECT, d)
    print(f"{d}/:", sorted(os.listdir(p))[:10] if os.path.isdir(p) else "does not exist yet")

try:
    urllib.request.urlopen(f"http://127.0.0.1:{MLFLOW_PORT}/health", timeout=2)
    print("MLflow server: running")
except Exception:
    print("MLflow server: not running")

try:
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip() or "no GPU")
except FileNotFoundError:
    print("no GPU")