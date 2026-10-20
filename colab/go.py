import os, sys, runpy, subprocess

PROJECT = "/content/drive/MyDrive/repositories/aq_edge"   # change to your folder
os.chdir(PROJECT)
sys.path[:0] = [PROJECT, PROJECT + "/src"]

# Packages Colab does not have by default (add as needed)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "mlflow"], check=True)

runpy.run_path("main.py", run_name="__main__")   # your entry script