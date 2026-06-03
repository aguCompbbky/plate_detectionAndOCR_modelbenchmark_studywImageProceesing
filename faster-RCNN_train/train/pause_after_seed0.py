import json
import time
import os
import signal

REPORT_PATH = "/home/muk/Masaüstü/python_scripts/plaka2/faster-RCNN_train/metrics_report.json"
PID_TO_KILL = 0

def get_pid():
    # Find the python process running train_frcnn.py
    import subprocess
    try:
        output = subprocess.check_output(["pgrep", "-f", "train/train_frcnn.py"]).decode().strip().split("\n")
        # Ensure it's not this watcher script
        return int(output[0])
    except:
        return None

def main():
    target_pid = get_pid()
    if not target_pid:
        print("Could not find training process PID.")
        return
        
    print(f"Monitoring process {target_pid} for Seed 0 completion...")
    
    while True:
        if os.path.exists(REPORT_PATH):
            try:
                with open(REPORT_PATH, "r") as f:
                    data = json.load(f)
                    
                if "seeds" in data and "seed_0" in data["seeds"]:
                    # Seed 0 is fully complete and saved!
                    print("Seed 0 finished! Sending SIGINT to training process.")
                    os.kill(target_pid, signal.SIGINT)
                    # Modify the training script to remove seed 0 for next time
                    break
            except Exception as e:
                pass
        time.write("Waiting...\n")
        time.sleep(30)

if __name__ == "__main__":
    main()
