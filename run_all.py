"""Execute the complete repository workflow in dependency order."""
from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parent

def main():
    """Generate data and run Parts A through D followed by the retention copilot."""
    for script in ["generate_data.py","part_a.py","part_b.py","part_c.py","part_d.py","copilot.py","update_readme.py"]:
        print(f"\n===== RUNNING {script} =====")
        subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT,check=True)

if __name__=="__main__": main()
