import subprocess
import sys
import time
import os

def main():
    print("=========================================")
    print("⚡ CRYPTOTIME PREMIUM EXECUTION WRAPPER ⚡")
    print("=========================================")
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    setup_script = os.path.join(project_dir, "setup_data.py")
    backend_script = os.path.join(project_dir, "main.py")
    frontend_script = os.path.join(project_dir, "app.py")
    
    print("\n🔄 [1/3] Triggering Data Architecture Validation...")
    try:
        subprocess.check_call([sys.executable, setup_script], cwd=project_dir)
        print("✅ Data validation verified successfully.")
    except Exception as e:
        print(f" Data download engine failure: {e}")
        return

    print("\n [2/3] Spinning Up FastAPI Backend Grid Server...")
    backend_process = subprocess.Popen([sys.executable, backend_script], cwd=project_dir)
    
    print("⏳ Synchronizing communication ports (5 seconds)...")
    time.sleep(5)

    print("\n [3/3] Opening Interactive Visual UI Console...")
    frontend_cmd = [sys.executable, "-m", "streamlit", "run", frontend_script]
    
    try:
        subprocess.call(frontend_cmd, cwd=project_dir)
    except KeyboardInterrupt:
        print("\n Halting terminal connections safely...")
        backend_process.terminate()
        print("👋 Offline complete.")

if __name__ == "__main__":
    main()