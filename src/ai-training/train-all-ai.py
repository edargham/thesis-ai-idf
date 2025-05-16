import subprocess
import os
import sys

#!/usr/bin/env python3


def run_script(script_path):
    """Run a Python script and handle any errors."""
    print(f"Running {script_path}...")
    try:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Create full path to the script if only filename provided
        if not os.path.isabs(script_path):
            script_path = os.path.join(current_dir, script_path)

        # Check if the script exists
        if not os.path.exists(script_path):
            print(f"Error: Script {script_path} not found.")
            return False

        # Run the script
        _ = subprocess.run([sys.executable, script_path], check=True)
        print(f"Successfully completed {script_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}. Return code: {e.returncode}")
        return False
    except Exception as e:
        print(f"An error occurred running {script_path}: {str(e)}")
        return False


def main():
    # First run lstm.py
    if not run_script("lstm.py"):
        print("Failed to run lstm.py. Aborting.")
        return

    # Then run tcn.py
    if not run_script("tcn.py"):
        print("Failed to run tcn.py.")
        return

    print("All scripts executed successfully.")


if __name__ == "__main__":
    main()
