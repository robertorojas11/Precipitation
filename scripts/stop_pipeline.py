import ee
import os
import subprocess
import sys

def kill_local_python():
    print("Stopping local Python processes...")
    # Kills any python process running pipeline_runner or drive_manager
    try:
        subprocess.run(["pkill", "-f", "pipeline_runner.py"], check=False)
        subprocess.run(["pkill", "-f", "drive_manager.py"], check=False)
        subprocess.run(["pkill", "-f", "ee.Authenticate"], check=False)
        print("Done.")
    except Exception as e:
        print(f"Error killing processes: {e}")

def cancel_gee_tasks():
    print("Initializing Earth Engine...")
    try:
        ee.Initialize()
        print("Fetching active tasks...")
        tasks = ee.data.listOperations() # New API method for listing operations
        # Or the classic way:
        tasks = ee.data.getTaskList()
        
        count = 0
        for task in tasks:
            if task['state'] in ['READY', 'RUNNING']:
                ee.data.cancelTask(task['id'])
                print(f"Cancelled GEE Task: {task['id']} ({task['description']})")
                count += 1
        
        if count == 0:
            print("No active GEE tasks found.")
        else:
            print(f"Successfully cancelled {count} tasks.")
    except Exception as e:
        print(f"Error cancelling tasks: {e}")
        print("Try running 'earthengine task cancel all' in your terminal instead.")

if __name__ == "__main__":
    kill_local_python()
    cancel_gee_tasks()
    print("\n--- Everything Stopped ---")
