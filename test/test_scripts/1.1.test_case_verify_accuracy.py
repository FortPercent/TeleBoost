# CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=12395 bash examples/vast/run_unified_sanity_check.sh 1 2 &
# pid1=$!
# CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=12335 bash examples/vast/run_unified_sanity_check.sh 2 1 &
# pid2=$!
# CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=12365 bash examples/vast/run_unified_sanity_check.sh 2 2 &
# pid3=$!
# # wait $pid0
# # echo "finish tp1 cp1 $pid0"
# wait $pid1 
# echo "finish tp1 cp2 $pid1"
# wait $pid2
# echo "finish tp2 cp1 $pid2"
# wait $pid3
# echo "finish tp2 cp2 $pid3"
# cd test/test_scripts

# python verify_accuracy.py



# rewrite the above code to python

import subprocess
import os
import sys
import yaml
import multiprocessing

class TestRunner:
    """
    Manages the execution of test scripts, including running multiple
    sanity checks in parallel and a final verification script.
    """

    def __init__(self):
        # chdir to current file dir 
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        _taskconfs="""
- CUDA_VISIBLE_DEVICES: [0,1]
  MASTER_PORT: 12395
  arg1: 1
  arg2: 2
- CUDA_VISIBLE_DEVICES: [2,3]
  MASTER_PORT: 12335
  arg1: 2
  arg2: 1
- CUDA_VISIBLE_DEVICES: [4,5,6,7]
  MASTER_PORT: 12365
  arg1: 2
  arg2: 2
"""
        taskconfs=yaml.safe_load(_taskconfs)
        joins=[]
        for taskconf in taskconfs:
            joins.append(self.spawn_task(
                taskconf["CUDA_VISIBLE_DEVICES"], 
                taskconf["MASTER_PORT"], 
                taskconf["arg1"], 
                taskconf["arg2"]))
            
        for proc in joins:
            proc.join()
            print(f"Process {proc.pid} finished with exit code: {proc.exitcode}")

        print("all tasks finished, verify accuracy")

        # verify accuracy
        os.system("python3 verify_accuracy.py")


        
    def spawn_task(self, devices: list, master_port: int, arg1: int, arg2: int) -> multiprocessing.Process:
        """
        Spawns a new process to run the sanity check script.
        """
        current_env = os.environ.copy()
        current_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        current_env["MASTER_PORT"] = str(master_port)

        # Assuming this script is run from the project root, so 'examples/...' path is correct.
        # If run from 'test/test_scripts/', the path to the script would need adjustment (e.g., '../../examples/...')
        cmd_list = [
            "bash",
            "examples/vast/run_unified_sanity_check.sh",
            str(arg1),
            str(arg2)
        ]

        def task_target(command_list, env_vars):
            """
            Target function for the spawned process.
            Executes the shell script and prints its output/errors.
            """
            print(f"Starting task: {' '.join(command_list)} with CUDA_VISIBLE_DEVICES={env_vars['CUDA_VISIBLE_DEVICES']}, MASTER_PORT={env_vars['MASTER_PORT']}")
            try:
                process = subprocess.Popen(command_list, env=env_vars, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate() # Waits for process to terminate

                print(f"Task {' '.join(command_list)} finished with return code {process.returncode}")
                if stdout:
                    print(f"Stdout for {' '.join(command_list)}:\n{stdout}")
                if stderr:
                    # Print stderr to sys.stderr to distinguish it or if it's actual error output
                    print(f"Stderr for {' '.join(command_list)}:\n{stderr}", file=sys.stderr)
                
                if process.returncode != 0:
                    print(f"ERROR: Task {' '.join(command_list)} failed with code {process.returncode}", file=sys.stderr)

            except Exception as e:
                print(f"Exception in task_target for {' '.join(command_list)}: {e}", file=sys.stderr)


        process = multiprocessing.Process(target=task_target, args=(cmd_list, current_env))
        process.start()
        return process
    
# call it
if __name__ == "__main__":
    TestRunner()