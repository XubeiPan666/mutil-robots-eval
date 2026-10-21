import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run RoboFactory planner with task config input.")
    parser.add_argument('config', type=str, help="Task config file to use")
    parser.add_argument('status', type=str, help="Status of the task")
    args = parser.parse_args()

    command = (
        f"python -m planner.run "
        f"-c \"{args.config}\" " 
        f"--render-mode=\"human\" "
        f"-b=\"cpu\" "
        f"-n 1 "
        f"--vis "
        f"--status \"{args.status}\""
    )

    os.system(command)

if __name__ == "__main__":
    main()
