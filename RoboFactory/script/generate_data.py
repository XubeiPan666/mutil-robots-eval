import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run RoboFactory planner to generate data.")
    parser.add_argument('config', type=str, help="Task config file to use")
    parser.add_argument('num', type=int, help="Number of trajectories to generate.")
    # parser.add_argument('name', type=str, help="The name of H5.")
    parser.add_argument("--record-dir", type=str, default="demo", help="where to save the recorded trajectories")
    parser.add_argument("--status", type=str, help="Status of the solution", default="rgb")
    parser.add_argument('--save-video', action='store_true', help="Save video of the generated trajectories.")
    parser.add_argument('--shader', type=str, default="default", help="Shader to use for rendering.")
    args = parser.parse_args()

    command = (
        f"python -m planner.run "
        f"-c \"{args.config}\" " 
        f"-o=\"rgb\" "  #rgbd可以获得深度图信息
        f"--render-mode=\"sensors\" "
        f"-b=\"cpu\" "
        f"-n {args.num} "
        # f"--traj-name \"{args.name}\""
        f"--only-count-success "
        f"--record-dir {args.record_dir} "
        f"--status {args.status} "
        + (f"--save-video" if args.save_video else "")
        + " "
        + (f"--shader {args.shader}" if args.shader != "default" else "")
    )
    print("command: ", command)
    os.system(command)

if __name__ == "__main__":
    main()
