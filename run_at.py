
import argparse
import os
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Run commands in a particular image, with this implementation copied into it")
    parser.add_argument("id", metavar='ID', type=str,
                    help='ID of the image to run in')

    parser.add_argument("command", metavar='COMMAND', type=str,
                    help='Command to run')
    
    parser.add_argument("--base", type=str, dest="base", default="origin/main", help="git rev to diff from")
    parser.add_argument("--capfile", type=str, dest="capfile", help="capability file to use with hoke")

    args = parser.parse_args()

    id_pattern = r"^[0-9a-fA-F]{64}$"

    if not re.match(id_pattern, args.id):
        print(f"Not an ID: {args.id}")
        exit(1)

    git_process = subprocess.run(["git", "diff", args.base], capture_output=True, text=True, check=True)
    patch_content = git_process.stdout.replace("\"", "\\\"").replace("\n", "\\n")

    print("Patch to apply:")
    print(patch_content)

    copy_in = f"cat {patch_content} | git apply -"

    # (run (shell \"{copy_in}\"))

    script = f"((base {args.id}) (workdir /home/tmf/app) (run (shell \"{args.command}\")) )"

    print("Script to run")
    print(script)

    try:
      path = os.environ["PATH"]
      user = os.environ["USER"]
      capfile = args.capfile or os.path.abspath(f"./secrets/{user}.cap")
      cmd = ["sudo", "env", f"PATH={path}", "dune", "exec", "--", "hoke", "build", f"--connect={capfile}"]
      hoke_process = subprocess.run(cmd,
                                    capture_output=True, text=True, check=True, input=script, cwd="../tmf-pipeline")
    except subprocess.CalledProcessError as e:
        print("Error running command: ", " ".join(cmd))
        print(e)
        print(e.stderr)
        exit(2)

    output = hoke_process.stdout

    print("Output")
    print(output)

if __name__ == "__main__":
    main()