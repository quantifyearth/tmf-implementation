import argparse
from datetime import datetime
import os
import re
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Copy inputs from an image and run local code against that image")
    parser.add_argument("--id", dest='id', type=str,
                    help='ID of the image to duplicate inputs of')

    parser.add_argument("--command", dest="command", type=str,# required=True,
                    help='Override command to run')
    
    parser.add_argument("--name", dest="name", type=str,
                        help="Name for the directory to keep this in")

    parser.add_argument("--force-copy", dest="force_copy", type=bool,
                        help="Do the copy even if it looks like it has already run")

    args = parser.parse_args()

    name = args.name or datetime.utcnow().isoformat(timespec="seconds").replace(":","").replace("-","")
    user = os.environ["USER"]
    directory = f"/maps/{user}/rerunner/{name}"
    print("Directory:", directory)
    try:
        os.mkdir(directory)
        os.mkdir(f"{directory}/inputs")
        os.mkdir(f"{directory}/data")
    except:
        # Pass if the directories already exist
        pass

    # Output the dir copy lines
    if args.id:
        if not os.listdir(f"{directory}/inputs") or args.force_copy:
            # Fetch the copy images
            rom = subprocess.check_output(f"sudo cat /obuilder-zfs/result/{args.id}/rom", shell=True, text=True)
            rom_pattern = r'\(Build\(([a-z0-9]{64})' 
            matches = re.findall(rom_pattern, rom)
            include_jrc = False
            for match in matches:
                content = subprocess.check_output(f"sudo ls /obuilder-zfs/result/{match}/rootfs/home/tmf/app/data", shell=True, text=True)
                content = content.splitlines()
                ## Filter out JRC, as it's massive
                if content == ["jrc"]:
                    include_jrc = True
                    continue
                # Copy files
                print(f"Copying {' '.join(content)} ({match})...")
                # Run in a shell to get expanding of the * glob with sudo access, else nothing is found
                subprocess.check_output(f"sudo dash -c 'cp -r /obuilder-zfs/result/{match}/rootfs/home/tmf/app/data/* {directory}/inputs'", shell=True, text=True)
            
            ## Link JRC if needed
            if include_jrc:
                print("Linking JRC...")
                subprocess.check_output(f"mkdir -p {directory}/inputs/jrc/tif/products/tmf_v1 && ln -s /maps/pf341/jrc/AnnualChange {directory}/inputs/jrc/tif/products/tmf_v1/", shell=True, text=True)
        else:
            print(f"Skipping copy (use --force-copy if you want to copy again, or change --name or blow away {directory}/inputs).")

    # Find the command
    ## Rewrite ./input/ and ./data/
    command = args.command
    command = command.replace("./inputs/", f"{directory}/inputs/").replace("./data/", f"{directory}/data/").replace("python", "arkpython3")

    # Run with arkpython3
    print("Running...")
    print(command)
    try:
        print(subprocess.check_output(command, shell=True, text=True, stderr=subprocess.PIPE))
    except subprocess.CalledProcessError as e:
        print("Error running command")
        print(e.stderr or e.stdout)


# Parse the sexp

# (run (rom (((kind (Build (f2f97d72c95acef858a61e9aeed1348598db9a59a470f8bd97a57604666b2d5f /home/tmf/app/data))) (target /data/f2f97d72c95acef858a61e9aeed1348598db9a59a470f8bd97a57604666b2d5f)) ((kind (Build (bc0962fff8c4bbc0b6a4de0dd956f28d18a6b36eb9b5f2bb06e01db1f2c6dadc /home/tmf/app/data))) (target /data/bc0962fff8c4bbc0b6a4de0dd956f28d18a6b36eb9b5f2bb06e01db1f2c6dadc)) ((kind (Build (445690f6760bd88652946b463ffa810575d131546984ca7eb23e4feb29d26f3a /home/tmf/app/data))) (target /data/445690f6760bd88652946b463ffa810575d131546984ca7eb23e4feb29d26f3a)) ((kind (Build (5be243c63606222238d2966f7d4f23f8623f77f725e1d76f7a837e9796fea482 /home/tmf/app/data))) (target /data/5be243c63606222238d2966f7d4f23f8623f77f725e1d76f7a837e9796fea482)) ((kind (Build (9cb71ae33dd173deaecccbda4204059ed34a93d8ed236892ce640672d949cc9d /home/tmf/app/data))) (target /data/9cb71ae33dd173deaecccbda4204059ed34a93d8ed236892ce640672d949cc9d)) ((kind (Build (23c649863fa7c9974e81dd8f667ccfca59515216a6789766e2377022464b7a7c /home/tmf/app/data))) (target /data/23c649863fa7c9974e81dd8f667ccfca59515216a6789766e2377022464b7a7c)) ((kind (Build (5481606a68542af56112dd3dfc47dd18fad4d0dd0f4e556081687157209df6ed /home/tmf/app/data))) (target /data/5481606a68542af56112dd3dfc47dd18fad4d0dd0f4e556081687157209df6ed)) ((kind (Build (e88cf32eb1663352c1763d5d13c7354f16e3f5eab33cb21089afb5c03f92c471 /home/tmf/app/data))) (target /data/e88cf32eb1663352c1763d5d13c7354f16e3f5eab33cb21089afb5c03f92c471))))
#                    (shell "python -m methods.matching.find_potential_matches --k ./inputs/1201-k.parquet --matching ./inputs/1201-matching-area.geojson --start_year 2012 --evaluation_year 2021 --jrc ./inputs/jrc/tif/products/tmf_v1/AnnualChange --cpc ./inputs/cpc --ecoregions ./inputs/ecoregions.tif --elevation ./inputs/srtm_tif --slope ./inputs/slope --access ./inputs/accessibility_tiles -j 30 --output ./data/1201-matches.parquet"))


if __name__ == "__main__":
    main()
