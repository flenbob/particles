from src import Runner, LAMMPSScript
import argparse
from pathlib import Path

def parser():
    """Parses arguments for a simulation given an existing LAMMPS formatted packing file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("script", help=f"Selected LAMMPS script type: {[script.name for script in LAMMPSScript]}",
                    type=str)
    parser.add_argument("packing_path", help=f"Path to LAMMPS input packing file", type=str)
    parser.add_argument("n_tasks", help=f"Number of MPI-tasks to use in LAMMPS script", type=int)
    parser.add_argument("--folder", help=f"Path to folder where data for this simulation will be written to.", type=str, default=None)
    parser.add_argument("--screen", help=f"Show LAMMPS output if set to 'True'. Defaults to 'False'", type=bool, default=False)
    args = parser.parse_args()

    if args.folder is not None:
        args.folder = Path(args.folder)

    #Packing path as <Path> variable 
    args.packing_path = Path(args.packing_path)
    assert args.packing_path.exists(), f'Provided packing path {args.packing_path} is invalid.'

    #Initializing and running entire simulation
    Runner().run_existing(script=args.script, 
                          packing_path=args.packing_path,
                          n_tasks=args.n_tasks,
                          folder_path=args.folder,
                          screen=args.screen)

if __name__ == "__main__":
    parser()
