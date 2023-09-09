import argparse
from pathlib import Path

from src import LAMMPSScript, Runner


def parser():
    """Parses arguments for a simulation where a new packing is created.
    """
    #Get list of available input tables
    path_to_input_tables = Path('./input_tables')
    csv_tables = list(path_to_input_tables.glob('*.csv'))

    #Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("script", help=f"Selected LAMMPS script type. Available are: {[script.name for script in LAMMPSScript]}",
                    type=str)
    parser.add_argument("table", help=f"Selected input CSV table name. Available examples are: {[csv_table.name for csv_table in csv_tables]}", type=str)
    parser.add_argument("n_tasks", help=f"Number of MPI-tasks to use in LAMMPS script", type=int)
    parser.add_argument("--folder", help=f"Path to folder where data for this simulation will be written to. If not set will create a folder in ./simulations/data/", type=str, default=None)
    parser.add_argument("--screen", help=f"Show LAMMPS output if set to 'True'. Defaults to 'False'", type=bool, default=False)
    args = parser.parse_args()

    if args.folder is not None:
        args.folder = Path(args.folder)

    #Responsible for initializing and running entire simulation
    Runner().run_new(script=args.script, 
                     table=args.table, 
                     n_tasks=args.n_tasks, 
                     folder_path=args.folder,
                     screen=args.screen)

if __name__ == "__main__":
    parser()