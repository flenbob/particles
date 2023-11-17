# Prediction of bulk properties of highly polydisperse particle packings

## Overview
This repository is a part of the Master Thesis conducted at Chalmers during spring/summer of 2023. The thesis aimed to generate any multi-component packings from which to draw properties of interest such as component-wise coordination numbers and coordination number distributions, content uniformity measurement and packing fraction.

## Features

## Usage
1. Create an input CSV-table and save in folder ```./input_tables/<table_name.csv>```. For a N-component packing the table is of size (N, 6). See formatting below (omit the headers):

| Component ID | Density [kg/m3] | Mass fraction (%/100) | Diameter mean | Diameter STD | Maximum COV (%/100) |
|:------------:|:---------------:|:-------------:|:-------------:|:------------:|:-----------:|
| 1            |        .        |       .       |       .       |       .      |      .      |
| 2            |        .        |       .       |       .       |       .      |      .      |
| ...          |        .        |       .       |       .       |       .      |      .      |
| N            |        .        |       .       |       .       |       .      |      .      |

3. Determine which ```<simulation_type>``` to employ:
   - Compression phase only: ```compression``` 
   - Compression followed by adhesion phase: ```two_phase```
4. Determine ```<# of MPI tasks>``` supported by your system
5. To run simulation on a new packing: ```run_new.py -h```, on an already existing packing file: ```run_existing.py -h``` 
6. Interact with the written HDF5 datafile. See examples for how this is done in the files 
   * ```example_content_uniformity_analysis.py```
   * ```example_coordination_number_analysis.py```
   * ```example_packing_fraction_analysis.py```

### Example use cases
#### Study content uniformity of a packing at a given total mass in $\mu$g: 
  1. Select input table ```./input_tables/<table_name.csv>```. Ensure to set maximum allowed COV in last column.
  2. Run new simulation: ```run_new.py compression table_name.csv n_tasks```. We select `compression` because it has significantly shorter simulation time than `two_phase` but yields practically the same result. Note that coordination number analysis is not valid with this setting. `n_tasks` is just number of MPI-tasks. 
   3. When simulation is finished, run ```example_content_uniformity_analysis.py``` and make sure to set path to the written HDF5 file.

#### Study packing fraciton prediction of a packing:
   1. Select input table ```./input_tables/<table_name.csv>```. Ensure to set maximum allowed COV in last column.
   2. Run ```example_packing_fraction_analysis.py``` and make sure to set path to the selected table. It is also possible to read an already existing input file ```input.txt``` containing particles, more details are covered inside the file.

#### Study coordination number of a packing
1. Select input table ```./input_tables/<table_name.csv>```. Ensure to set maximum allowed COV in last column.
2. Run new simulation: ```run_new.py two_phase table_name.csv n_tasks```. We select `two_phase` because it ensures that particles can transition from being rattlers to forming contacts with each other.
3. When simulation is finished, run ```example_coordination_number_analysis.py``` and make sure to set path to the written HDF5 file.

###



## Dependencies 

### LAMMPS 
Simulations use LAMMPS, see [Install LAMMPS](https://docs.lammps.org/Install.html). We built it inside WSL2 following instructions in [Using LAMMPS on Windows 10 with WSL](https://docs.lammps.org/Howto_wsl.html).

When running it is assumed that the LAMMPS excecutable is a ```$PATH```-variable. E.g. add the following to  `~/.bashrc`:

      export PATH=/home/<USERNAME>/lammps/build:$PATH

### Python
Python3 dependencies are found in ```./requirements.txt```.

If facing issues with OVITO Python interface dependencies (e.g. Qt plugin), try:

      sudo apt-get install libgl-dev libglfw3-dev libxkbcommon-x11-0

If you get the error "QWidget: Cannot create a QWidget without QApplication", see [Compatibility issue between python-ovito library and matplotlib](https://matsci.org/t/compatibility-issue-between-python-ovito-library-and-matplotlib/50794/1).

Packing can be visualized using OVITO, see [Install OVITO](https://www.ovito.org/manual/installation.html). Within the software, load the global dumpfile, e.g. ```./simulations/data/ID/out_global.txt``` for a given simulation ```ID```.
