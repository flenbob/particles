# Prediction of bulk properties of highly polydisperse particle packings

## Overview
This repository is a part of the Master Thesis conducted at Chalmers during spring/summer of 2023. The thesis aimed to generate any multi-component packings from which to draw properties of interest such as component-wise coordination numbers and coordination number distributions, content uniformity measurement and packing fraction.

## Features

## Usage
1. Create an input CSV-table and save in folder ```./input_tables/<table_name.csv>```. For a N-component the table is of size (N, 6). See formatting below (omit the headers):

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
5. To start simulation: ```python3 run.py <table_name.csv> <simulation_type> <# of MPI tasks>```
6. Interact with the written HDF5 datafile in ```./simulations/data/ID``` using the Post-Processing object in ```./classes/postprocessing.py```

## Dependencies 
Simulations uses LAMMPS, see [Install LAMMPS](https://docs.lammps.org/Install.html). Note, the simulation script assumes that the LAMMPS excecutable is a ```$PATH```-variable.

Python3 dependencies are found in ```./requirements.txt```.

Packing can be visualized using OVITO, see [Install OVITO](https://www.ovito.org/manual/installation.html). Within the software, load ```./simulations/data/ID/out_global.txt``` for a given simulation ```ID```.
