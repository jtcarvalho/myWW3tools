## Pre-processing tools for WW3

This repository contains my tools for WW3.

1) Generate regular grid with python
2) Generate boundary files to be used on a parent grid as Bondary Condition (ongoing task....)

Instructions:

1) Edit conf_bounds.yaml
2) python genWW3grid_fromBounds.py
3) python prepMasWW3.py {area}.mask   # if you want to generate boundary ids on grid
4) python computeObstrWW3.py {area}.mask  # if you want to add obstacles on grid
5) python prepBC2WW3.py {area}.mask2 # To generate boundary condition points list to WW3 run
