## Pre and Post processing tools for WW3

### WW3 regular grid generation 

I'm using gridgen (https://github.com/NOAA-EMC/gridgen) to generate regular grids. Its a matlab program, but executing at my local
machine with octave. Steps:

1) Generate a conda env for octave and install (pip install octave)
2) Need to install netcdf4 for octave
3) Download etopo data 
4) Load octave on terminal and 
5) Edit and execute run "create_area.m"
