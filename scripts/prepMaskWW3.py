# This script reads the WW3 mask generated and add
# number 2 (for boundary condition), skipping the land
# Land is 0 and Sea point is 1. 
# Read ww3 manual for more info if needed.
#
# The script also plots the area for visualization purpose 
# and save a new mask file.
#
# A mask file is available at example folder
#
import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("------------------------")
    print("Enter the mask file name")
    print("Example: python program.py file.mask")
    print("------------------------")
    sys.exit(1)  

input_arq=sys.argv[1]

print ("you entered: " + input_arq)

mask=np.loadtxt(input_arq)

latx=mask.shape[0]
lonx=mask.shape[1]

for i in range(1,2):
    for j in range(1,lonx-1):
        if mask[i,j]==1: mask[i,j]=2
        else: print ('skipping land... index: ', i, j)

for i in range(1,latx-1):
    for j in range(1,2):
        if mask[i,j]==1: mask[i,j]=2
        else: print ('skipping land... index: ',i, j)

for i in range(1,latx-1):
    for j in range(lonx-2,lonx-1):
        if mask[i,j]==1: mask[i,j]=2
        else: print ('skipping land... index: ',i,  j)

for i in range(latx-2,latx-1):
    for j in range(1,lonx-1):
        if mask[i,j]==1: mask[i,j]=2
        else: print ('skipping land... index: ',i, j)



import os

fig, ax = plt.subplots(figsize=(8, 6))
pm = ax.pcolormesh(mask, cmap='RdYlGn')
plt.colorbar(pm, ax=ax, label='Mask (0=land, 1=sea, 2=boundary)')
ax.set_title(os.path.basename(input_arq))
ax.set_xlabel('i (lon index)')
ax.set_ylabel('j (lat index)')
plt.tight_layout()

out_mask  = input_arq + '2'
out_fig   = os.path.splitext(out_mask)[0] + '_mask.png'
plt.savefig(out_fig, dpi=150)
plt.close(fig)
print(f'Figure saved: {out_fig}')

np.savetxt(out_mask, mask, fmt='%d')




