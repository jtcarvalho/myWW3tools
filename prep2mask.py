import sys
import numpy as np
import matplotlib.pyplot as plt


#input_arq = input("Enter the name of mask file: ")
#print ("you entered: " + input_arq)

input_arq=sys.argv[1]

print ("you entered: " + input_arq)

mask=np.loadtxt(input_arq)

latx=mask.shape[0]
lonx=mask.shape[1]

for i in range(1,2):
    for j in range(1,lonx-1):
        if mask[i,j]==1: mask[i,j]=2
        else: print ('eh terra')

for i in range(1,latx-1):
    for j in range(1,2):
        if mask[i,j]==1: mask[i,j]=2
        else: print ('eh terra')

for i in range(1,latx-1):
    for j in range(lonx-2,lonx-1):
        if mask[i,j]==1: mask[i,j]=2
        else: print ('eh terra')

for i in range(latx-2,latx-1):
    for j in range(1,lonx-1):
        if mask[i,j]==1: mask[i,j]=2
        else: print ('eh terra')



plt.pcolormesh(mask)
plt.colorbar()
plt.show()


np.savetxt(input_arq+str('2'),mask,fmt='%d')




