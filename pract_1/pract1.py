#!/usr/bin/env python
import sys
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem  #import rdkit to get atomic mass
#==================================================Split Line==========================================================
if(len(sys.argv) != 2):
    print("Usage: pract1.py input_file")  # if incorrect input, declare the correct input format
    sys.exit(-1)  # if incorrect input, exit
#==================================================Split Line==========================================================
input_name = sys.argv[1]  # sys.argv[1] refers to the input file
data_item = [] # the list used to store the data
with open(input_name,'r') as fi: # open the input file
    for line in fi:
        words = line.split()
        for seq in range(len(words)-1): # check if valid number
            try:
                float(words[seq+1]) # numbers start from words[1]
            except ValueError:
                line_format = line.replace("\n", "")
                print("\033[91m" +
                      f"\nThe Line [{line_format}], [Column {seq+2}] is not a valid number, check your input file!\n"
                      + "\033[0m")
                sys.exit()
        data_item.append(words) # store each line(a list) into an empty list
num_atom = len(data_item) # the number of atoms equals to the length of the list
print(f"\nThere are {num_atom} atoms in file {input_name}\n")
#==================================================Split Line==========================================================
coords, names, eles = [], [], []
for i in data_item:
    s = i[0] # element name is the first element of each line
    ele = re.findall("^[A-Z]{1}[a-z]*",s) # use regular expression to get each element's symbol
    names.append(s)
    eles.append(ele)
    coords.append(i[1:]) # coordinates data start from the second element of each line
atom_coords = np.array(coords).astype(np.float64) # set numbers in the array as float for further calculation
x_coords, y_coords, z_coords = atom_coords.T # Array.T split the array by columns as X_column, Y_column, Z_column

print(f"The maximum extent of the molecule is "
      f"{max(x_coords)}, "
      f"{max(y_coords)}, "
      f"{max(z_coords)}\n")  # Use max() to get the maximum extent

print(f"The minimum extent of the molecule is "
      f"{min(x_coords)}, "
      f"{min(y_coords)}, "
      f"{min(z_coords)}\n")  # Use min() to get the minimum extent

print(f"The simple centre of gravity is "
      f"{(x_coords.sum() / num_atom)}, "
      f"{(y_coords.sum() / num_atom)}, "
      f"{(z_coords.sum() / num_atom)}\n") # simple centre calculation
#==================================================Split Line==========================================================
sum_x_coords, sum_y_coords, sum_z_coords, sum_denominator = 0, 0, 0, 0  # accumulators
pd = Chem.GetPeriodicTable()  # get the periodic table for atomic mass

for j in range(num_atom):
    try:
        pd.GetAtomicWeight(eles[j][0]) # try getting atmoic mass
    except (IndexError, RuntimeError): # incorrect element symbol like "ABC" and missing of character like "5'"
        print("\033[91m" +
              f"The line {data_item[j]} has a bad element(first column). "
              f"To get the atom binding, and the weighted centre of gravity, check your input file!\n"
              + "\033[0m")
        sys.exit()
    # in every loop, add atomic mass * coordinates to the summary for X,Y,Z
    sum_x_coords += x_coords[j] * pd.GetAtomicWeight(eles[j][0])
    sum_y_coords += y_coords[j] * pd.GetAtomicWeight(eles[j][0])
    sum_z_coords += z_coords[j] * pd.GetAtomicWeight(eles[j][0])
    sum_denominator += pd.GetAtomicWeight(eles[j][0]) # the denominator used to calculate the weighted centre

print(f"The weighted centre of gravity is "
      f"{(sum_x_coords / sum_denominator)}, "
      f"{(sum_y_coords / sum_denominator)}, "
      f"{(sum_z_coords / sum_denominator)}\n") # weighted centre of gravity

print("================Binding Position================")
#==================================================Split Line==========================================================
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x_coords, y_coords, z_coords, c='y', s=50) # 3d scatter plot
for x, y, z, label in zip(x_coords, y_coords, z_coords, names):
    ax.text(x, y, z, label, size=8) # set annotation on point
for i in range(num_atom-1): # nested loop
    for j in range(i+1, num_atom):
        if np.linalg.norm(atom_coords[i] - atom_coords[j]) < 1.8: # np.linalg.norm() solves the length of the vector
            x_pair = [atom_coords[i][0], atom_coords[j][0]] # binded pairs, used to draw lines between bindings
            y_pair = [atom_coords[i][1], atom_coords[j][1]]
            z_pair = [atom_coords[i][2], atom_coords[j][2]]
            ax.plot(x_pair, y_pair, z_pair, color='g') # draw binding lines
            print(f"Atom {names[i]} is bonded to Atom {names[j]}\n")
        else:
            continue
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("Binding Graph")
plt.show()
