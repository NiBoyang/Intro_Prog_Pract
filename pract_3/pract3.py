#!/usr/bin/env python
import sys
import numpy as np
from rdkit import Chem
from scipy.spatial import distance
from rdkit import DataStructs
from rdkit.Chem import AllChem, Draw
import pandas as pd

class Molecule:

    """The class of molecules. Four atoms
        and the complete descriptor are set
            to property"""

    def __init__(self, name, coords, mymol):

        """The molecule obj has the
            initial attributes of name and coordinates
                and molecule graph"""

        self.name = name
        self.coords = coords
        AllChem.Compute2DCoords(mymol)
        self.graph = Draw.MolToImage(mymol, legend=self.name)

    @property
    def cog(self):

        """Calculate the simple center of gravity"""

        # Get transpose of coordinate vectors to get split x, y, and z.
        x_coords, y_coords, z_coords = self.coords.T
        # Calculate simple centre of gravity
        sim_center = [sum(x_coords) / len(x_coords),
                      sum(y_coords) / len(y_coords),
                      sum(z_coords) / len(z_coords)]
        return sim_center

    @property
    def atom_close_to_cog(self):

        """Get the coordinates of the
            closest atom to the cog"""

        # Get distances between cog and each atom
        dist = np.array([np.linalg.norm(self.cog - i) for i in self.coords])
        # Return the closest atom's coordinates using its indices
        return self.coords[np.argmin(dist)]

    @property
    def atom_fur_to_cog(self):

        """Get the coordinates of the
            furthest atom to the cog"""

        # Get distances between cog and each atom
        dist = np.array([np.linalg.norm(self.cog - i) for i in self.coords])
        # Return the furthest atom's coordinates using its indices
        return self.coords[np.argmax(dist)]

    @property
    def atom_fur_to_atom_fur_to_cog(self):

        """Get the coordinates of the
            atom furthest to the atom
                furthest to the cog"""

        # Get distances between the atom above and each atom
        dist = np.array([np.linalg.norm(self.atom_fur_to_cog - i)
                         for i in self.coords])
        # Return the furthest atom's coordinates using its indices
        return self.coords[np.argmax(dist)]

    @property
    def descriptor(self):

        """Combine the four parts into a complete descriptor"""

        return np.concatenate((self.combine_desp(self.cog),
                               self.combine_desp(self.atom_close_to_cog),
                               self.combine_desp(self.atom_fur_to_cog),
                               self.combine_desp(self.atom_fur_to_atom_fur_to_cog)), axis=None)

    def combine_desp(self, atom):

        """Combine the below three results into
            a part of the complete descriptor"""

        def cal_mean():

            """Calculate the mean of distances, using numpy.mean"""

            return np.mean(np.array([np.linalg.norm(atom - i) for i in self.coords]))

        def cal_var():

            """Calculate the variance of distances, using numpy.var, ddof set to 1"""
            # ddof=1 means the denominator will be N-1
            return np.var(np.array([np.linalg.norm(atom - i) for i in self.coords]), ddof=1)

        def cal_skew():

            """Calculate the skewness of distances, ddof of numpy.std set to 1"""
            # calculate euclidean distance
            dist = np.array([np.linalg.norm(atom - i) for i in self.coords])
            dist_std = np.std(dist, ddof=1)
            dist_mean = np.mean(dist)
            # calculate skew
            skew = np.sum([((i - dist_mean) / dist_std) ** 3 for i in dist]) / (len(dist) - 1)
            return skew

        # return the combined result
        return np.concatenate((cal_mean(),
                               cal_var(),
                               cal_skew()), axis=None)

    @classmethod
    def from_sdf(cls, mymol):

        """From the parsed SDF file,
            get the molecule name and
                atom coordinates. Then
                    return them to the class
                        as initial attributes"""

        # Get name of the molecule
        name = mymol.GetProp("_Name")
        counter = 0
        c_list = []
        for atom in mymol.GetAtoms():
            # Get coordinate
            pos = mymol.GetConformer().GetAtomPosition(counter)
            c_list.append(pos)
            counter += 1
        coords = np.array(c_list)
        # Return name and coords to init attributes
        return cls(name, coords, mymol)

class Similarity:

    """Class of different similarity algorithms"""

    @staticmethod
    def usre(descp1, descp2):

        """Normalized Euclidean distance based on USR"""

        # calculate euclidean distance
        d = np.linalg.norm(descp1 - descp2)
        # normalize euclidean distance, between 0 and 1
        # the bigger, the more similar
        s = 1 / (1+d)
        return s

    @staticmethod
    def usrm(descp1, descp2):

        """Normalized Manhattan distance based on USR"""

        # calculate manhattan distance
        d = distance.cityblock(descp1, descp2)
        # normalize the distance
        s = 1 / (1+d)
        return s

    @staticmethod
    def usrc(descp1, descp2):

        """Distance weighted cosine based on USR"""

        d = np.linalg.norm(descp1 - descp2)
        # scipy calculates 1 - true cosine value
        # so 1 - distance.cosine gets true cosine
        c = 1-distance.cosine(descp1, descp2)
        # distance weighted cosine simi
        return c / (d**2 + c)

    @staticmethod
    def tanimoto(mol1, mol2):

        """Tanimoto similarity based on Morgan Fingerprint"""

        # Get Morgan Fingerprints using rdkit
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)
        # Calculate the tanimoto similarity
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    @staticmethod
    def dice(mol1, mol2):

        """Dice similarity based on Morgan"""

        # Get Morgan Fingerprints using rdkit
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)
        # Calculate the dice similarity
        return DataStructs.DiceSimilarity(fp1, fp2)

def generate_result(inp):

    """According to user's input ,
        generate final similarity in the form of a dataframe
         OR show the structure plot of the assigned mol"""

    # list of available structures to draw
    plot_support_list = ["plt-{}".format(i) for i in range(1, len(lismol) + 1)]
    plot_support_list.insert(0, "plt-refmol")
    # list of supported algorithms
    alg_support_list = ["usrm", "usre", "usrc", "tanimoto", "dice"]

    # if the last input not in the above two lists, show warning and exit
    if inp not in (plot_support_list + alg_support_list):
        print(f"\n Bad Input On Your Last Command, CHECK README! \n"
              f" OR molecule index out of range!! "
              f"\n There are {len(plot_support_list)-1} molecules in"
              f" your list of molecules\n")
        sys.exit()

    ###
    # when user call the similarity algorithm
    elif inp in alg_support_list:

        # get attribute from Similarity
        # i.e., define which alg to use
        attr = getattr(Similarity, inp)

        # if the input starts with 'usr'
        # it means ultrafast shape recognition will be used
        if inp.startswith("usr"):
            simi = np.array([attr(refmol.descriptor, lismol[i].descriptor)
                             for i in range(len(lismol))])
        # else the morgan fingerprint will be used
        else:
            simi = np.array([attr(Chem.MolFromMolFile(sys.argv[1]), x)
                             for x in Chem.SDMolSupplier(sys.argv[2])])

        # get names of molecules in lfp
        lismol_name = [lismol[i].name
                       for i in range(len(lismol))]

        # Create a dataframe including mol names and similarity,
        # sorted by similarity
        df = pd.DataFrame({"Molecule Name": lismol_name,
                           inp: simi}). \
            sort_values(by=[inp], ascending=False)
        # set index starts from 1,
        # then index indicate the position of the molecule in the list
        df.index += 1
        print(f"\n{df}\n")

    ###
    # when user call the plotting function
    elif inp.startswith("plt-"):

        def remove_prefix(input_string, prefix):

            """function used to remove the 'plt-' prefix"""

            if prefix and input_string.startswith(prefix):
                return input_string[len(prefix):]
            return input_string

        suffix = remove_prefix(inp, "plt-")

        # when user request refmol picture
        if suffix == "refmol":
            # call graph attribute
            eval(suffix).graph.show()
        # when user request mols in the list
        else:
            # since the indice starts from 1, thus -1
            var = int(suffix)-1
            lismol[var].graph.show()

if __name__ == "__main__":

    #Check the input format
    if len(sys.argv) != 4:
        print("Usage: pract3.py input_1 input_2 algorithm OR pract3.py input_1 input_2 plt-xxx")
        sys.exit(-1)

    # Read the first input,
    # create a class instance for the reference molecule
    refmol = Molecule.from_sdf(Chem.MolFromMolFile(sys.argv[1]))

    # Read the list of molecules
    # create instances for all the molecules in lfp, and stored in a list
    lismol = [Molecule.from_sdf(obj) for obj in
                 [x for x in Chem.SDMolSupplier(sys.argv[2])]]

    # call final result according to user's input
    generate_result(sys.argv[3])
