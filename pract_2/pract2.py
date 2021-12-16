#!/usr/bin/env python
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import matplotlib.pyplot as plt # plot distribution
from io import BytesIO
import base64 # enconde base64
import os # used to call shell command
#===============================Get Initial Data===============================
def get_init(ref, lis):

    '''Use rdkit.Chem to parse the SDF file,
        Store the list of molecules in a list'''

    ref_parse = Chem.MolFromMolFile(ref)
    lis_parse = Chem.SDMolSupplier(lis)
    mols = [x for x in lis_parse]

    return ref_parse, mols
#=======================Get Name, Extent, COG, Box_Size =======================
def boxcal(mymol):

    '''For a molecule, return its name, max extent, min extent
        single centre of gravity, weighted centre of gravity, and box size'''

    name = mymol.GetProp("_Name")
    icn = 0
    tlist, x_coords, y_coords, z_coords = [], [], [], []
    sum_x_coords, sum_y_coords, sum_z_coords, sum_denominator = 0, 0, 0, 0  # accumulators
    pd = Chem.GetPeriodicTable()  # get the periodic table for atomic mass

    for atom in mymol.GetAtoms():
        # get the 3d coordinates of each atom
        pos = mymol.GetConformer().GetAtomPosition(icn)
        # Used to calculate weighted centre
        sum_x_coords += pos[0] * pd.GetAtomicWeight(atom.GetAtomicNum())
        sum_y_coords += pos[1] * pd.GetAtomicWeight(atom.GetAtomicNum())
        sum_z_coords += pos[2] * pd.GetAtomicWeight(atom.GetAtomicNum())
        sum_denominator += pd.GetAtomicWeight(atom.GetAtomicNum())
        tlist.append(pos)
        icn += 1
    coords = np.array(tlist)
    x_coords, y_coords, z_coords = coords.T

    sim_center = [round(sum(x_coords)/len(x_coords), 5), # simple centre of gravity
                  round(sum(y_coords)/len(y_coords), 5),
                  round(sum(z_coords)/len(z_coords), 5)]

    weighted_center = np.array([round((sum_x_coords / sum_denominator), 5), # weighted centre of gravity
                                round((sum_y_coords / sum_denominator), 5),
                                round((sum_z_coords / sum_denominator), 5)])

    max_ext = np.array([max(x_coords), max(y_coords), max(z_coords)]) # max extent
    min_ext = np.array([min(x_coords), min(y_coords), min(z_coords)]) # min extent
    box_size = (max_ext - min_ext).round(5) # box_size = max - min

    return name, max_ext, min_ext, sim_center, weighted_center, box_size
#===========================Get Dataframe from boxcal==========================
def get_df(single, lis):

    '''Store the result of each molecule to the dataframe
        The first row will be the single molecule,
            below are the list of molecules'''

    # Store the single molecule data to the list first
    dat_of_lis = [boxcal(single)]
    for mol in lis:
        dat_of_lis.append(boxcal(mol))
    # Create the dataframe
    df = pd.DataFrame(dat_of_lis,
                      columns=['Molecule Name',
                               'Maximum Extent',
                               'Minimum Extent',
                               'Simple Centre of Gravity',
                               'Weighted Centre of Gravity',
                               'Box Size'],
                      ).rename(index={0: 'Single Mol'})

    return df
#===========================Get Three Diagonals of Box=========================
def get_diagonal(box):

    '''Calculte three diagonals of the box'''

    box = [arr.tolist() for arr in box.to_numpy()]
    box = np.array(box)
    # Get transpose of box size
    x, y, z = box.T
    # Calculate three diagonals
    xy = np.sqrt(x ** 2 + y ** 2)
    xz = np.sqrt(x ** 2 + z ** 2)
    yz = np.sqrt(y ** 2 + z ** 2)
    diag_size = np.array([xy, xz, yz]).swapaxes(0, 1)

    return diag_size
#================================Get Similarity================================
def get_simi(box_size):

    '''Calculate the similarity between
        reference molecule and molecules
            in the list'''

    # Single mol is the first in the list
    ref_box = box_size[0].copy()
    lis_box = box_size
    distance, cosine_simi= [], []
    for i in lis_box:
        # Get distance and cosine value between vectors
        distance.append(np.linalg.norm(ref_box - i))
        cosine_simi.append(np.dot(ref_box, i)/(np.linalg.norm(ref_box)*np.linalg.norm(i)))
    distance = np.array(distance)
    cosine_simi = np.array(cosine_simi)
    weighted_cosine_simi = cosine_simi / (distance**2 + cosine_simi)
    # Get the index of the most similar molecule
    simi_mol_num = weighted_cosine_simi[1:].argsort()[-1] + 1

    return simi_mol_num, weighted_cosine_simi
#======================Similarity Distribution Graph HTML======================
def plot_distribution(label, simi):

    '''Plot the distribution of the similarity'''

    fig, axs = plt.subplots(1, figsize=(8, 6.3))
    # X axis is the index of the molecules in the list
    x_axis = np.arange(1, len(simi[1][1:]) + 1).astype(np.int64)
    plt.xlabel("Molecules in List")
    axs.set_xticks(x_axis)
    axs.plot(x_axis, simi[1][1:], 'o--', color='grey', alpha=0.5)
    axs.plot(x_axis, simi[1][1:], c='orange', drawstyle='steps-mid')
    axs.set_title(label+f' (The molecule {simi[0]} has the most similar box size)')
    # Annotate the similarity on the point
    for i, j in enumerate(simi[1][1:]):
        axs.text(i+0.47, j+0.004, str(round(j, 3)))
    # Turn the plot to base64 to embed into html
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    return encoded
#=============================HTML Style Adjustment============================
def plot_inline(plot1, plot2):
    text = '''
            <html>
                <body>
                    <h1>Box Similarity Distribution</h1>
                        <div class='two_graphs'>
                        <img src=\'data:image/png;base64,{}\'>'
                        <img src=\'data:image/png;base64,{}\'>'
                        </div>
                </body>
            </html>
    '''.format(plot1, plot2)

    return text
#=============================Molecules Graph HTML=============================
def draw_molecules(single, lis):

    '''Draw all the molecules w/ their names'''

    # Insert the single mol into
    # the first of the list of molecules
    lis.insert(0, single)
    #Get molecule names
    mols_name = [x.GetProp("_Name") for x in lis]
    for m in lis:
        AllChem.Compute2DCoords(m)
    graph = Draw.MolsToGridImage(lis, subImgSize=(450, 450), legends=mols_name)
    # Turn the plot to base64 to embed into html
    tmpfile = BytesIO()
    graph.save(tmpfile, format='PNG')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    text = '''
    <html>
        <body>
            <h1>Molecules Graphs</h1>
            <img src=\'data:image/png;base64,{}\'>'
        </body>
    </html>
    '''.format(encoded)

    return text
#============================Turn Dataframe to HTML============================
def df_to_html(df):

    '''Turn the pandas dataframe to html format'''

    text = f'''
            <html>
                <body>
                    <h1>Data of Molecules</h1>
                    {df.to_html()}
                </body>
            </html>
            '''

    return text
#=============================Knit Results to HTML=============================
def knit_html(df, plot, mol_graphs):

    '''Knit all the html texts together
        to form a html file'''

    text_file = open("result.html", "w")

    # Write the similarity algorithm in HTML
    text_file.write('''
        <html>
            <body>
                <h1>Similarity Algorithm</h1>
                <p>Two box size similarities are calculated from (1)Box Size, (2)Diagonals.
                 Both are considered as 3d vectors to get vector similarities.</p>
                <p>Then the vector similarity is calculated using distance weighted cosine similarity.</p>
                <p>The algorithm is: <strong>cosine similarity / (distance&sup2;+ cosine similarity)</strong>
                , where cosine similarity is the cosine value between two vectors.</p>
                <p>The distance weighted cosine similarity range from 0 to 1, the higher the similarity,
                the more similar the two boxes are.</p>
            </body>
        </html>
        ''')
    text_file.write(df_to_html(df))
    text_file.write(plot)
    text_file.write(mol_graphs)
    text_file.close()
#================================Main Programme================================
if __name__ == "__main__":

    # Check if the input format is correct
    if len(sys.argv) != 3:
        print("Usage: pract2.py input_1 input_2")
        sys.exit(-1)

    # Get single mol and lis of mols
    single_mol, lis_mol = get_init(sys.argv[1], sys.argv[2])

    # Get dataframe of molecule data
    df_mols = get_df(single_mol, lis_mol)

    # Get similarity and add similarity
    # to the dataframe as a new column
    similarity_1 = get_simi(df_mols["Box Size"].to_numpy())
    similarity_2 = get_simi(get_diagonal(df_mols["Box Size"]))
    df_mols["Similarity 1"] = np.array(similarity_1[-1])
    df_mols["Similarity 2"] = np.array(similarity_2[-1])

    # Get plot-styled HTML text
    plot = plot_inline(plot_distribution("Box Size Derived Similarity", similarity_1),
                       plot_distribution("Box Diagonal Derived Similarity", similarity_2))

    # Knit a html file
    knit_html(df_mols,
              plot,
              draw_molecules(single_mol, lis_mol))

    # Open html file using inbuilt firefox
    os.system("export $(dbus-launch)")
    os.system("firefox result.html ")
