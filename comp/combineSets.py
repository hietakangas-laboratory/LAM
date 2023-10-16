# -*- coding: utf-8 -*-
"""
Combine separate, projected data sets from LAM.

Created on Wed Jan 15 10:51:31 2020
@author: artoviit
-------------------------------------------------------------------------------

Used to combine cropped data sets after each samples separate vectors have been
binned, projected and counted. For example, when each sample has been cropped
to data sets corresponding to R1-2, R3, R4-5 to provide better alignment of
samples, this script combines the sets in order to perform the rest of the
analysis as one set. The different partitions of each sample NEED to be in the
same coordinate system, i.e. they can not be rotated individually, if you want
to perform cell-to-cell calculations or clustering. For plotting and statistics
each sets individual rotation does not matter.

!!!!
When using LAM to analyse the combined data set, you need to use 'Count'
without projection; set 'project' to False. This way LAM uses the values given
by the earlier projection of the split datasets.
!!!!

Vars:
----
    data_sets : dict {int: [str, int]}:
        The data sets to combine. Key is the order in which the sets are
        combined, e.g. 1->3 from anterior to posterior. Values are the path to
        the root of LAM-hierarchical directory that contains the data set, and
        number of bins that the data set has been projected to.
        
    combine_chans : list [str]:
        Names of the channels that are to be combined.
        
    savepath : pathlib.Path:
        Path to directory where the combined data set is to be saved.
        
"""
import numpy as np
import pandas as pd
import pathlib as pl
import re
import shapely.geometry as gm
from shapely.ops import linemerge

# GIVE DATA SETS:
# format: {<order of combining>: [r"<path_to_dataset_root>", <bins>]}
data_sets = {1: [r"E:\Code_folder\test_ALLSTAT_split\R1R2", 7],
             2: [r"E:\Code_folder\test_ALLSTAT_split\R2R3", 24],
             3: [r"E:\Code_folder\test_ALLSTAT_split\R3R4", 7],
             4: [r"E:\Code_folder\test_ALLSTAT_split\R4R5", 15],
             5: [r"E:\Code_folder\test_ALLSTAT_split\END", 7]
             }
combine_chans = ['DAPI', 'GFP']  #'DAPIEC', 'DAPIsmall', 'GFP', 'Prospero', 'Delta']
savepath = pl.Path(r"E:\Code_folder\test_ALLSTAT_split\Combined")


def combine(path):
    """Combine data sets created by LAM."""
    # Define output-path
    fullpath = path.joinpath('Analysis Data', 'Samples')
    fullpath.mkdir(parents=True, exist_ok=True)
    # Make sure that data is in correct order
    order = sorted(data_sets.keys())
    bins = [0]  # Set first bin
    # Determine the amount to increase each sets bins
    binN = 0  # No increase for first set

    # Create list containing start bins of each set
    for ind in order[:-1]:
        binN += data_sets.get(ind)[1]
        bins.extend([binN])

    # Create list that contains paths for each dataset
    set_paths = []
    for ind in order:
        path = pl.Path(data_sets.get(ind)[0])
        samplespath = path.joinpath('Analysis Data', 'Samples')
        set_paths.append(samplespath)

    # Get names of all samples from found sample-directories
    sample_names = set([p.name for s in set_paths for p in s.iterdir() if
                        p.is_dir()])

    # Dictionary for holding vector subsets
    vectors = {s: [0] * len(data_sets.keys()) for s in sample_names}

    # COLLECT SPLIT VECTORS:
    print('Collecting vectors')
    for ind, path in enumerate(set_paths):  # Loop each dataset

        # Test found samples against total samplelist:
        sample_paths = [p for p in path.iterdir() if p.is_dir()]
        samples = set([p.name for p in sample_paths])
        diff = sample_names - samples

        if diff:  # If missing samples
            print(f"Missing samples in {list( data_sets.keys() )[ind]}:")
            print(f"  {n}" for n in diff)

        # Gather vector data:
        vect_pat = re.compile('^vector\.csv', re.I)  # Search pattern
        for smplpath in sample_paths:
            files = [p for p in smplpath.iterdir() if p.is_file()]

            try:
                # Find, read, and store vector
                vector_path = [p for p in files if vect_pat.match(p.name)]
                vector = pd.read_csv(vector_path[0])

                # Make coordinate list of vector data
                vlist = list(zip(vector.loc[:, 'X'].astype('float'),
                             vector.loc[:, 'Y'].astype('float')))

                # Get existing data from vector dictionary and update
                vects = vectors.get(smplpath.name)
                vects[ind] = gm.LineString(vlist)
                vectors.update({smplpath.name: vects})
            except StopIteration:
                print(f"Could not find vector-file for {path.name}")

    # FIND LENGTHS OF SUB-VECTORS AND GET FULL LENGTH:
    # Loop samples and their sub-vectors
    for sample, vector_data in vectors.items():
        smplpath = fullpath.joinpath(sample)
        smplpath.mkdir(exist_ok=True)

        # Define lengths of sub-vectors
        v_lengths = [v.length for v in vector_data]

        # Summed length of sub-vectors
        full_length = sum(v_lengths)

        # Merge vectors into one and save:
        all_vects = gm.MultiLineString(vector_data) # Change to lines
        full_vector = linemerge(all_vects)  # Merge the lines
        x_d, y_d = full_vector.xy  # Coordinates of merged vector
      
        # Save vector as a DF
        vector_df = pd.DataFrame().assign(X=x_d, Y=y_d)
        vector_df.to_csv(smplpath.joinpath('Vector.csv'))

        # Update vector dict to have vector lengths to modify normDist
        vector_data = np.array(v_lengths) / full_length
        vectors.update({sample: vector_data})

    # MODIFYING OF CHANNEL DATA:
    # Loop all datasets
    for ind, path in enumerate(set_paths):
        print("Dataset {}: {}".format(ind+1, path.parents[1].name))
        # Get names and paths of samples
        samples = [[p.name, p] for p in path.iterdir() if p.is_dir()]

        # Loop the samples
        for smpl in samples:
            smplpath = fullpath.joinpath(smpl[0])

            # Modify channel data:
            for chan in combine_chans:  # Only modify desired channels

                # Create search pattern
                string = '^' + re.escape(chan + ".csv")
                regc = re.compile(string, re.I)

                # Match pattern
                paths = [p for p in smpl[1].iterdir() if
                         regc.fullmatch(p.name)]

                if not paths: # If no matching channels
                    print(f"No {chan} data found for {smpl[0]}")
                    continue

                # Read channel data and modify normalized distance to full set
                data = pd.read_csv(paths[0])
                data = modify_normdist(data, vectors.get(smpl[0]), ind)

                # Create savepath for data
                chan_savep = smplpath.joinpath(paths[0].name)

                # Combine channel subset with earlier data or save first set
                if ind != 0:
                    combine_chan(data, chan_savep, bins[ind])
                else:
                    data.to_csv(chan_savep, index=False)


def modify_normdist(data, fractions, f_index):
    """Adjust normalized distance of features to the full dataset."""
    if f_index == 0:  # If first in order, set to begin from zero
        min_v = 0

    # In later sets continue from distance defined by earlier sets
    else:
        min_v = sum(fractions[:f_index])

    # Define end of current set
    max_v = min_v + fractions[f_index]

    # Interpret values to the new interval
    data.NormDist = np.interp(data.NormDist, (0, 1), (min_v, max_v))
    return data


def combine_chan(data, filepath, add_bin):
    """Redefine 'ID' and 'DistBin' based on location in 'order'."""
    # Add earlier sets largest bin number to the current set for continuity
    data.DistBin = data.DistBin.apply(lambda x, b=add_bin: x+b)

    # Add earlier set's largest ID-number to current sets IDs
    try:
        org_data = pd.read_csv(filepath)
        maxID = org_data.ID.max()
        data.ID = data.ID.apply(lambda x, b=maxID: x+b)
    except FileNotFoundError:
        data.to_csv(filepath, index=False)
        return

    # Add new data to the old data and save
    org_data = pd.concat([org_data, data], ignore_index=True).reindex()
    org_data.to_csv(filepath, index=False)


if __name__ == '__main__':
    combine(savepath)
    print("DONE")
