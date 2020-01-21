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

When using LAM to analyse the combined data set, you need to use 'Count'
without projection; set 'project' to False. This way LAM uses the values given
by the earlier projection.

Vars:
----
    data_sets - dict {int: [str, int]}:
        The data sets to combine. Key is the order in which the sets are
        combined, e.g. 1->3 from anterior to posterior. Values are the path to
        the root of LAM-hierarchical directory that contains the data set, and
        number of bins that the data set has been projected to.
        
    combine_chans - list [str]:
        Names of the channels that are to be combined.
        
    savepath - pathlib.Path:
        Path to directory where the combined data set is to be saved.
        
"""

import pandas as pd
import pathlib as pl
import re

# GIVE DATA SETS:
# format: {<order>: [r"<path_to_dataset_root>", <bins>]}
data_sets = {1: [r"P:\h919\hietakangas\Arto\R1_R2", 25],
             2: [r"P:\h919\hietakangas\Arto\R4_R5", 25]
             }
combine_chans = ['DAPI', 'GFP', 'Prospero', 'Delta']
savepath = pl.Path(r"P:\h919\hietakangas\Arto\combined")


def combine(path):
    """Combine data sets created by LAM."""
    fullpath = path.joinpath('Analysis Data', 'Samples')
    order = sorted(data_sets.keys())
    bins = [0]
    # Determine the amount to increase each sets bins
    for ind in order[:-1]:
        binN = data_sets.get(ind)[1]
        bins.extend([binN])
    set_paths = []
    for ind in order:
        path = pl.Path(data_sets.get(ind)[0])
        samplespath = path.joinpath('Analysis Data', 'Samples')
        set_paths.append(samplespath)
    for ind, path in enumerate(set_paths):
        print("Data set {}: {}".format(ind+1, path.parents[1].name))
        samples = [[p.name, p] for p in path.iterdir() if p.is_dir()]
        for smpl in samples:
            smplpath = fullpath.joinpath(smpl[0])
            smplpath.mkdir(parents=True, exist_ok=True)
            for chan in combine_chans:
                regc = re.compile("^{}.csv".format(chan), re.I)
                paths = [p for p in smpl[1].iterdir() if
                         regc.fullmatch(p.name)]
                if not paths:
                    continue
                chan_savep = smplpath.joinpath(paths[0].name)
                data = pd.read_csv(paths[0])
                if ind != 0:
                    combine_chan(data, chan_savep, bins[ind])
                else:
                    data.to_csv(chan_savep, index=False)


def combine_chan(data, filepath, add_bin):
    """Redefine 'ID' and 'DistBin' based on location in 'order'."""
    data.DistBin = data.DistBin.apply(lambda x, b=add_bin: x+b)
    try:
        org_data = pd.read_csv(filepath)
        maxID = org_data.ID.max()
        data.ID = data.ID.apply(lambda x, b=maxID: x+b)
    except FileNotFoundError:
        data.to_csv(filepath, index=False)
        return
    org_data = pd.concat([org_data, data], ignore_index=True).reindex()
    org_data.to_csv(filepath, index=False)


if __name__ == '__main__':
    combine(savepath)
