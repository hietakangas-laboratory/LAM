# -*- coding: utf-8 -*-
"""
Split data sets by coordinate points to allow LAM analysis of sub-regions.

Created on Thu Feb  6 12:14:14 2020

@author: artoviit
-------------------------------------------------------------------------------
DESCRIPTION:
-----------
    Splits a data set and its vectors based on user given points after their
    projection during regular LAM 'Count'-functionality. In its simplicity, the
    projected points determine cut-off points in the data set and its vectors.
    The script creates LAM-hierarchical folders for each of the sub-sections of
    the data set, that can then be analysed separately.

    Intended use is to input biologically identifiable cut-off points, i.e. co-
    ordinates of region borders as seen on the microscopy image. This allows
    proper alignment of the different regions between the samples, and enables
    data collection from specific regions.

    After projecting each of the subsets, you can use combineSets.py to re-
    combine the sets to possibly create better comparisons between sample
    groups when there is variability in region proportions. HOWEVER, this
    breaks the sample-specific equivalency of the bins at cut-off points, and
    should be handled with great care.

USAGE:
-----
    Perform a LAM projection (i.e. 'Count') using a full data set, with each
    cut-off point determined by a separate 'channel', similar to a MP. After
    projection, define the required variables below and run the script. The
    script creates N+1 directories, where N is the number of defined cutpoints.
    The directories will contain the data of all samples cut followingly: from
    the first bin to the first cut-off point (directory named by the point),
    from that point to the next cut-off point (named by the second cut point),
    ad infinitum, and then from the final cut point to the final bin of the
    samples (named as 'END').

    To analyze the split sets with LAM, make sure that settings.header_row is
    set to zero (or the alternative in GUI).

DEPS:
----
    - Shapely >= 1.7.0
    - Pandas
    - Numpy
    - Pathlib

VARS:
----
    ROOT : pathlib.Path
        The root analysis directory of the data set to split.

    SAVEDIR : pathlib.Path
        The destination directory for the split data sets. Each split set will
        be saved into a directory defined by CUT_POINTS.

    CHANNELS : list [str]
        List of all the channels that are to be split. All channels out of this
        list are disregarded and will not be copied to the SAVEDIR.

    CUT_POINTS : list [str]
        List of the channel names of the cut-off points. The order of the list
        determines the order of the data split, e.g. ["R2R3", "R3R4"]
        will cut the data set from smallest value along the vector to R2R3,
        from R2R3 to R3R4, and from R3R4 to the end of the vector.

    TOTAL_BIN : int
        The wanted bin number of the full data set. The script gives suggestion
        for how many bins each sub-set should be analyzed on in order to retain
        better biological relevancy. The suggestion is calculated from the
        total vector lengths and the lengths of the divided vectors.

    CONT_END : BOOL
        If True, continue the data set to the end of the vector after the final
        cut-off point. If False, the data after the final cut-off point is not
        used.
"""
import numpy as np
import pandas as pd
import pathlib as pl
import shapely.geometry as gm
import shapely.ops as op

ROOT = pl.Path(r"E:\Code_folder\DSS")
SAVEDIR = pl.Path(r"E:\Code_folder\DSS_62bin_split")
CUT_POINTS = ["R2R3", "R3R4"]  # "R4R5"]
CHANNELS = ['DAPI', 'GFP', 'Delta', 'Prospero', 'DAPIbig', 'DAPIpienet']
# Number of bins for whole length of samples. Script gives recommendation for
# numbers of bins for each split region based on this value:
TOTAL_BIN = 62
CONT_END = True


def get_cut_points(CUT_POINTS, samplepath):
    cut_distances = []
    for point in CUT_POINTS:
        data = pd.read_csv(samplepath.joinpath("{}.csv".format(point)))
        dist = data.NormDist.iat[0]
        cut_distances.append(dist)
    if CONT_END:
        cut_distances.append(1.0)
    return cut_distances


def cut_data(data, cut_distances):
    idx_cuts = []
    for i, point in enumerate(cut_distances):
        if i == 0:
            idx_before = data.loc[(data.NormDist <= point)].index
        else:
            idx_before = data.loc[(data.NormDist <= point) &
                                  (data.NormDist > cut_distances[i-1])].index
        idx_cuts.append(idx_before)
    return idx_cuts


def read_vector(vector_path):
    if vector_path.name == "Vector.csv":
        vector_df = pd.read_csv(vector_path)
    # If vector is user-generated with ImageJ line tools:
    elif vector_path.name == "Vector.txt":
        vector_df = pd.read_csv(vector_path, sep="\t", header=None)
        vector_df.columns = ["X", "Y"]
    Vect = list(zip(vector_df.loc[:, 'X'].astype('float'),
                    vector_df.loc[:, 'Y'].astype('float')))
    vector = gm.LineString(Vect)
    
    return vector


def cut_vector(vector_path, cut_distances):
    sub_vectors = []
    vector = read_vector(vector_path)
    for i, dist in enumerate(cut_distances):
        if i == 0:
            sub_v = op.substring(vector, 0, dist, normalized=True)
        else:
            sub_v = op.substring(vector, cut_distances[i-1], dist,
                                 normalized=True)
        sub_vectors.append(sub_v)
    return sub_vectors


def save_vector(vector_dir, sub_vector):
    vector_df = pd.DataFrame(np.vstack(sub_vector.xy).T, columns=['X', 'Y'])
    vector_df.to_csv(vector_dir.joinpath("Vector.csv"), index=False)


def get_sample_data(samplepath, POINTS, length_data):
    cut_distances = get_cut_points(CUT_POINTS, samplepath)
    file_paths = samplepath.glob("*.csv")
    try:
        vector_path = next(samplepath.glob("Vector.*"))
        sub_vectors = cut_vector(vector_path, cut_distances)
    except StopIteration:
        sub_vectors = None
    sample_name = path.stem
    length_data.save_substrings(sample_name, sub_vectors)
    # Cutting and saving of new data:
    for datafile in file_paths:
        if datafile.stem in CHANNELS:
            data = pd.read_csv(datafile)
            idx_cuts = cut_data(data, cut_distances)
            for i, cut in enumerate(idx_cuts):
                vector_dir = SAVEDIR.joinpath(POINTS[i], "Analysis Data",
                                              "Samples", sample_name)
                vector_dir.mkdir(parents=True, exist_ok=True)
                name = "Position.csv".format()
                chan_dir = '{}_{}_Stats'.format(sample_name, datafile.stem)
                data_dir = SAVEDIR.joinpath(POINTS[i], sample_name, chan_dir)
                data_dir.mkdir(parents=True, exist_ok=True)
                data.loc[cut, :].to_csv(data_dir.joinpath(name), index=False)
                save_vector(vector_dir, sub_vectors[i])


class vector_lengths:
    
    def __init__(self, SAMPLES, POINTS):
        sample_list = [s.stem for s in SAMPLES]
        group_list = set(sorted([s.split('_')[0] for s in sample_list]))
        # Data variables:
        self.lengths = pd.DataFrame(columns=sample_list, index=POINTS)
        self.averages = pd.DataFrame(index=POINTS)
    
    def save_substrings(self, sample_name, sub_vectors):
        for i, sub in enumerate(sub_vectors):
            self.lengths.loc[:, sample_name].iat[i] = sub.length
    
    def find_averages(self):
        self.lengths.loc['Group', :] = [s.split('_')[0] for s in
                                        self.lengths.columns]
        grouped = self.lengths.T.groupby(by='Group', axis=0)
        for group in grouped.groups:
            self.averages = pd.concat(
                [self.averages, grouped.get_group(group).mean()], axis=1)
        self.averages.columns = grouped.groups.keys()

    def find_bins(self):
        pass


if __name__ == '__main__':
    SAMPLES = [p for p in ROOT.joinpath("Analysis Data", "Samples").iterdir()
               if p.is_dir()]
    POINTS = CUT_POINTS.copy()
    if CONT_END:
        POINTS.append("END")
    length_data = vector_lengths(SAMPLES, POINTS)
    for path in SAMPLES:
        print(path.name)
        get_sample_data(path, POINTS, length_data)
    print("Finding averages ...")
    length_data.find_averages()
    print("\n- Average lengths:\n", length_data.averages, "\n")

    summed_length = length_data.averages.sum(axis=0)
    bin_suggestion = pd.DataFrame(index=length_data.averages.index,
                                  columns=length_data.averages.columns)
    for grp in summed_length.index:
        fraction = length_data.averages.loc[:, grp] / summed_length[grp]
        bins = TOTAL_BIN * fraction
        bin_suggestion.loc[:, grp] = bins
    if len(bin_suggestion.columns) > 1:
        bin_suggestion.loc['TOTAL', :] = bin_suggestion.sum().values
        bin_suggestion.loc[:, 'MEAN'] = bin_suggestion.mean(
            axis=1).values.astype(int)
    print("- Bin fractions:\n", bin_suggestion)
    bin_suggestion.to_csv(SAVEDIR.joinpath('Bin_suggestions.csv'))
    print('\n\nSPLIT DONE')
