# -*- coding: utf-8 -*-
"""
Split data sets by coordinate points to allow LAM analysis of sub-regions.

Created on Thu Feb  6 12:14:14 2020

@author: Arto I. Viitanen
-------------------------------------------------------------------------------
DESCRIPTION:
-----------
    Splits a data set and its vectors based on user given points after their
    projection during regular LAM 'Count'-functionality, or alternatively split
    based on given bins for samples. In its simplicity, the projected points
    determine cut-off points in the data set and its vectors. The script
    creates LAM-hierarchical folders for each of the sub-sections of the
    dataset that can then be analysed separately.

    Intended use is to input biologically identifiable cut-off points, i.e. co-
    ordinates of region borders as seen on the microscopy image. This allows
    proper alignment of the different regions between the samples, and enables
    data collection from specific regions. Alternatively, the border detection
    of LAM outputs indexes of each sample at detected average group peaks to
    "Borders_peaks.csv" that can be used for cutting each sample.

    After projecting each of the subsets, you can use combineSets.py to re-
    combine the sets to possibly create better comparisons between sample
    groups when there is variability in region proportions. HOWEVER, this
    breaks the sample-specific equivalency of the bins at cut-off points, and
    should be handled with care.

USAGE:
-----
    1. Using projected coordinates:
    -------------------------------
    Perform a LAM projection (i.e. 'Count') using a full data set, with each
    cut-off point determined by a separate 'channel', similar to a MP. After
    projection, define the required variables below and run the script.
    -------------------------------

    2. Using bins of samples:
    -------------------------
    Perform LAM 'Count' with border detection amd get border location indices
    of samples from "Borders_peaks.csv" (see variables SAMPLE_BINS and
    BIN_PATH below). Alternatively, just run 'Count' and provide self-defined
    bins to the file that is pointed to by BIN_PATH.
    -------------------------

    By default, N+1 directoriesare created, where N is the number of defined
    cutpoints. The directories will contain the data of all samples cut
    followingly: from the first bin to the first cut-off point (directory named
    by the point), from the first point to the next cut-off point (named by the
    second cut point), etc., and then from the final cut point to the final bin
    of the samples (named as 'END').

    !!!
    To analyze the split sets with LAM, make sure that settings.header_row is
    set to zero (or the alternative setting in GUI).
    !!!

    FOR SPLIT-AND-COMBINE:
        After splitting, you can use a batch-file (or similar) to just run the
        'Count'-option for each of the split sets to get projections and counts
            (see LAM-master/src/examples/run_split_count.bat).
        After running 'Count' for each of the split sets, you can use the
        CombineSets.py-script.

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
        for how many bins each sub-set should be analyzed with in order to
        retain better biological relevancy. The suggestion is calculated from
        the total vector lengths and the lengths of the divided vectors. Note
        that the suggestions are given in floats to allow user decision, but
        the used bin numbers in analysis must naturally be integers.

    CONT_END : BOOL
        If True, continue the data set to the end of the vector after the final
        cut-off point. If False, the data after the final cut-off point is not
        used.

    SAMPLE_BINS : BOOL
        If True, cut the dataset based on bins and not projected points. Each
        sample must be given each cut point in the csv-file pointed by
        BIN_PATH.

    BIN_PATH : pathlib.Path
        Path to csv-file that contains each cut-off bin of each sample. The bin
        value must be the bin number of the SAMPLE and not the bin number in
        the anchored matrix that contains all samples. The border detection of
        LAM automatically outputs the sample indexes of found borders into the
        peak-file "Borders_peaks.csv". Each sample must have a value for each
        cut location for the split to work. The file must have samplenames at
        the columns and names of cut points as index. The data is collected in
        row-order defined by CUT_POINTS -variable. The given bin-values can be
        floats; the value is only used to determine distance along the full-
        length sample.

        Style of file: (index = cut point name, columns = samplenames)

                ctrl_1  ctrl_2  ctrl_3  exp_1   exp_2   exp_3   ...
        R1R2    5.5     6       7       8       8.5     8       ...
        R2R3    25      26      26.5    27      27      27.5    ...
        ...     ...     ...     ...     ...     ...     ...     ...

        !!! Use the exact sample names of the analysis !!!
"""
import numpy as np
import pandas as pd
import pathlib as pl
import shapely.geometry as gm
import shapely.ops as op


ROOT = pl.Path(r"E:\Code_folder\ALLSTATS")
SAVEDIR = pl.Path(r"E:\Code_folder\test_ALLSTAT_split")
CUT_POINTS = ["R1R2", "R2R3", "R3R4", "R4R5"]  # Names of channel folders or index labels in cutpoint file
CHANNELS = ['DAPI', 'GFP', 'SuH', 'Pros']
# Number of bins for whole length of samples. Script gives recommendation for
# numbers of bins for each split region based on this value:
TOTAL_BIN = 62
CONT_END = True

# Cutting of individual samples based on bin numbers:
SAMPLE_BINS = True
# Path to file containing cutting bins for each sample
# (index = names of cut points, columns = samples)
BIN_PATH = pl.Path(r"E:\Code_folder\ALLSTATS\Analysis Data\cutpoints_ALLSTATS.csv")


class VectorLengths:
    """Manage vector-related data."""

    def __init__(self, samples, points):
        # List of samples
        sample_list = [s.stem for s in samples]

        # Variables for holding data:
        self.lengths = pd.DataFrame(columns=sample_list, index=points)
        self.averages = pd.DataFrame(index=points)

    def find_averages(self):
        """Calculate average sub-vector lengths for each sample group."""
        # Add group identifier to the samples
        self.lengths.loc['Group', :] = [s.split('_')[0] for s in self.lengths.columns]

        # Group the data by their sample groups
        grouped = self.lengths.T.groupby(by='Group', axis=0)

        # Get average lengths of sub-vectors for each sample group
        for group, data in grouped:
            means = data.infer_objects().mean()
            means.rename(group, inplace=True)
            self.averages = pd.concat([self.averages, means], axis=1)

        # Rename columns to correspond to the groups
        # self.averages.columns = grouped.groups.keys()

    def save_substrings(self, sample_name, sub_vectors):
        """Store a sample's sub-vector lengths."""
        # Loop sub-vectors
        for i, sub in enumerate(sub_vectors):
            # Store length to data matrix
            self.lengths.loc[:, sample_name].iat[i] = sub.length


def bins_to_dist(cut_points, samplepath, bins, total=TOTAL_BIN):
    """Change given bins to normalized distances along vector."""

    samplename = samplepath.name
    cut_distances = []  # Holds distances

    # Loop through the cut point names
    for point in cut_points:
        cbin = bins.at[point, samplename]  # Get bin number of the cutpoint
        dist = cbin / total  # Get the bins fraction of the total
        cut_distances.append(dist)  # Add distance to list
    # Add final segment if wanted
    if CONT_END:
        cut_distances.append(1.0)
    return cut_distances


def cut_data(data, cut_distances):
    """Cut data at defined distances."""

    idx_cuts = []  # Holds data subsets

    # Cut data at every distance in cut_distances
    for i, point in enumerate(cut_distances):
        # First segment
        if i == 0:
            idx_before = data.loc[(data.NormDist <= point)].index
        # Later segments
        else:
            idx_before = data.loc[(data.NormDist <= point) &
                                  (data.NormDist > cut_distances[i-1])].index

        # Add cut segment to list
        idx_cuts.append(idx_before)
    return idx_cuts


def cut_vector(vector_path, cut_distances):
    """Cut vector at defined distances."""

    sub_vectors = []  # Holds sub-vectors

    # Read the full vector
    vector = read_vector(vector_path)

    # Cut vector at every distance point in cut_distances
    for i, dist in enumerate(cut_distances):
        # First segment
        if i == 0:
            sub_v = op.substring(vector, 0, dist, normalized=True)
        # Later segments
        else:
            sub_v = op.substring(vector, cut_distances[i-1], dist,
                                 normalized=True)

        # Add cut segment to list
        sub_vectors.append(sub_v)
    return sub_vectors


def get_cut_points(CUT_POINTS, samplepath):
    """Get cutpoints based on point projection distances along vector."""

    cut_distances = []  # Distance holder

    # Loop all defined projections
    for point in CUT_POINTS:
        # Read the projection data
        data = pd.read_csv(samplepath.joinpath(f"{point}.csv"))
        # Get the distance from data
        dist = data.NormDist.iat[0]
        # Add distance to cutpoint list
        cut_distances.append(dist)

    # If using data past final cut point, add final vector length to list
    if CONT_END:
        cut_distances.append(1.0)
    return cut_distances


def get_sample_data(samplepath, points, l_data, bins):
    """Get and split a sample's vector and data."""

    sample_name = samplepath.stem  # Name of sample

    # DETERMINE CUTPOINTS IN DATA:
    # Based on projected points
    if not SAMPLE_BINS:
        cut_distances = get_cut_points(CUT_POINTS, samplepath)
    # Or change user-given bins to distances along sample
    else:
        cut_distances = bins_to_dist(CUT_POINTS, samplepath, bins)

    # Find all data channel paths
    file_paths = samplepath.glob("*.csv")

    # Get vector and cut it based on cut distances
    try:
        vector_path = next(samplepath.glob("Vector.*"))
        sub_vectors = cut_vector(vector_path, cut_distances)
    except StopIteration:
        sub_vectors = None

    # Store cut vectors
    l_data.save_substrings(sample_name, sub_vectors)

    # CUTTING AND SAVING OF NEW DATA:
    # Find files to cut
    datafiles = [file for file in file_paths if file.stem in CHANNELS]

    # Loop found files
    for datafile in datafiles:
        # Read the data and make an index cut based on distances
        data = pd.read_csv(datafile)
        idx_cuts = cut_data(data, cut_distances)

        # Loop cut index-sets
        for i, cut in enumerate(idx_cuts):

            # Define and make channel data save-directory, then save
            chan_dir = '{}_{}_Stats'.format(sample_name, datafile.stem)
            data_dir = SAVEDIR.joinpath(points[i], sample_name, chan_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            name = "Position.csv".format()
            data.loc[cut, :].to_csv(data_dir.joinpath(name), index=False)

    # Define and make vector's save-directory, then save
    for ind, cut_point in enumerate(points):
        vector_dir = SAVEDIR.joinpath(cut_point, "Analysis Data", "Samples", sample_name)
        vector_dir.mkdir(parents=True, exist_ok=True)
        save_vector(vector_dir, sub_vectors[ind])


def read_vector(vector_path):
    """Read vector and change to LineString"""

    # Read depending on file type
    if vector_path.name == "Vector.csv":
        vector_df = pd.read_csv(vector_path)
    elif vector_path.name == "Vector.txt":
        vector_df = pd.read_csv(vector_path, sep="\t", header=None)
        vector_df.columns = ["X", "Y"]

    # Reformat data and change to LineString
    Vect = list(zip(vector_df.loc[:, 'X'].astype('float'),
                    vector_df.loc[:, 'Y'].astype('float')))
    vector = gm.LineString(Vect)
    return vector


def save_vector(vector_dir, sub_vector):
    """Transform sub-vector to DF and save"""
    # Transform to FataFrame
    vector_df = pd.DataFrame(np.vstack(sub_vector.xy).T, columns=['X', 'Y'])
    # Save
    vector_df.to_csv(vector_dir.joinpath("Vector.csv"), index=False)


if __name__ == '__main__':
    # Find samples in the dataset
    SAMPLES = [p for p in ROOT.joinpath("Analysis Data", "Samples").iterdir()
               if p.is_dir()]

    # Make a copy of the cutpoint names
    POINTS = CUT_POINTS.copy()

    # Placeholder
    BIN_FILE = None

    # Add final segment of segment to cutpoints if wanted
    if CONT_END:
        POINTS.append("END")

    # Create object to store vector data
    length_data = VectorLengths(SAMPLES, POINTS)

    # If using user-defined cutting points, read file
    if SAMPLE_BINS:
        BIN_FILE = pd.read_csv(BIN_PATH, index_col=0)

    # Loop samples and get their data
    for path in SAMPLES:
        print(path.name)
        get_sample_data(path, POINTS, length_data, bins=BIN_FILE)


    print("Finding averages ...")
    length_data.find_averages()
    print(f"\n- Average lengths:\n {length_data.averages.round(decimals=2)}\n")

    summed_length = length_data.averages.sum(axis=0)
    bin_suggestion = pd.DataFrame(index=length_data.averages.index,
                                  columns=length_data.averages.columns)
    for grp in summed_length.index:
        fraction = length_data.averages.loc[:, grp] / summed_length[grp]
        bins = TOTAL_BIN * fraction
        bin_suggestion.loc[:, grp] = bins
    if len(bin_suggestion.columns) > 1:
        bin_suggestion.loc[:, 'MEAN'] = bin_suggestion.mean(axis=1).values
        bin_suggestion.loc['TOTAL', :] = bin_suggestion.sum().values
    print("- Bin fractions:\n", bin_suggestion.round(decimals=2))
    bin_suggestion.to_csv(SAVEDIR.joinpath('Bin_suggestions.csv'))
    print('\n\nSPLIT DONE')
