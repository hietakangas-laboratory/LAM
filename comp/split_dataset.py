# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:14:14 2020

@author: artoviit
"""
import numpy as np
import pandas as pd
import pathlib as pl
import shapely.geometry as gm
import shapely.ops as op
import copy

ROOT = pl.Path(r"P:\h919\hietakangas\Arto\fed_full_data")
SAVEDIR = pl.Path(r"P:\h919\hietakangas\Arto\split_test")
CUT_POINTS = ["R2R3", "R3R4", "R4R5"]
# Number of bins for whole length of samples. Script gives recommendation for
# numbers of bins for each split region based on this value:
TOTAL_BIN = 40
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


def save_vector(output_dir, sub_vector):
    vector_df = pd.DataFrame(np.vstack(sub_vector.xy).T, columns=['X', 'Y'])
    vector_df.to_csv(output_dir.joinpath("Vector.csv"), index=False)


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
        ban = ["Vector", "MP"] + POINTS
        if datafile.stem not in ban:
            data = pd.read_csv(datafile)
            idx_cuts = cut_data(data, cut_distances)
            for i, cut in enumerate(idx_cuts):
                output_dir = SAVEDIR.joinpath(POINTS[i], "Analysis Data",
                                              "Samples", sample_name)
                output_dir.mkdir(parents=True, exist_ok=True)
                name = "{}_{}.csv".format(datafile.stem, POINTS[i])
                data.loc[cut, :].to_csv(output_dir.joinpath(name), index=False)
                save_vector(output_dir, sub_vectors[i])


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


if __name__ == '__main__':
    SAMPLES = [p for p in ROOT.joinpath("Analysis Data", "Samples").iterdir()
               if p.is_dir()]
    POINTS = CUT_POINTS.copy()
    if CONT_END:
        POINTS.append("END")
    length_data = vector_lengths(SAMPLES, POINTS)
    for path in SAMPLES:
        get_sample_data(path, POINTS, length_data)
    length_data.find_averages()
    summed_length = length_data.averages.sum(axis=0)
    bin_suggestion = pd.DataFrame(index=length_data.averages.index,
                                  columns=length_data.averages.columns)
    for grp in summed_length.index:
        fraction = length_data.averages.loc[:, grp] / summed_length[grp]
        bins = TOTAL_BIN * fraction
        bin_suggestion.loc[:, grp] = np.round(bins).astype(int)
    if len(bin_suggestion.columns) > 1:
        bin_suggestion.loc['TOTAL', :] = bin_suggestion.sum().values
        bin_suggestion.loc[:, 'MEAN'] = bin_suggestion.mean(axis=1).values
    print(bin_suggestion)
    bin_suggestion.to_csv(SAVEDIR.joinpath('Bin_suggestions.csv'))
    print('\n\nSPLIT DONE')
