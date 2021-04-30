# -*- coding: utf-8 -*-
"""
For creating vector plots for all samples in a dataset.

Uses channel and vector data found in the 'Samples'-directory (channel files
created during LAM 'Project').

PLOTS CREATED TO THE "ANALYSIS DATA\SAMPLES"-DIRECTORY

USAGE:
    1. Input path to the analysis directory to variable analysis_dir below.
    2. Give name of a channel to variable base_channel, which will be plotted
       alongside the vector, i.e. to show that vector is on top of cells
    3. If anchoring point is also wanted, give name of the directory (default 'MP')
    3. Give header row of the channel data into channel_data_header
    4. Run script

Created on Tue May  5 08:20:28 2020

@author: ArtoVi
"""

import pathlib2 as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Path to analysis directory
analysis_dir = pl.Path(r'E:\Code_folder\All_statistics_P-folder')
# Name channel to plot under vector (gives shape of midgut)
base_channel = 'DAPI'
# Name of anchoring point channel (MP). Set to None if not needed.
mp_name = 'MP'
# Row for datafile column labels
channel_data_header = 2


def main():
    sample_dir = analysis_dir.joinpath('Analysis Data', 'Samples')
    samplepaths = [p for p in sample_dir.iterdir() if p.is_dir()]
    create_vector_plots(analysis_dir, sample_dir, samplepaths, mp_name)


def get_vector_data(files):
    """Collect vector data based file type."""
    files = list(files)
    if len(files) > 1:
        try:
            file = [p for p in files if p.suffix=='.txt'][0]
        except IndexError:
            file = files[0]
    else:
        file = files[0]
    if file.suffix == '.txt':
        data = pd.read_csv(file, sep="\t", header=None)
        data.columns = ["X", "Y"]
    elif file.suffix == '.csv':
        data = pd.read_csv(file, index_col=False)
    return data


def get_ch_data(sample_name, dir_glob, header):
    try:
        dir_path = [p for p in dir_glob if p.is_dir()][0]
        data_path = dir_path.glob('*Position.csv')
        data = pd.read_csv(next(data_path), header=header)
        return data.loc[:, ['Position X', 'Position Y']].assign(sample=sample_name)
    except (IndexError, StopIteration, FileNotFoundError):
        return pd.DataFrame({'Position X': [np.nan], 'Position Y': [np.nan], 'sample': [sample_name]})


def create_vector_plots(workdir, savedir, sample_dirs, mp_dir_name):
    """"Create single plot file that contains vectors of all found samples."""
    full = pd.DataFrame()
    vectors = pd.DataFrame()
    mp_full = pd.DataFrame()

    # Loop all samples:
    for path in sample_dirs:
        sample = path.name
        print(sample)
        data = get_ch_data(sample, workdir.joinpath(sample).glob(f'*{base_channel}_*'), channel_data_header)
        try:
            files = path.glob('Vector.*')
            vector = get_vector_data(files)
            vector = vector.assign(sample=sample)
        except (FileNotFoundError, StopIteration):
            vector = pd.DataFrame({'X': [np.nan], 'Y': [np.nan], 'sample': [sample]})
        full = pd.concat([full, data])
        vectors = pd.concat([vectors, vector])

        # read MP data
        if mp_dir_name is None:
            continue
        mp_data = get_ch_data(sample, workdir.joinpath(sample).glob(f'*{mp_dir_name}_*'), channel_data_header)
        mp_full = pd.concat([mp_full, mp_data])


    grid = sns.FacetGrid(data=full, col='sample', col_wrap=4, sharex=False,
                         sharey=False, height=2, aspect=3.5)
    plt.subplots_adjust(hspace=1)
    samples = pd.unique(full.loc[:, 'sample'])
    for ind, ax in enumerate(grid.axes.flat):
        # Channel data:
        data = full.loc[full.loc[:, 'sample'] == samples[ind], :]
        sns.scatterplot(data=data, x='Position X', y='Position Y', color='xkcd:tan', linewidth=0, ax=ax)
        # Vector:
        vector_data = vectors.loc[vectors.loc[:, 'sample'] == samples[ind], :]
        ax.plot(vector_data.X, vector_data.Y)
        # MP:
        if mp_dir_name is not None and not mp_full.empty:
            mp_data = mp_full.loc[mp_full.loc[:, 'sample'] == samples[ind], :]
            ax.scatter(x=mp_data.at[0, 'Position X'], y=mp_data.at[0, 'Position Y'], c='red', s=36, zorder=4)
        # Set labels and title
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(samples[ind])
    grid.savefig(str(savedir.joinpath('Vectors.png')))
    print(f"\nSave location: {str(savedir.joinpath('Vectors.png'))}")
    plt.close('all')


if __name__== '__main__':
    print('Creating vector plot')
    main()
    print('----------\nDONE')