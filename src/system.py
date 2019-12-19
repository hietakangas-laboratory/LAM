# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen

"""
# Standard libraries
import inspect
import shutil
# Other packages
import pandas as pd
import pathlib as pl
# LAM modules
import logger as lg
from settings import settings as Sett
try:
    LAM_logger = lg.get_logger(__name__)
except AttributeError:
    print('Cannot get logger')


class paths:
    def __init__(self, workdir):
        """Creation of output folders."""
        try:
            # Create path-variables necessary for the analysis
            self.outputdir = workdir.joinpath('Analysis Data')
            self.datadir = pl.Path(self.outputdir / 'Data Files')
            self.plotdir = pl.Path(self.outputdir / 'Plots')
            self.samplesdir = pl.Path(self.outputdir / 'Samples')
            self.statsdir = pl.Path(self.outputdir / 'Statistics')
            # If samples are to be processed and output data directory exists,
            # the directory will be removed with all files as not to interfere
            # with analysis.
            if self.datadir.exists() and Sett.process_counts:
                files = list(self.datadir.glob('*'))
                msg = "'Analysis Data'-folder will be cleared. Continue? [y/n]"
                if files:
                    flag = 1
                    while flag:
                        ans = input(msg)
                        if ans == "y" or ans == "Y":
                            flag = 0
                            shutil.rmtree(self.datadir)
                        elif ans == "n" or ans == "N":
                            flag = 0
                            print('Analysis terminated')
                            raise KeyboardInterrupt
                        else:
                            print('Command not understood.')
            # Create output directories
            pl.Path.mkdir(self.outputdir, exist_ok=True)
            pl.Path.mkdir(self.plotdir, exist_ok=True)
            pl.Path.mkdir(self.samplesdir, exist_ok=True)
            pl.Path.mkdir(self.datadir, exist_ok=True)
            pl.Path.mkdir(self.statsdir, exist_ok=True)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        lg.logprint(LAM_logger, 'Directories successfully created.', 'i')

    def save_AnalysisInfo(self, sample, group, channels):
        """For saving information of all analyzed samples."""
        pd.DataFrame(sample).to_csv(self.outputdir.joinpath('SampleList.csv'),
                                    index=False, header=False)
        pd.DataFrame(group).to_csv(self.outputdir.joinpath('SampleGroups.csv'),
                                   index=False, header=False)
        pd.DataFrame(channels).to_csv(self.outputdir.joinpath('Channels.csv'),
                                      index=False, header=False)
        lg.logprint(LAM_logger, 'All metadata successfully saved.', 'i')


def read_data(filepath, header=Sett.header_row, test=True, index_col=False):
    """For reading csv-data."""
    try:
        data = pd.read_csv(filepath, header=header, index_col=index_col)
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        if test:
            try:
                data.loc[:, 'ID']
            except KeyError:
                msg = 'Column label test failed: ID not present at {}'\
                                                            .format(filepath)
                lg.logprint(LAM_logger, msg, 'ex')
                print('WARNING: read_data() call from {} line {}'.format(
                                inspect.stack()[1][1], inspect.stack()[1][2]))
                print("Key 'ID' not found. Wrong header row?")
                print("If all correct, set test=False\nPath: {}"
                      .format(filepath))
    except FileNotFoundError:
        lg.logprint(LAM_logger, 'File not found at {}'.format(filepath), 'e')
        print('WARNING: read_data() call from {} line {}'.format(
                                inspect.stack()[1][1], inspect.stack()[1][2]))
        print('File {} not found at {}'.format(filepath.name,
                                               str(filepath.parent)))
        return
    return data


def saveToFile(data, directory, filename, append=True, w_index=False):
    """Takes a Series and saves it to a DataFrame file, or alternatively saves
    a DataFrame to a file. Expects the Series to be of same length as the data
    in the csv"""
    path = directory.joinpath(filename)
    if not append:
        if isinstance(data, pd.DataFrame):
            data.to_csv(str(path), index=w_index)
        else:
            data.to_frame().to_csv(str(path), index=w_index)
    elif path.exists():
        file = pd.read_csv(str(path), index_col=w_index)
        if data.name not in file.columns:
            file = pd.concat([file, data], axis=1)
        else:
            file.loc[:, data.name] = data
        file = file.sort_index(axis=1)
        file.to_csv(str(path), index=w_index)
    else:
        if isinstance(data, pd.DataFrame):
            data.to_csv(str(path), index=w_index)
        else:
            data.to_frame().to_csv(str(path), index=w_index)


def start():
    """Check that everything is OK when starting a run."""
    # If workdir variable isn't pathlib.Path, make it so
    if not isinstance(Sett.workdir, pl.Path):
        Sett.workdir = pl.Path(Sett.workdir)
    # Check that at least one primary setting is True
    if not any([Sett.process_samples, Sett.process_counts,
                Sett.Create_Plots, Sett.process_dists, Sett.statistics]):
        lg.logprint(LAM_logger, 'All primary settings are False', 'e')
        print("\nAll primary settings are set to False.\n\nExiting ...")
        raise SystemExit
    else:  # Otherwise create paths and directories
        PATHS = paths(Sett.workdir)
        store.samples = [p.name for p in PATHS.samplesdir.iterdir() if
                         p.is_dir()]
        return PATHS


class store:
    """Store important variables for the analysis."""
    samplegroups = []  # All samplegroup in analysis
    channels = []  # All channels in analysis
    samples = []  # All samples in analysis
    binNum = len(Sett.projBins)  # Number of used bins
    totalLength = 0  # The length of DataFrame after all samples are anchored
    center = 0  # The index of the anchoring point within the DataFrame
    clusterPaths = []