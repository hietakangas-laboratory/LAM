# -*- coding: utf-8 -*-
from settings import settings as Sett
import sys
import pathlib as pl, shutil, pandas as pd, inspect
import logger
LAM_logger = logger.get_logger()

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
            # If samples are to be processed and output data directory exists, the 
            # directory will be removed with all files as not to interfere with 
            # analysis.
            if self.datadir.exists() == True and any([Sett.process_samples, 
                                  Sett.process_counts]):
                shutil.rmtree(self.datadir)
            # Create output directories
            pl.Path.mkdir(self.outputdir, exist_ok=True)
            pl.Path.mkdir(self.plotdir, exist_ok=True)
            pl.Path.mkdir(self.samplesdir, exist_ok=True)
            pl.Path.mkdir(self.datadir, exist_ok=True)
            pl.Path.mkdir(self.statsdir, exist_ok=True)
        except:
            logger.log_print(LAM_logger, 'Problem with directory creation.', 'e')
            return
        logger.log_print(LAM_logger, 'Directories successfully created.', 'i')

    def save_AnalysisInfo(self, sample, group, channels):
        """For saving information of all analyzed samples."""
        pd.DataFrame(sample).to_csv(self.outputdir.joinpath('SampleList.csv'), 
                     index=False, header=False)
        pd.DataFrame(group).to_csv(self.outputdir.joinpath('SampleGroups.csv'), 
                     index=False, header=False)
        pd.DataFrame(channels).to_csv(self.outputdir.joinpath('Channels.csv'), 
                     index=False, header=False)
        logger.log_print(LAM_logger, 'All metadata successfully saved.', 'i')
                   
def read_data(filepath, header=Sett.header_row, test=True, index_col=False):
    """For reading csv-data."""
    try:
        data = pd.read_csv(filepath, header=header, index_col=index_col)
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        if test:
            data.loc[:, 'ID']
    except KeyError:
        logger.log_print(LAM_logger, 'Column label test failed: ID not present at {}'\
                             .format(filepath), 'w')
        print('WARNING: read_data() call from {} line {}'.format(
                                inspect.stack()[1][1], inspect.stack()[1][2]))
        print("Key 'ID' not found. Wrong header row?")
        print("If all correct, set test=False\nPath: {}".format(filepath))
    except FileNotFoundError:
        logger.log_print(LAM_logger, 'File not found at {}'.format(filepath), 'e')
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
    if append == False:
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
    if not isinstance(Sett.workdir, pl.Path):
        Sett.workdir = pl.Path(Sett.workdir)
    try:
        if True not in [Sett.process_samples, Sett.process_counts, 
                    Sett.Create_Plots, Sett.process_dists, 
                    Sett.statistics]:
            logger.log_print(LAM_logger, 'All primary settings are False', 'w')
            sys.exit("\nAll primary settings are set to False.\n\nExiting ...")
        else:
            PATHS = paths(Sett.workdir)
            return PATHS
    except SystemExit:
        raise
        
class store:
    samplegroups = []
    channels = []
    samples = []
    totalLength = 0
    center = 0