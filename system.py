# -*- coding: utf-8 -*-
from settings import settings
import pathlib as pl, shutil, pandas as pd
import inspect

class paths:
    def __init__(self, workdir):
        """Creation of output folders."""
        # Create path-variables necessary for the analysis
        self.outputdir = workdir.joinpath('Analysis Data')
        self.areadir = pl.Path(self.outputdir / 'Areas')
        self.voldir = pl.Path(self.outputdir / 'Volumes')
        self.datadir = pl.Path(self.outputdir / 'Data Files')   
        self.plotdir = pl.Path(self.outputdir / 'Plots')
        self.samplesdir = pl.Path(self.outputdir / 'Samples')
        # If samples are to be processed and output data directory exists, the 
        # directory will be removed with all files as not to interfere with analysis.
        if settings.process_samples == True and self.datadir.exists() == True:
            shutil.rmtree(self.datadir)
        # Create output directories
        pl.Path.mkdir(self.outputdir, exist_ok=True)
        pl.Path.mkdir(self.areadir, exist_ok=True)
        pl.Path.mkdir(self.voldir, exist_ok=True)
        pl.Path.mkdir(self.plotdir, exist_ok=True)
        pl.Path.mkdir(self.samplesdir, exist_ok=True)
        pl.Path.mkdir(self.datadir, exist_ok=True)
        if settings.statistics == True:
            pl.Path.mkdir(self.statsdir, exist_ok=True)

    def save_AnalysisInfo(self, sample, group, channels):
        """For saving information of all analyzed samples."""
        pd.DataFrame(sample).to_csv(self.outputdir.joinpath("SampleList.csv"),
                     index=False,header=False)
        pd.DataFrame(group).to_csv(self.outputdir.joinpath("SampleGroups.csv"),
                     index=False,header=False)
        pd.DataFrame(channels).to_csv(self.outputdir.joinpath("Channels.csv"),
                     index=False,header=False)

def read_data(filepath, header = 2, test=True):
        """For reading csv-data."""
        try:
            data = pd.read_csv(filepath, header=header, index_col=False)
            data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
            if test:
                data.loc[:,"ID"]
        except KeyError:
            print("WARNING: read_data() call from {} line {}".format(
                                inspect.stack()[1][1], inspect.stack()[1][2]))
            print("Key not found. Wrong header row? \nPath: {}".format(filepath))
        except FileNotFoundError:
            print("WARNING: read_data() call from {} line {}".format(
                                inspect.stack()[1][1], inspect.stack()[1][2]))
            print("File {} is not found at {}".format(filepath.name, 
                  str(filepath.parent)))
            return
        return data

def saveToFile(data, directory, filename, append = True):
    """Takes a Series and saves it to a file with a DataFrame"""
    path = directory.joinpath(filename)
    if append == False:
        data.to_csv(str(path),index=False)
    elif path.exists():
        file = pd.read_csv(str(path),index_col=False)
        if data.name not in file.columns:
            file = pd.concat([file,data],axis=1)
            file = file.sort_index(axis=1)
            file.to_csv(str(path),index=False)
    else:
        data.to_frame().to_csv(str(path),index=False)

class store:
    samplegroups = []
    channels = []
    samples = []
    totalLength = 0