# -*- coding: utf-8 -*-
from settings import settings
import pathlib as pl, shutil, pandas as pd, json

class paths:
    def __init__(self, workdir):
        """Creation of output folders."""
        self.outputdir = workdir.joinpath('Analysis Data')
        if self.outputdir.exists() == False:
            pl.Path.mkdir(self.outputdir)
        self.areadir = pl.Path(self.outputdir / 'Areas')
        if self.areadir.exists() == False:
            pl.Path.mkdir(self.areadir)
        self.voldir = pl.Path(self.outputdir / 'Volumes')
        if self.voldir.exists() == False:
            pl.Path.mkdir(self.voldir)
        self.datadir = pl.Path(self.outputdir / 'Data Files')
        if settings.process_samples == True and self.datadir.exists() == True:
            shutil.rmtree(self.datadir)
        self.plotdir = pl.Path(self.outputdir / 'Plots')
        if self.plotdir.exists() == False:
            pl.Path.mkdir(self.plotdir)
        self.samplesdir = pl.Path(self.outputdir / 'Samples')
        if self.samplesdir.exists() == False:
            pl.Path.mkdir(self.samplesdir)
        if settings.statistics == True:
            self.statsdir = pl.Path(self.outputdir / 'Stats')
            if self.statsdir.exists() == False:
                pl.Path.mkdir(self.statsdir)
        if self.datadir.exists() == False:
            pl.Path.mkdir(self.datadir)

    def analysis_info(self, sample, group, channels):
        """For saving information of all analyzed samples."""
        
        with open(str(pl.Path(self.outputdir / 'Sample list.txt')), 'w') as (file):
            file.write(json.dumps((str(store.samplelist)), indent=0))
        with open(str(pl.Path(self.outputdir / 'Sample groups.txt')), 'w') as (file):
            file.write(json.dumps((str(store.samplegroups)), indent=0))
        with open(str(pl.Path(self.outputdir / 'Channels.txt')), 'w') as (file):
            file.write(json.dumps((str(store.channels)), indent=0))

def read_data(filepath, header = 2):
        """For reading csv-data."""
        try:
            data = pd.read_csv(filepath, header=header, index_col=False)
            data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        except FileNotFoundError:
            print("File {} is not found at {}".format(filepath.name, str(filepath.parent)))
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