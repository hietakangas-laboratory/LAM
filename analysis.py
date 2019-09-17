# -*- coding: utf-8 -*-
from settings import settings
from plot import plotter
import system
from system import store as store
import pandas as pd
import numpy as np
import pathlib as pl
import seaborn as sns

class Samplegroups(object):
    # TODO create gathering of all samplegroup data
    #_instance = None
    _groups, _chanPaths, _samplePaths, _addData = [], [], [], []
    _grpColors, _chColors = None, None
    
#    def __new__(cls, groups = None, PATHS = None):
#        if not cls._instance:
#            cls._instance = super(Samplegroups, cls).__new__(cls)
#        return cls._instance
    
    def __init__(self, groups, PATHS):
        Samplegroups._groups = groups
        Samplegroups._chanPaths = list(PATHS.datadir.glob("Norm_*"))
        Samplegroups._samplePaths = [p for p in PATHS.samplesdir.iterdir() if p.is_dir()]
        Samplegroups._addData = list(PATHS.datadir.glob("Avg_*"))
        Samplegroups._grpColors = sns.color_palette("colorblind", len(groups))
        Samplegroups._chColors = sns.color_palette("colorblind", len(self._chanPaths))
    
    def read_channel(self, channel):
        chanData = system.read_data(header=0, test=False)
    
class Group(Samplegroups):
    # TODO create gathering of data for one samplegroup
    
    def __init__(self, group):
        self.name = group
        
#    def __getattr__(self, name):
#        try:
#            return getattr(self.parent, name)
#        except AttributeError:
#            raise AttributeError("Group object {} has no attribute {}".format(self.name, name))