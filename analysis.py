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
    _instance = None
    
    def __new__(cls, groups = None, PATHS = None):
        print(cls.count)
        cls.count += 1
        if not cls._instance:
            cls._instance = super(Samplegroups, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, groups, PATHS): 
        self.groups = groups
        self.chanPaths = list(PATHS.datadir.glob("Norm_*"))
        self.samplePaths = [p for p in PATHS.samplesdir.iterdir() if p.is_dir()]
        self.addData = list(PATHS.datadir.glob("Avg_*"))
        self.grpColors = sns.color_palette("colorblind", len(groups))
        self.chColors = sns.color_palette("colorblind", len(self.chanPaths))
    
#    def read_channel(self, channel):
#        chanData = system.read_data(header=0, test=False)
    
class Group(Samplegroups):
    # TODO create gathering of data for one samplegroup
    def __init__(self, group):
        super().__new__()
        self.name = group
#        self.color = 