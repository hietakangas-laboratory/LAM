# -*- coding: utf-8 -*-
from settings import settings
from plot import plotter
import system
from system import store as store
import pandas as pd
import numpy as np
import warnings
import shapely.geometry as gm
import pathlib as pl
import math
from scipy.ndimage import morphology as mp
from skimage.morphology import skeletonize
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import seaborn as sns

class Samplegroups:
    # TODO create gathering of all samplegroup data
    def __init__(self, groups, channels):
        self.groups = groups
        self.channels = channels
        return
    
class Group:
    # TODO create gathering of data for one samplegroup
    def __init__(self, group, channels):
        super().__init__(group, channels)
        return