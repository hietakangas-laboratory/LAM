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

class get_samplegroup:
    # TODO create gathering of sample group data
    def __init__(self, group, channel):
        return