# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:09:56 2019

@author: artoviit
"""
import pathlib as pl
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

def boxPlot(**kws):
    axes = plt.gca()
    data = kws.pop('data')
    x = kws.pop('x')
    y = kws.pop('y')
    palette = kws.pop('palette')
    sns.boxplot(data=data, x=x, y=y, hue="Sample Group", palette = palette, 
                saturation=0.5, linewidth=0.1, showmeans=False, ax = axes)
    xticks = np.arange(0, data.loc[:,x].unique().size, 5)
    plt.xticks(xticks)
    axes.set_xticklabels(xticks)

workdir = pl.Path(r'\\ad.helsinki.fi\home\a\artoviit\Desktop\test\Analysis Data\Data Files')

data = pd.read_csv(str(workdir.joinpath("Norm_DAPI.csv")), index_col = False)

plotData = pd.DataFrame()
groups = ["Holidic", "starv", "starvgln"]
for grp in groups:
    namer = str(grp+'_')
    namerreg = re.compile(namer, re.I)
    temp = data.loc[:, data.columns.str.contains(namerreg)].T
    temp.loc[:,"Sample Group"] = grp
    if plotData.empty: plotData = temp
    else: plotData = pd.concat([plotData, temp])


colors = sns.color_palette("colorblind", 3)
palette = {}
for i, grp in enumerate(groups):
    palette.update({grp: colors[i]})


plotData = pd.melt(plotData, id_vars = "Sample Group")


g = sns.FacetGrid(plotData, row = "Sample Group", hue = "Sample Group", palette = palette,
                      sharex=True, sharey=True, row_order = groups, height = 5, aspect = 3)

kws = {'data': plotData, 'x': 'variable', 'y': 'value', 'palette': palette}
g = g.map_dataframe(boxPlot, **kws)