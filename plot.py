# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:07:33 2019

@author: artoviit
"""
from settings import settings
import system
from system import store
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib as pl
import warnings

class plotter:
    __palette = settings.grpColors
    
    def __init__(self, plotData, savepath, palette = None, color = 'b'):
        #TODO add group etc arguments
#        if isinstance(sampleData, pl.Path):
#            self.data = system.read_data(sampleData, header = 0)
#            self.name = str(sampleData.name).split('_')[1]
#        else:
        self.data = plotData
        self.name = plotData.name
        self.savepath = savepath
        self.palette = palette
        self.color = color
        try:
            self.MPbin = store.centerpoint
            self.vmax = plotData.max()
#            self.palette = 
        except: pass
            # self.palette = 
        # TODO create dictionary palette from samplegroups
        #   e.g. palette ={"A":"C0","B":"C1","C":"C2", "Total":"k"}        
        return
    
    def vector(self, samplename, vectordata, X, Y, binaryArray = None, skeleton = None):
        if skeleton is not None: 
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6),
                                     sharex=True, sharey=True)
            ax = axes.ravel()
            ax[0].imshow(binaryArray, cmap=plt.cm.gray)
            ax[0].axis('off')
            ax[0].set_title('modified', fontsize=16)        
            ax[1].imshow(skeleton, cmap=plt.cm.gray)
            ax[1].axis('off')
            ax[1].set_title('skeleton', fontsize=16)        
            fig.tight_layout()
            name = str('Skeleton_' + samplename + settings.figExt)
            fig.savefig(str(self.savepath.joinpath(name)), format=settings.saveformat)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax = sns.scatterplot(x=X, y=Y, color='brown')
        ax = plt.plot(*vectordata.xy)
        name = str('Vector_' + samplename + settings.figExt)
        fig.savefig(str(self.savepath.joinpath(name)), format=settings.saveformat)
        plt.close('all')
    
    def melt_data(self, **kws):
        plotData = pd.melt(self.data, id_vars = kws.get('id_str'), value_vars = 
                           kws.get('value_str'), var_name = kws.get('var_str'))
        return plotData
        
    def plot_Data(self, plotfunc, palette = None, *args, **kws):
#        if self.palette == None:
#            total = self.data.loc[:,id_vars].unique().size
#            palette = self.__palette[:total]
        if 'id_str' in kws:
            plotData = self.melt_data(**kws)
            kws.update({'x': 'variable', 'y': 'value', 'data': plotData})
        else: 
            plotData = self.data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            g = sns.FacetGrid(plotData, row = kws.get('row'), hue = kws.get('hue'), 
                              sharex=True, sharey=True, gridspec_kws={'hspace': 0.2},
                              height=5, aspect=3, legend_out=True)
            g = (g.map_dataframe(plotfunc, self.palette, *args, **kws).add_legend())
#            plt.legend(loc='upper right', markerscale = 2, fancybox = True, 
#                       shadow = True, fontsize = 'large')
        plt.suptitle(self.name, weight='bold', size = 24)
        g.set(xlabel = kws.get('xlabel'), ylabel = kws.get('ylabel'))
        filepath = self.savepath.joinpath(self.name+settings.figExt)
        g.savefig(str(filepath), format=settings.saveformat)
        plt.close()
        
    def boxPlot(palette, *args, **kws):
        axes = plt.gca()
        data = kws.pop('data')
        sns.boxplot(data=data, x=kws.get('x'), y=kws.get('y'), hue=kws.get('id_str'), 
                    saturation=0.5, linewidth=0.1, showmeans=False, palette=palette, 
                    ax=axes)
        if 'centerline' in kws.keys(): plotter.centerline(axes, kws.get('centerline'))
        xticks = np.arange(0, data.loc[:,kws.get('x')].unique().size, 5)
        plt.xticks(xticks)
        axes.set_xticklabels(xticks)
    
    def distPlot(palette, *args, **kws):
        axes = plt.gca()
        return axes
    
    def linePlot(palette, *args, **kws):
        axes = plt.gca()
        data = kws.pop('data')
        err_kws = {'alpha': 0.4}
        sns.lineplot(data=data, x=kws.get('x'), y=kws.get('y'), hue=kws.get('id_str'), 
                     alpha=0.5, dashes=False, err_style='band', ci='sd', palette=palette, 
                     ax=axes, err_kws = err_kws)
        if 'centerline' in kws.keys(): plotter.centerline(axes, kws.get('centerline'))
        xticks = np.arange(0, data.loc[:,kws.get('x')].unique().size, 5)
        plt.xticks(xticks)
        axes.set_xticklabels(xticks)
        return axes
    
    def jointPlot(palette, *args, **kws):
        axes = plt.gca()
        data = kws.pop('data')
        sns.jointplot(data=data, x=kws.get('x'), y=kws.get('y'), hue=kws.get('id_str'), 
                    saturation=0.5, kind = 'kde', palette=palette, 
                    ax=axes)
        ticks = np.arange(0, data.loc[:,kws.get('x')].unique().size, 5)
        plt.xticks(ticks)
        axes.set_yticklabels(ticks)
        plt.yticks(ticks)
        axes.set_yticklabels(ticks)
        
    def centerline(axes, MPbin, **kws):
        __, ytop = axes.get_ylim()
        ycoords = (0,ytop)
        xcoords = (MPbin,MPbin)
        axes.plot(xcoords,ycoords, 'r--',**kws)
        return axes