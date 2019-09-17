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
        
    def plot_Data(self, plotfunc, hue = None, palette = None, **kws):
#        if self.palette == None:
#            total = self.data.loc[:,id_vars].unique().size
#            palette = self.__palette[:total]
        if 'id_str' in kws:
            plotData = self.melt_data(**kws)
            kws.update({'x': 'variable', 'y': 'value', 'data': plotData})
        else: 
            plotData = self.data
        g = sns.FacetGrid(plotData, row = kws.get('row'), hue = kws.get('hue'), 
                          sharex=True, sharey=True, height=5, aspect=3)
        g = g.map_dataframe(plotfunc, self.palette, **kws)
#        g = plotfunc(plotData, g) # add color
#        ax = self.centerline(self, ax)
        plt.suptitle(self.name)
        g.set(xlabel = "Longitudinal Position", ylabel = "Value")
        filepath = self.savepath.joinpath(self.name+settings.figExt)
        g.savefig(str(filepath), format=settings.saveformat)
        plt.close()
        
    def boxPlot(palette, **kws):
        axes = plt.gca()
        data = kws.pop('data')
        sns.boxplot(data=data, x=kws.get('x'), y=kws.get('y'), hue=kws.get('id_str'), 
                    saturation=0.5, linewidth=0.1, showmeans=False, palette = palette, ax=axes)
        xticks = np.arange(0, data.loc[:,kws.get('x')].unique().size, 5)
        plt.xticks(xticks)
        axes.set_xticklabels(xticks)
    
    def distPlot(self, axes, color = 'b', **kws):
        return axes
    
    def linePlot(self, axes, color = 'b', **kws):
        return axes
        
    def centerline(self, axes, **kws):
        __, ytop = axes.ylim()
        ycoords = (0,ytop)
        xcoords = (self.MPbin,self.MPbin)
#        kws = dict(linewidth=settings.lw*3,alpha=.4)
        axes.plot(xcoords,ycoords, 'r--',**kws)
        return axes
    
    def channelCounts(self):
        end = len(settings.projBins)
        plotbins = np.arange(0,end,1)
        xticks = np.arange(0,end,10)
#            
#            
#            for col in self.data.columns:
#                MPpath = self.directory.joinpath(str(settings.R3name+".csv"))
#                MP = pd.read_csv(MPpath,index_col=False)
#                point = MP.loc[tuple([0,col])]
#                figure, ax = plt.subplots(figsize=(5,5))
#                counts = self.data.loc[:,col]
#                sns.barplot(y=counts, x=plotbins, color=self.color,saturation=0.5,ax=ax)
#                x = [point,point]
#                y = [0,self.ylim]
#                sns.lineplot(x=x,y=y, color='black', ax=ax)
#                name = str('Counts_'+self.channel + '_' +col)
#                ax.set_xlabel('Longitudinal position %') # X LABEL
#                ax.set_ylabel(str(col+" cells")) # Y LABEL
#                plt.xticks(xticks, xticks)
#                ax.set_title(name)
#                ax.set_ylim(bottom=0, top=self.ylim)
#                filename = str(name+settings.figExt)
#                filepath = system.samplesdir.joinpath(col,filename)
#                figure.savefig(str(filepath), format=settings.saveformat)
#                plt.close()