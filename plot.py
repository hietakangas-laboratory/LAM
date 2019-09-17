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
    def __init__(self, sampleData, savepath, color = 'b'):
        #TODO add group etc arguments
        if isinstance(sampleData, pl.Path):
            self.data = system.read_data(sampleData, header = 0)
            self.name = str(sampleData.name).split('_')[1]
        else:
            self.data = sampleData
            self.name = sampleData.name
        self.savepath = savepath
        self.color = color
        try:
            self.MPbin = store.centerpoint
            self.vmax = sampleData.max()
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
        
    def plot_Data(self, plotfunc, group = None):    
        # TODO implement plotfuncs with self-call?
        # TODO Change plotting into row-wise facetgrid?
        # g = sns.FacetGrid(data, row = "Sample Group", hue = "Sample Group", 
        #                  sharex=True, sharey=True)
        if group is not None:
            namer = str(group+'_')
            plotData = self.Data.loc[:, self.Data.columns.str.contains(namer)]
            plotName = str("{} {}".format(group, self.name))
            # TODO add color picker
        else: 
            plotData = self.Data
            plotName = str(self.name)
        g = sns.FacetGrid(data, row = "Sample Group", hue = "Sample Group", 
                      sharex=True, sharey=True, palette = self.palette)
#        fig, ax = plt.subplots(figsize=(10,5))
        if hasattr(self, vmax):
            ax.ylim(top = self.max)
        ax = plotfunc(plotData, ax) # add color
        ax = self.centerline(self, ax)
        ax.set_title(plotName)
        ax.set_ylabel(str(self.name)) # Change Y LABEL
        ax.set_xlabel("Longitudinal Position")
        filepath = self.savepath.joinpath(plotName+settings.figExt)
        fig.savefig(str(filepath), format=settings.saveformat)
        plt.close()
        
    def boxPlot(Data, axes, color = 'b', **kws):
        xticks = np.arange(0, Data.shape[0], 5)
        sns.boxplot(data=Data.T,  width=0.6, color = color, saturation=0.5, 
                    linewidth=0.1, showmeans=False, ax=axes)
        plt.xticks(xticks)
        axes.set_xticklabels(xticks)
        return axes
    
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