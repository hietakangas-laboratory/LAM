# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:07:33 2019

@author: artoviit
"""
from settings import settings
import matplotlib.pyplot as plt
import seaborn as sns

class plotter:
    def __init__(self, sampleData, channel, savepath):
        self.name = sampleData.name
        if isinstance(channel, str):
            self.channel = channel
        else:
            self.channel = channel.name
        self.fullName = str(self.name + "_" + self.channel)
        try: 
            self.bin = sampleData.MPbin
        except: pass
        self.path = savepath
        return
    
    def vector(self, vectordata, X, Y, binaryArray, skeleton):
        try:
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
            name = str('Skeleton_' + self.name + settings.figExt)
            fig.savefig(str(self.path.joinpath(name)), format=settings.saveformat)
        except: pass
        fig, ax = plt.subplots(figsize=(12, 6))
        ax = sns.scatterplot(x=X, y=Y, color='brown')
        ax = plt.plot(*vectordata.xy)
        name = str('Vector_' + self.name + settings.figExt)
        fig.savefig(str(self.path.joinpath(name)), format=settings.saveformat)
        plt.close('all')
            
    def channelCounts(self, samplegroup = False):
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