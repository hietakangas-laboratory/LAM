# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:07:33 2019

@author: artoviit
"""
from settings import settings
import analysis
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, warnings

class plotter:

    def __init__(self, plotData, savepath, title=None, palette=None, color='b'):
        sns.set_style(settings.seaborn_style)
        sns.set_context(settings.seaborn_context)
        self.data = plotData
        self.name = plotData.name
        self.savepath = savepath
        self.palette = palette
        self.color = color
        if title is not None:
            self.title = title
        else:
            self.title = self.name
        try:
            self.MPbin = plotData.center
            self.vmax = plotData.max()
        except: pass

    def vector(self, samplename, vectordata, X, Y, binaryArray=None, skeleton=None):
        if skeleton is not None:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharex=True,
              sharey=True)
            ax = axes.ravel()
            ax[0].imshow(binaryArray, cmap=(plt.cm.gray))
            ax[0].axis('off')
            ax[0].set_title('modified', fontsize=16)
            ax[1].imshow(skeleton, cmap=(plt.cm.gray))
            ax[1].axis('off')
            ax[1].set_title('skeleton', fontsize=16)
            fig.tight_layout()
            name = str('Skeleton_' + samplename + settings.figExt)
            fig.savefig((str(self.savepath.joinpath(name))), format=(settings.saveformat))
        fig, ax = plt.subplots(figsize=(12, 6))
        ax = sns.scatterplot(x=X, y=Y, color='brown')
        ax = (plt.plot)(*vectordata.xy)
        name = str('Vector_' + samplename + settings.figExt)
        fig.savefig((str(self.savepath.joinpath(name))), format=(settings.saveformat))
        plt.close('all')

    def plot_Data(self, plotfunc, savepath, palette=None, **kws):

        def __melt_data(Data, **kws):
            if 'var_str' in kws.keys(): varname = kws.get('var_str')
            else: varname = 'variable'
            plotData = pd.melt((self.data), id_vars=(kws.get('id_str')),
                               value_vars=(kws.get('value_str')),var_name=varname)
            return plotData, varname

        def __set_xtick():
            length = kws.get('xlen')
            xtick = np.arange(0, length, 5)
            plt.xticks(xtick, xtick)

        def __centerline():
            MPbin = kws.get('centerline')
            __, ytop = plt.ylim()
            for ax in g.axes.flat:
                ycoords = (0, ytop)
                xcoords = (MPbin, MPbin)
                ax.plot(xcoords, ycoords, 'r--')
        #---------#
        if 'id_str' in kws:
            plotData, varname = __melt_data((self.data), **kws)
            kws.update({'x':varname,  'y':'value',  'data':plotData})
        else:
            plotData = self.data
            kws.update({'data': plotData})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            if plotfunc.__name__ == 'jointPlot': # If jointplot:
                # Seaborn unfortunately doesn't support multi-axis jointplots,
                # consequently these are created as individual files.
                key = plotData.iat[(0, 0)]
                g = sns.jointplot(data=plotData, x=(plotData.loc[:, kws.get('x')]), 
                          y=(plotData.loc[:, kws.get('y')]), kind='kde',
                          color=(palette.get(key)), joint_kws={'shade_lowest': False})
            else: # Creation of plot grids containing each sample group.
                g = sns.FacetGrid(plotData, row=(kws.get('row')), hue=(kws.get('hue')), 
                          sharex=True, sharey=True, gridspec_kws={'hspace': 0.3},
                          height=(kws.get('height')),aspect=(kws.get('aspect')),
                          row_order=(analysis.Samplegroups._groups), 
                          legend_out=True,dropna=False)
                g = (g.map_dataframe)(plotfunc, (self.palette), **kws).add_legend()
                if 'centerline' in kws.keys(): __centerline()
                if 'xlen' in kws.keys(): __set_xtick()
                g.set(ylabel=(kws.get('ylabel')))
                plt.xlabel((kws.get('x')), labelpad=20)
        # Giving a title and then saving the plot
        plt.suptitle((self.title), weight='bold', size=20)
        filepath = savepath.joinpath(self.title + settings.figExt)
        g.savefig((str(filepath)), format=(settings.saveformat))
        plt.close('all')

    def boxPlot(palette, **kws):
        axes = plt.gca()
        data = kws.pop('data')
        sns.boxplot(data=data, x=(kws.get('x')), y=(kws.get('y')), hue=(kws.get('id_str')), 
                    saturation=0.5, linewidth=0.1, showmeans=False, palette=palette,
                    ax=axes)
        return axes

    def distPlot(palette, **kws):
        axes = plt.gca()
        return axes

    def linePlot(palette, **kws):
        axes = plt.gca()
        data = kws.pop('data')
        err_kws = {'alpha': 0.4}
        sns.lineplot(data=data, x=(kws.get('x')), y=(kws.get('y')), hue=(kws.get('hue')), 
                     alpha=0.5, dashes=False, err_style='band', ci='sd', 
                     palette=palette, ax=axes, err_kws=err_kws)
        return axes

    def jointPlot(palette, **kws):
        axes = plt.gca()
        data = kws.pop('data')
        key = data.iat[(0, 0)]
        sns.jointplot(data=data, x=(data.loc[:, kws.get('X')]), y=(data.loc[:,
                      kws.get('Y')]),kind='kde',color=(palette.get(key)), 
                        ax=axes, joint_kws={'shade_lowest': False})
