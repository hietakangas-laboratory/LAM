# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:07:33 2019

@author: artoviit
"""
from settings import settings
import warnings
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=FutureWarning)
    import pandas as pd

class plotter:
    def __init__(self, plotData, savepath, center=0, title=None, 
                 palette=None, color='b'):
        sns.set_style(settings.seaborn_style)
        sns.set_context(settings.seaborn_context)
        self.data = plotData
        self.title = title
        self.savepath = savepath
        self.palette = palette
        self.color = color
        self.ext = ".{}".format(settings.saveformat)
        self.format = settings.saveformat
        if center != 0:
            self.MPbin = center
        else:
            self.MPbin = 0

    def vector(self, samplename, vectordata, X, Y, binaryArray=None, skeleton=None):
        if skeleton is not None and settings.SkeletonVector:
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
            name = str('Skeleton_' + samplename + self.ext)
            fig.savefig(str(self.savepath.joinpath(name)), format=self.format)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax = sns.scatterplot(x=X, y=Y, color='brown')
        ax = plt.plot(*vectordata.xy)
        name = str('Vector_' + samplename + self.ext)
        fig.savefig(str(self.savepath.parent.joinpath(name)), format=self.format)
        plt.close('all')

    def plot_Data(self, plotfunc, savepath, palette=None, **kws):
        def __melt_data(Data, **kws):
            if 'var_str' in kws.keys(): varname = kws.get('var_str')
            else: varname = 'variable'
            if 'value_str' in kws.keys(): valname = kws.get('value_str')
            else: valname = 'value'
            plotData = pd.melt(self.data, id_vars=kws.get('id_str'),
                               value_name=valname,var_name=varname)
            return plotData, varname, valname

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
                
        def __marker(value, colors):
            if value <= 0.001:
                pStr = "*\n*\n*"
                color = colors[3]
            elif value <= 0.01:
                pStr = "*\n*"
                color = colors[2]
            elif value <= 0.05:
                pStr = "*"
                color = colors[1]
            else:
                pStr = " "
                color = colors[0]
            return pStr, color
        
        def __stats():
            stats = kws.pop('Stats')
            __, ytop = plt.ylim()
            LScolors = sns.color_palette('Reds',n_colors=4)
            GRcolors = sns.color_palette('Blues',n_colors=4)
            yaxis = [ytop, ytop]
            yheight = ytop*0.8
            for index, row in stats.iterrows():
                if row[0:2].any() != False and row[3] == True:
                    xaxis = [index-0.5, index+0.5]
                    pStr, color = __marker(row[2], LScolors)
                    if settings.fill:
                        plt.fill_between(xaxis,yaxis, color=color, alpha=0.2)
                    if settings.stars:
                        plt.text(index, yheight, pStr)
                if row[4:6].any() != False and row[7] == True:
                    xaxis = [index-0.5, index+0.5]
                    pStr, color= __marker(row[6], GRcolors)
                    if settings.fill:
                        plt.fill_between(xaxis,yaxis, color=color, alpha=0.2)
                    if settings.stars:
                        plt.text(index, yheight, pStr)
            return g
        
        def __add():
            if 'centerline' in kws.keys(): __centerline()
            if 'xlen' in kws.keys(): __set_xtick()
            if 'ylabel' in kws.keys():
                g.set(ylabel=kws.get('ylabel'))
            if 'xlabel' in kws.keys():
                plt.xlabel(kws.get('xlabel'), labelpad=20)
            return g
        #---------#
        if 'id_str' in kws:
            plotData, varname, valname = __melt_data(self.data, **kws)
            kws.update({'xlabel':varname,  'ylabel':valname, 'data':plotData})
        else:
            plotData = self.data
            kws.update({'data': plotData})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            if plotfunc.__name__ == 'jointPlot': # If jointplot:
                # Seaborn unfortunately doesn't support multi-axis jointplots,
                # consequently these are created as individual files.
                key = plotData.iat[0, 0]
                g = sns.jointplot(data=plotData, x=plotData.loc[:, kws.get('x')], 
                          y=plotData.loc[:, kws.get('y')], kind='kde',
                          color=palette.get(key), joint_kws={'shade_lowest': False})
            elif plotfunc.__name__ == 'catPlot':
                g = self.catPlot(self.palette, **kws)
                __stats()
                __add()
            elif plotfunc.__name__ == 'pairPlot':
                g = self.pairPlot(**kws)
            else:
                g = sns.FacetGrid(plotData, row=kws.get('row'), hue=kws.get('hue'), 
                          sharex=True, sharey=True, gridspec_kws={'hspace': 0.3},
                          height=kws.get('height'),aspect=kws.get('aspect'),
                          legend_out=True, dropna=False)
                g = g.map_dataframe(plotfunc, self.palette, **kws).add_legend()
                for ax in g.axes.flat:
                    ax.xaxis.set_tick_params(labelbottom=True)
                __add()
        # Giving a title and then saving the plot
        plt.suptitle(self.title, weight='bold', y=1)
        filepath = savepath.joinpath(self.title + self.ext)
        g.savefig(str(filepath), format=self.format)
        plt.close('all')

    def boxPlot(palette, **kws):
        axes = plt.gca()
        data = kws.pop('data')
        sns.boxplot(data=data, x=kws.get('xlabel'), y=kws.get('ylabel'), 
                    hue=kws.get('id_str'), saturation=0.5, linewidth=0.2, 
                    showmeans=False, notch=False, palette=palette,
                    fliersize=kws.get('flierS'), ax=axes)
        return axes
    
    def pairPlot(self, **kws):
        data = self.data.sort_values(by="Sample Group").drop(
                            'Longitudinal Position', axis=1).replace(np.nan, 0)
        grpOrder = data["Sample Group"].unique().tolist()
        colors = [self.palette.get(k) for k in grpOrder]
        edgeC = []
        for color in colors:
            edgeC.append(tuple([0.7 * v for v in color]))
        pkws = {'x_ci': None, 'order': 4, 'truncate': True, 'x_jitter': 0.49, 
                'y_jitter': 0.49, 
                'scatter_kws': {'linewidth': 0.2, 's': 10, 'alpha':0.3, 
                                'edgecolors': edgeC},
                'line_kws': {'alpha':0.7, 'linewidth': 1.5}}
        dkws = {'linewidth': 2}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            g = sns.pairplot(data=data, hue=kws.get('hue'), kind=kws.get('kind'), 
                         diag_kind=kws.get('diag_kind'), palette=self.palette,
                         plot_kws=pkws, diag_kws=dkws)
        for lh in g._legend.legendHandles: 
            lh.set_alpha(1)
            lh._sizes = [30]
        filepath = self.savepath.joinpath("{}.jpg".format(self.title))
        g.savefig(str(filepath), format='jpg', optimize=True)
            
        return g

    def distPlot(palette, **kws):
        axes = plt.gca()
        return axes

    def linePlot(palette, **kws):
        axes = plt.gca()
        data = kws.pop('data')
        err_kws = {'alpha': 0.4}
        sns.lineplot(data=data, x=kws.get('xlabel'), y=kws.get('ylabel'), hue=kws.get('hue'), 
                     alpha=0.5, dashes=False, err_style='band', ci='sd', 
                     palette=palette, ax=axes, err_kws=err_kws)
        return axes

    def jointPlot(palette, **kws):
        axes = plt.gca()
        data = kws.pop('data')
        key = data.iat[(0, 0)]
        sns.jointplot(data=data, x=data.loc[:, kws.get('X')], y=data.loc[:,
                      kws.get('Y')],kind='kde',color=palette.get(key), 
                        ax=axes, joint_kws={'shade_lowest': False})
    
    def catPlot(self, palette, **kws):        
        data = kws.pop('data')
        col = kws.get('ylabel')
        plotData = data.dropna(subset=[col])
#        Sdata = pd.to_numeric(data[col])   
#        plotdata = plotData.apply(pd.to_numeric, **{'errors':'ignore'})
        flierprops = kws.pop('fliersize')
        g = sns.catplot(data=plotData, x=kws.get('xlabel'), y=kws.get('ylabel'), 
                    hue="Sample Group", kind="box", palette=palette, 
                    linewidth=0.15, height=kws.get('height'),
                    aspect=kws.get('aspect'), **flierprops)
        return g
