# -*- coding: utf-8 -*-
"""
LAM-module for data handling of analysis, e.g. for statistics and plotting.

Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen

"""
# Standard libraries
import re
import warnings
from itertools import product, chain
# Other packages
import numpy as np
import pandas as pd
import pathlib as pl
import seaborn as sns
# import shapely.geometry as gm
from scipy.spatial import distance
# LAM imports
import system as system
import process as process
from settings import store, settings as Sett
from statsMWW import statistics, Total_Stats
from plot import plotting
import logger as lg
try:
    LAM_logger = lg.get_logger(__name__)
except AttributeError:
    print('Cannot get logger')


class Samplegroups:
    """Holds and handles all sample groups, i.e. every sample of analysis."""

    # Initiation of variables shared by all samples.
    _groups, _chanPaths, _samplePaths, _addData = [], [], [], []
    paths = pl.Path('./')
    _grpPalette = {}
    _AllMPs = None
    _length = 0
    _center = None

    def __init__(self, PATHS=None, child=False):
        # Creation of variables related to all samples, that are later passed
        # on to child classes.
        if not child:
            Samplegroups._groups = sorted(store.samplegroups)
            Samplegroups._chanPaths = list(PATHS.datadir.glob('Norm_*'))
            Samplegroups._samplePaths = [p for p in PATHS.samplesdir.iterdir()
                                         if p.is_dir()]
            Samplegroups._addData = list(PATHS.datadir.glob('Avg_*'))
            # Data and other usable directories
            Samplegroups.paths = PATHS
            # Total length of needed data matrix of all anchored samples
            Samplegroups._length = store.totalLength
            # Get MPs of all samples
            MPpath = PATHS.datadir.joinpath('MPs.csv')
            Samplegroups._AllMPs = system.read_data(MPpath, header=0,
                                                    test=False)
            # If anchor point index is defined, find the start index of samples
            if store.center is not None:
                Samplegroups._center = store.center
            # Assign color for each sample group
            groupcolors = sns.xkcd_palette(Sett.palette_colors)
            for i, grp in enumerate(Samplegroups._groups):
                Samplegroups._grpPalette.update({grp: groupcolors[i]})
            lg.logprint(LAM_logger, 'Sample groups established.', 'i')

    def create_plots(self):
        """Handle data for the creation of most plots."""

        # If no plots handled by this method are True, return
        plots = [Sett.Create_Channel_Plots, Sett.Create_AddData_Plots,
                 Sett.Create_Channel_PairPlots, Sett.Create_Heatmaps,
                 Sett.Create_Distribution_Plots, Sett.Create_Cluster_Plots,
                 Sett.Create_ChanVSAdd_Plots, Sett.Create_AddVSAdd_Plots,
                 Sett.plot_width]
        if not any(plots):
            return

        # Conditional function calls to create each of the plots.
        lg.logprint(LAM_logger, 'Begin plotting.', 'i')
        print("\n---Creating plots---")
        # Update addData variable to contain newly created average-files
        self._addData = list(self.paths.datadir.glob('Avg_*'))

        # PLOT SAMPLE GROUP WIDTHS
        if Sett.plot_width:
            lg.logprint(LAM_logger, 'Plotting widths', 'i')
            print('Plotting widths  ...')
            plotting(self).width()
            lg.logprint(LAM_logger, 'width plot done.', 'i')

        # CHANNEL PLOTTING
        if Sett.Create_Channel_Plots:
            lg.logprint(LAM_logger, 'Plotting channels', 'i')
            print('Plotting channels  ...')
            plotting(self).channels()
            lg.logprint(LAM_logger, 'Channel plots done.', 'i')

        # ADDITIONAL DATA PLOTTING
        if Sett.Create_AddData_Plots:
            lg.logprint(LAM_logger, 'Plotting additional data', 'i')
            print('Plotting additional data  ...')
            plotting(self).add_data()
            lg.logprint(LAM_logger, 'Additional data plots done.', 'i')

        # CHANNEL MATRIX PLOTTING
        if Sett.Create_Channel_PairPlots:  # Plot pair plot
            lg.logprint(LAM_logger, 'Plotting channel matrix', 'i')
            print('Plotting channel matrix  ...')
            plotting(self).channel_matrix()
            lg.logprint(LAM_logger, 'Channel matrix done.', 'i')

        # SAMPLE AND SAMPLE GROUP HEATMAPS
        if Sett.Create_Heatmaps:  # Plot channel heatmaps
            lg.logprint(LAM_logger, 'Plotting heatmaps', 'i')
            print('Plotting heatmaps  ...')
            plotting(self).heatmaps()
            lg.logprint(LAM_logger, 'Heatmaps done.', 'i')

        # CHANNEL VS ADDITIONAL BIVARIATE
        if Sett.Create_ChanVSAdd_Plots:
            lg.logprint(LAM_logger, 'Plotting channel VS additional data', 'i')
            print('Plotting channel VS additional data  ...')
            plotting(self).chan_bivariate()
            lg.logprint(LAM_logger, 'Channels VS Add Data done.', 'i')

        # ADDITIONAL VS ADDITIONAL BIVARIATE
        if Sett.Create_AddVSAdd_Plots:  # Plot additional data against self
            lg.logprint(LAM_logger, 'Plotting add. data vs add. data', 'i')
            print('Plotting additional data VS additional data  ...')
            plotting(self).add_bivariate()
            lg.logprint(LAM_logger, 'Add Data VS Add Data done', 'i')

        # CHANNEL AND ADD DISTRIBUTIONS
        if Sett.Create_Distribution_Plots:  # Plot distributions
            lg.logprint(LAM_logger, 'Plotting distributions', 'i')
            print('Plotting distributions')
            plotting(self).distributions()
            lg.logprint(LAM_logger, 'Distributions done', 'i')

        # CLUSTER PLOTS
        if Sett.Create_Cluster_Plots:  # Plot cluster data
            lg.logprint(LAM_logger, 'Plotting clusters', 'i')
            print('Plotting clusters  ...')
            plotting(self).clusters()
            lg.logprint(LAM_logger, 'Clusters done', 'i')

        lg.logprint(LAM_logger, 'Plotting completed', 'i')

    def read_channel(self, path, groups, drop=False, name_sep=1):
        """Read channel data and concatenate sample group info into DF."""
        Data = system.read_data(path, header=0, test=False)
        readData = pd.DataFrame()
        # Loop through given groups and give an identification variable for
        # each sample belonging to the group.
        for grp in groups:
            namerreg = re.compile('^{}_'.format(grp), re.I)
            # Get only the samples that belong to the loop's current group
            temp = Data.loc[:, Data.columns.str.contains(namerreg)].T
            if Sett.Drop_Outliers and drop:  # conditionally drop outliers
                temp = DropOutlier(temp)
            temp['Sample Group'] = grp  # Giving of sample group identification
            if readData.empty:
                readData = temp
            else:
                readData = pd.concat([readData, temp])
        # Finding the name of the data under analysis from its filepath
        name = '_'.join(str(path.stem).split('_')[name_sep:])
        center = self._center  # Getting the bin to which samples are centered
        return readData, name, center

    def Get_Clusters(self):
        """Gather sample data to compute cell clusters."""
        lg.logprint(LAM_logger, 'Finding clusters', 'i')
#        allpaths = [] ???
        for grp in self._groups:  # Get one sample group
            lg.logprint(LAM_logger, '-> clusters for group {}'.format(grp),
                        'i')
            print('\n---Finding clusters for group {}---'.format(grp))
            SampleGroup = Group(grp)
            for path in SampleGroup.groupPaths:  # Get one sample of the group
                Smpl = Sample(path, SampleGroup)
                print('{}  ...'.format(Smpl.name))
                Smpl.Clusters(Sett.Cl_maxDist)  # Find clusters
        lg.logprint(LAM_logger, 'Clusters calculated', 'i')

    def Get_DistanceMean(self):
        """Get sample data and pass for cell-to-cell distance calculation."""
        lg.logprint(LAM_logger, 'Finding cell-to-cell distances', 'i')
        for grp in self._groups:  # Get one sample group
            lg.logprint(LAM_logger, '-> Distances for group {}'.format(grp),
                        'i')
            print('\n---Finding nearest cells for group {}---'.format(grp))
            SampleGroup = Group(grp)
            for path in SampleGroup.groupPaths:  # Get one sample of the group
                Smpl = Sample(path, SampleGroup)
                print('{}  ...'.format(Smpl.name))
                # Find distances between nuclei within the sample
                Smpl.DistanceMean(Sett.maxDist)
        lg.logprint(LAM_logger, 'Distances calculated', 'i')

    def Get_Statistics(self):
        """Handle data for group-wise statistical analysis."""

        if len(self._groups) <= 1:
            print("Statistics require multiple sample groups. Stats passed.")
            lg.logprint(LAM_logger, 'Stats passed. Not enough sample groups',
                        'i')
            return
        lg.logprint(LAM_logger, 'Calculation of statistics', 'i')
        if Sett.Create_Plots and Sett.Create_Statistics_Plots:
            print('\n---Calculating and plotting statistics---')
        else:
            print('\n---Calculating statistics---')

        # VERSUS STATS
        if Sett.stat_versus:
            lg.logprint(LAM_logger, '-> Versus statistics', 'i')
            print('-Versus-')
            # Finding control and other groups
            control = Sett.cntrlGroup
            ctrlName = re.compile(control, re.I)
            others = [g for g in self._groups if not ctrlName.fullmatch(g)]
            # Create all possible combinations of control versus other groups
            grouping = [[control], others]
            pairs = product(*grouping)
            # Loop through all the possible group pairs
            for pair in pairs:
                (__, testgroup) = pair
                # Initiate group-class for both groups
                ctrl = Group(control)
                Grp = Group(testgroup)
                # Print names of groups under statistical analysis
                print("{} Vs. {}  ...".format(ctrl.group, Grp.group))
                # Initiate statistics-class with the two groups
                Stats = statistics(ctrl, Grp)
                # Find stats of cell counts and additional data by looping
                # through each.
                for path in chain(self.paths.datadir.glob('Norm_*'),
                                  self.paths.datadir.glob('Avg_*'),
                                  self.paths.datadir.glob('ClNorm_*')):
                    Stats = Stats.MWW_test(path)
                    if Stats.error:
                        msg = "Missing or faulty data for {}".format(path.name)
                        lg.logprint(LAM_logger, msg, 'e')
                        continue
                    # If plotting set to True, make plots of current stats
                    if Sett.Create_Statistics_Plots and Sett.Create_Plots:
                        plotting(self).stat_versus(Stats, path)
            lg.logprint(LAM_logger, '--> Versus done', 'i')

        # TOTAL STATS
        if Sett.stat_total:
            lg.logprint(LAM_logger, '-> Total statistics', 'i')
            print('-Totals-')
            # Find the data file, initialize class, and count stats
            datapaths = self.paths.datadir.glob('Total*.csv')
            for path in datapaths:
                TCounts = Total_Stats(path, self._groups, self.paths.plotdir,
                                      self.paths.statsdir)
                # If error in data, continue to next totals file
                if TCounts.dataerror:
                    continue
                TCounts.stats()
                for key in TCounts.errorVars.keys():
                    msg = "Value Error between control and {} in".format(key)
                    errVars = ', '.join(TCounts.errorVars.get(key))
                    lg.logprint(LAM_logger, '{} {}'.format(msg, errVars), 'e')
                # If wanted, create plots of the stats
                if Sett.Create_Plots and Sett.Create_Statistics_Plots:
                    plotting(self).stat_totals(TCounts, path)
            lg.logprint(LAM_logger, '--> Totals done', 'i')
        lg.logprint(LAM_logger, 'All statistics done', 'i')

    def Get_Totals(self):
        """Count sample & channel -specific cell totals."""
        def _readAndSum():
            """Read path and sum cell numbers of bins for each sample."""
            ChData, __, _ = self.read_channel(path, self._groups, drop=dropB)
            # Get sum of cells for each sample
            ChSum = ChData.sum(axis=1, skipna=True, numeric_only=True)
            # Get group of each sample
            groups = ChData.loc[:, 'Sample Group']
            # Change the sum data into dataframe and add group identifiers
            ChSum = ChSum.to_frame().assign(group=groups.values)
            ChSum.rename(columns={'group': 'Sample Group'}, inplace=True)
            return ChSum

        lg.logprint(LAM_logger, 'Finding total counts', 'i')
        dropB = Sett.Drop_Outliers  # Find if dropping outliers
        datadir = self.paths.datadir
        All = pd.DataFrame()
        # Loop through files containing cell count data, read, and find sums
        for path in datadir.glob('All_*'):
            ChSum = _readAndSum()
            channel = path.stem.split('_')[1]  # Channel name
            ChSum = ChSum.assign(Variable=channel)
            All = pd.concat([All, ChSum], ignore_index=False, sort=False)
        # Save dataframe containing sums of each channel for each sample
        system.saveToFile(All, datadir, 'Total Counts.csv',
                          append=False, w_index=True)

        # Find totals of additional data
        for channel in [c for c in store.channels if c not in ['MP', 'R45',
                                                               Sett.MPname]]:
            All = pd.DataFrame()
            for path in datadir.glob('Avg_{}_*'.format(channel)):
                ChData, __, _ = self.read_channel(path, self._groups,
                                                  drop=dropB)
                # Assign channel identifier
                add_name = path.stem.split('_')[2:]  # Channel name
                ChData = ChData.assign(Variable='_'.join(add_name))
                All = pd.concat([All, ChData], ignore_index=False, sort=False)
            # Drop samples that have nonvariant data
            All = All[All.iloc[:, :-3].nunique(axis=1, dropna=True) > 1]
            # Save dataframe containing sums of each channel for each sample
            filename = 'Total {} AddData.csv'.format(channel)
            system.saveToFile(All, datadir, filename, append=False,
                              w_index=True)

        # Find totals of data obtained from distance calculations
        All = pd.DataFrame()
        for path in chain(datadir.glob('Clusters-*.csv'),
                          datadir.glob('*Distance Means.csv'),
                          datadir.glob('Sample_widths_norm.csv')):
            if 'Clusters-' in path.name:
                name = "{} Clusters".format(path.stem.split('-')[1])
            elif 'Distance Means' in path.name:
                name = "{} Distances".format(path.name.split('_')[1])
            else:
                name = "Widths"
            ChData, __, _ = self.read_channel(path, self._groups,
                                              drop=dropB)
            # Assign data type identifier
            ChData = ChData.assign(Variable=name)
            All = pd.concat([All, ChData], ignore_index=False, sort=False)
        if not All.empty:
            filename = 'Total Distance Data.csv'
            system.saveToFile(All, datadir, filename, append=False,
                              w_index=True)
        lg.logprint(LAM_logger, 'Total counts done', 'i')


class Group(Samplegroups):
    """For storing sample group-specific data."""

    def __init__(self, group, child=False):
        super().__init__(child=True)  # Inherit from samplegroups-class
        self.group = group  # group
        # For finding group-specific columns etc.
        self.namer = '{}_'.format(group)
        # When first initialized, create variables inherited by samples:
        if not child:
            self.color = self._grpPalette.get(self.group)
            namerreg = re.compile("^{}".format(self.namer), re.I)
            self.groupPaths = [p for p in self._samplePaths if namerreg.search(
                                p.name)]
            self.MPs = self._AllMPs.loc[:, self._AllMPs.columns.str.contains(
                                        self.namer)]


class Sample(Group):
    """For sample-specific data and handling sample-related functionalities."""

    def __init__(self, path, grp):
        # Inherit variables from the sample's group
        super().__init__(grp, child=True)
        # Sample's name, path to its directory, and paths to data it has
        self.name = str(path.stem)
        self.path = path
        self.channelPaths = [p for p in path.iterdir() if p.suffix == '.csv' if
                             p.stem not in ['vector', 'MPs', Sett.MPname]]
        # Sample's group-specific color, and it's anchoring bin.
        self.color = grp.color
        self.MP = grp.MPs.loc[0, self.name]

    def Count_clusters(self, Data, name):
        """Count total clustered cells per bin."""
        # Find bins of the clustered cells to find counts per bin
        idx = Data.loc[:, 'ClusterID'].notna().index
        binnedData = Data.loc[Data.dropna(subset=['ClusterID']).index,
                              'DistBin']
        # Sort values and then get counts
        bins = binnedData.sort_values().to_numpy()
        unique, counts = np.unique(bins, return_counts=True)
        idx = np.arange(0, Sett.projBins)
        # Create series to store the cell count data
        binnedCounts = pd.Series(np.full(len(idx), np.nan), index=idx,
                                 name=self.name)
        binnedCounts.loc[unique] = counts
        filename = 'Clusters-{}.csv'.format(name)
        system.saveToFile(binnedCounts, self.paths.datadir, filename)
        # Relate the counts to context, i.e. anchor them at the MP
        insert, _ = process.relate_data(binnedCounts, self.MP,
                                        self._center, self._length)
        # Save the data
        SCounts = pd.Series(data=insert, name=self.name)
        filename = 'ClNorm_Clusters-{}.csv'.format(name)
        system.saveToFile(SCounts, self.paths.datadir, filename)

    def Clusters(self, dist=10):
        """Handle data for finding clusters of cells."""
        kws = {'Dist': dist}  # Maximum distance for considering clustering
        # Listing of paths of channels on which clusters are to be found
        clustChans = [p for p in self.channelPaths for t in
                      Sett.Cluster_Channels if t.lower() == p.stem.lower()]
        for path in clustChans:  # Loop paths, read file, and find clusters
            try:
                Data = system.read_data(path, header=0)
            except (FileNotFoundError, AttributeError):
                msg = "No file for channel {}".format(path.stem)
                lg.logprint(LAM_logger, "{}: {}".format(self.name, msg), 'w')
                print("-> {}".format(msg))
            # Discard earlier versions of found clusters, if present
            Data = Data.loc[:, ~Data.columns.str.contains('ClusterID')]
            Data.name = path.stem  # The name of the clustering channel
            # Find clusters
            self.find_distances(Data, volIncl=Sett.Cl_inclusion,
                                compare=Sett.Cl_incl_type, clusters=True,
                                **kws)

    def find_distances(self, Data, volIncl=200, compare='smaller',
                       clusters=False, **kws):
        """Calculate cell-to-cell distances or find clusters."""
        def _get_nearby(ind, row, target, maxDist, rmv_self=False):
            """Within an iterator, find all cells near the current cell."""
            # When finding nearest in the same channel, remove the current
            # cell from the frame, otherwise nearest cell would be itself.
            if rmv_self:
                target = target.loc[target.index.difference([ind]), :]
            # Find cells within the accepted limits (Sett.maxDist)
            nID = target[((abs(target.x - row.x) <= maxDist) &
                          (abs(target.y - row.y) <= maxDist))].index
            if nID.size == 0:
                return None
            # Get coordinate data of current cell
            point = np.asarray([row.at['x'], row.at['y'], row.at['z']])
            point = np.reshape(point, (-1, 3))
            # DF for storing nearby cells
            r_df = pd.DataFrame(index=nID, columns=['XYZ', 'Dist', 'ID'])
            # Calculate distances to each nearby cell
            # NOTE: z-distance is also taken into account at this step
            r_df['Dist'] = distance.cdist(point, target.loc[
                nID, ['x', 'y', 'z']].to_numpy(), 'euclidean').ravel()
            # Drop data that is more distant than the max distance
            r_df = r_df[r_df.Dist <= maxDist]
            if r_df.empty:  # If no cells are close enough
                return None
            # Otherwise, insert coordinates and cell ID to DF
            r_df['XYZ'] = target.loc[nID, ['x', 'y', 'z']].apply(tuple,
                                                                 axis=1)
            r_df['ID'] = target.loc[r_df.index, 'ID']
            return r_df

        def _find_clusters():
            """Find cluster 'seeds' and merge to create full clusters."""
            def __merge(Seeds):
                """Merge seeds that share cells."""
                r = sum(Seeds, [])  # List of all cells
                # Create map object containing a set for each cell ID:
                r = map(lambda x: set([x]), set(r))
                # Loop through a set of each seed
                for item in map(set, Seeds):
                    # For each seed, find IDs from the set of cell
                    # IDs and merge them
                    out = [x for x in r if not x & item]  # ID-sets not in seed
                    mSeeds = [x for x in r if x & item]  # found ID-sets
                    # make union of the ID sets that are found
                    mSeeds = set([]).union(*mSeeds)
                    # Reassign r to contain the newly merged ID-sets
                    r = out + [mSeeds]
                yield r

            maxDist = kws.get('Dist')  # max distance to consider clustering
            clusterSeed = {}  # For storing cluster 'seeds'
            for i, row in XYpos.iterrows():  # Iterate over all cells
                # Find nearby cells
                nearby = _get_nearby(i, row, XYpos, maxDist)
                # If nearby cells, make a list of their IDs and add to seeds
                if nearby is not None:
                    if nearby.shape[0] > 1:
                        clusterSeed[i] = nearby.ID.tolist()
            # Make a sorted list of lists of the found cluster seeds
            Cl_lst = [sorted(list(clusterSeed.get(key))) for key in
                      clusterSeed.keys()]
            # Merging of the seeds
            Cl_gen = __merge(Cl_lst)
            # Change the generator into list of lists and drop clusters of size
            # under/over limits
            Clusters = [list(y) for x in Cl_gen for y in x if y and len(y) >=
                        Sett.Cl_min and len(y) <= Sett.Cl_max]
            return Clusters

        def _find_nearest():
            """Iterate passed data to determine nearby cells."""
            maxDist = kws.get('Dist')  # distance used for subsetting target
            # If distances are found to features on another channel:
            if 'targetXY' in locals():
                target = targetXY
                comment = Sett.target_chan
                filename = 'Avg_{} VS {}_Distance Means.csv'.format(Data.name,
                                                                    comment)
                rmv = False
            else:  # If using the same channel:
                target = XYpos
                rmv = True
                comment = Data.name
                filename = 'Avg_{}_Distance Means.csv'.format(Data.name)
            # Creation of DF to store found data (later concatenated to data)
            cols = ['Nearest_XYZ_{}'.format(comment), 'Nearest_Dist_{}'.format(
                    comment), 'Nearest_ID_{}'.format(comment)]
            NewData = pd.DataFrame(columns=cols, index=XYpos.index)
            # If not finding nearest on other channel, search distance can be
            # limited if current cell has already been found to be near another
            # -> we know there's a cell at least at that distance
            if 'targetXY' not in locals():
                found_ids = {}  # Stores the IDs already found
                # Iterate over each cell (row) in the data
                for i, row in XYpos.iterrows():
                    find_dist = maxDist
                    # Search if cell ID already used
                    if row.at['ID'] in found_ids.keys():
                        find_dist = found_ids.get(row.at['ID'])  # Limit dist
                    # Find nearby cells
                    nearby = _get_nearby(i, row, target, find_dist,
                                         rmv_self=rmv)
                    # If some are found:
                    if nearby is not None:
                        min_idx = nearby.Dist.idxmin()  # Find nearest of cells
                        row2 = nearby.loc[min_idx]  # Get data of cell
                        # Update ID to the 'founds'
                        found_ids.update({row2.at['ID']: row2.at['Dist']})
                        # Insert cell data to the storage DF
                        NewData.loc[i, cols] = row2.to_list()
            else:  # If cells are found on another channel:
                # Iterate each cell and find nearest at the user-defined dist
                for i, row in XYpos.iterrows():
                    nearby = _get_nearby(i, row, target, maxDist, rmv_self=rmv)
                    if nearby is not None:
                        NewData.loc[i, cols] = nearby.loc[nearby.Dist.idxmin()
                                                          ].to_list()
            # Concatenate the obtained data with the read data.
            NewData = pd.concat([Data, NewData], axis=1)
            # Get bin and distance to nearest cell for each cell, calculate
            # average distance within each bin.
            binnedData = NewData.loc[:, 'DistBin']
            distances = NewData.loc[:, cols[1]].astype('float64')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                means = [np.nanmean(distances[binnedData.values == k]) for k in
                         np.arange(0, Sett.projBins)]
            return NewData, means, filename

        if volIncl > 0:  # Subsetting of data based on cell volume
            dInd = subset_data(Data, compare, volIncl)
            if 'tData' in kws.keys():  # Obtain target channel if used
                tData = kws.pop('tData')
                tData.name = Data.name
                tInd = subset_data(tData, compare, volIncl)
        elif 'tData' in kws.keys():
            tData = kws.pop('tData')
            tInd = tData.index
            dInd = Data.index
        else:
            dInd = Data.index
        # Accessing the data for the analysis via the indexes taken before.
        # Cells for which the nearest cells will be found:
        XYpos = Data.loc[dInd, ['Position X', 'Position Y', 'Position Z',
                                'ID', 'DistBin']]
        renames = {'Position X': 'x', 'Position Y': 'y', 'Position Z': 'z'}
        XYpos.rename(columns=renames, inplace=True)  # rename for dot notation
        if 'tInd' in locals():  # Get data from target channel, if used
            targetXY = tData.loc[tInd, ['Position X', 'Position Y',
                                        'Position Z', 'ID']]
            targetXY.rename(columns=renames, inplace=True)
        if not clusters:  # Finding nearest distances
            NewData, Means, filename = _find_nearest()
            SMeans = pd.Series(Means, name=self.name)
            insert, _ = process.relate_data(SMeans, self.MP, self._center,
                                            self._length)
            IMeans = pd.Series(data=insert, name=self.name)
            system.saveToFile(IMeans, self.paths.datadir, filename)
        else:  # Finding clusters
            Clusters = _find_clusters()
            # Create dataframe for storing the obtained data
            clustData = pd.DataFrame(index=Data.index, columns=['ID',
                                                                'ClusterID'])
            clustData = clustData.assign(ID=Data.ID)  # Copy ID column
            # Give name from a continuous range to each of the found clusters
            # and add it to cell-specific data (for each belonging cell).
            if Clusters:
                for i, vals in enumerate(Clusters):
                    vals = [int(v) for v in vals]
                    clustData.loc[clustData.ID.isin(vals), 'ClusterID'] = i
            else:
                print(f"-> No clusters found.")
                clustData.loc[:, 'ClusterID'] = np.nan
            # Merge obtained data with the original data
            NewData = Data.merge(clustData, how='outer', copy=False, on=['ID'])
            self.Count_clusters(NewData, Data.name)
        # Overwrite original sample data with the data containing new columns
        OW_name = '{}.csv'.format(Data.name)
        system.saveToFile(NewData, self.path, OW_name, append=False)

    def DistanceMean(self, dist=25):
        """Prepare and handle data for cell-to-cell distances."""
        kws = {'Dist': dist}  # Maximum distance used to find cells
        # List paths of channels where distances are to be found
        distChans = [p for p in self.channelPaths for t in
                     Sett.Distance_Channels if t.lower() == p.stem.lower()]

        if Sett.use_target:  # If distances are found against other channel:
            target = Sett.target_chan  # Get the name of the target channel
            try:  # Find target's data file, read, and update data to keywords
                file = '{}.csv'.format(target)
                tNamer = re.compile(file, re.I)
                targetPath = [p for p in self.channelPaths if
                              tNamer.fullmatch(str(p.name))]
                tData = system.read_data(targetPath[0], header=0)
                kws.update({'tData': tData})
            except (FileNotFoundError, IndexError):
                msg = "No file for channel {}".format(target)
                lg.logprint(LAM_logger, "{}: {}".format(self.name, msg), 'w')
                print("-> {}".format(msg))
                return

        # Loop through the channels, read, and find distances
        for path in distChans:
            try:
                Data = system.read_data(path, header=0)
            except FileNotFoundError:
                msg = "No file for channel {}".format(path.stem)
                lg.logprint(LAM_logger, "{}: {}".format(self.name, msg), 'w')
                print("-> {}".format(msg))
                return
            # Discard earlier versions of calculated distances, if present
            Data = Data.loc[:, ~Data.columns.str.startswith('Nearest_')]
            # Find distances
            Data.name = path.stem
            self.find_distances(Data, volIncl=Sett.inclusion,
                                compare=Sett.incl_type, **kws)


def DropOutlier(Data):
    """Drop outliers from a dataframe."""
    with warnings.catch_warnings():  # Ignore warnings regarding empty bins
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean = np.nanmean(Data.values)
        std = np.nanstd(Data.values)
        dropval = Sett.dropSTD * std
        if isinstance(Data, pd.DataFrame):
            Data = Data.applymap(lambda x, dropval=dropval: x if
                                 np.abs(x - mean) <= dropval else np.nan)
        elif isinstance(Data, pd.Series):
            Data = Data.apply(lambda x, dropval=dropval: x if np.abs(x - mean)
                              <= dropval else np.nan)
    return Data


def subset_data(Data, compare, volIncl):
    """Get indexes of cells based on values in a column."""
    if not isinstance(Data, pd.DataFrame):
        lg.logprint(LAM_logger, 'Wrong data type for subset_data()', 'e')
        C = 'Wrong datatype for find_distance, Has to be pandas DataFrame.'
        print(C)
        return None
    ErrorM = "Column {} not found for {}".format(Sett.incl_col, Data.name)
    match_str = re.compile(Sett.incl_col, re.I)
    cols = Data.columns.str.match(match_str)
    if compare.lower() == 'greater':
        try:  # Get only cells that are of greater volume
            subInd = Data.loc[(Data.loc[:, cols].values >= volIncl), :].index
        except KeyError:
            print(ErrorM)
    else:
        try:  # Get only cells that are of lesser volume
            subInd = Data.loc[(Data.loc[:, cols].values <= volIncl), :].index
        except KeyError:
            print(ErrorM)
    return subInd


def test_control():
    """Assert that control group exists, and if not, handle it."""
    # If control group is not found:
    if Sett.cntrlGroup in store.samplegroups:
        return
    lg.logprint(LAM_logger, 'Set control group not found', 'c')
    # Test if entry is due to capitalization error:
    namer = re.compile(r"{}$".format(re.escape(Sett.cntrlGroup)), re.I)
    for group in store.samplegroups:
        if re.match(namer, group):  # If different capitalization:
            msg = "Control group-setting is case-sensitive!"
            print("WARNING: {}".format(msg))
            # Change control to found group
            Sett.cntrlGroup = group
            msg = "Control group has been changed to"
            print("{} '{}'\n".format(msg, group))
            lg.logprint(LAM_logger, '-> Changed to {}'.format(group), 'i')
            return
    # If control not found at all:
    msg = "control group NOT found in sample groups!"
    print("\nWARNING: {}\n".format(msg))
    flag = 1
    # Print groups and demand input for control:
    while flag:
        print('Found groups:')
        for i, grp in enumerate(store.samplegroups):
            print('{}: {}'.format(i, grp))
        msg = "Select the number of control group: "
        print('\a')
        ans = system.ask_user(msg, dlgtype='integer')
        if ans is None:
            raise KeyboardInterrupt
        if 0 <= ans <= len(store.samplegroups):
            # Change control based on input
            Sett.cntrlGroup = store.samplegroups[ans]
            print("Control group set as '{}'.\n".format(Sett.cntrlGroup))
            flag = 0
        else:
            print('Command not understood.')
    msg = "-> Changed to group '{}' by user".format(
        Sett.cntrlGroup)
    lg.logprint(LAM_logger, msg, 'i')


def Get_Widths(samplesdir, datadir):
    msg = "Necessary files for width approximation not be found for "
    for path in samplesdir.iterdir():
        files = [p for p in path.iterdir() if p.is_file()]
        vreg = re.compile('^vector\.', re.I)
        dreg = re.compile(f'^{Sett.vectChannel}\.csv', re.I)
        try:
            vect_path = [p for p in files if vreg.match(p.name)]
            data_path = [p for p in files if dreg.match(p.name)]
            vector_data = system.read_data(vect_path[0], header=0, test=False)
            data = system.read_data(data_path[0], header=0)
        except (StopIteration, IndexError):
            name = path.name
            full_msg = msg + name
            print(f"WARNING: {full_msg}")
            if 'vector_data' not in locals():
                print("-> Could not read vector data.")
                continue
            if 'data' not in locals():
                print("Could not read channel data")
                print("Make sure channel is set right (vector channel)\n")
                continue
            lg.logprint(LAM_logger, full_msg, 'w')
        process.DefineWidths(data, vector_data, path, datadir)
