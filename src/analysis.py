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
from sklearn.neighbors import KDTree

# LAM imports
import src.logger as lg
import src.process as process
import src.system as system
from src.plot import plotting
from src.settings import store, settings as Sett
from src.statsMWW import statistics, TotalStats

LAM_logger = None


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
        if child:
            return

        # Creation of variables related to all samples, that are later passed
        # on to child classes.
        Samplegroups._groups = sorted(store.samplegroups)
        Samplegroups._chanPaths = list(PATHS.datadir.glob('Norm_*'))
        Samplegroups._samplePaths = [p for p in PATHS.samplesdir.iterdir() if p.is_dir()]
        Samplegroups._addData = list(PATHS.datadir.glob('Avg_*'))

        # Data and other usable directories
        Samplegroups.paths = PATHS

        # Total length of needed data matrix of all anchored samples
        Samplegroups._length = store.totalLength

        # Get MPs of all samples
        MPpath = PATHS.datadir.joinpath('MPs.csv')
        Samplegroups._AllMPs = system.read_data(MPpath, header=0, test=False)

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
        plots = [Sett.Create_Channel_Plots, Sett.Create_AddData_Plots, Sett.Create_Channel_PairPlots,
                 Sett.Create_Heatmaps, Sett.Create_Distribution_Plots, Sett.Create_Cluster_Plots,
                 Sett.Create_ChanVSAdd_Plots, Sett.Create_AddVSAdd_Plots, Sett.plot_width]
        if not any(plots):
            return

        # Conditional function cfull_dfs to create each of the plots.
        lg.logprint(LAM_logger, 'Begin plotting.', 'i')
        print("\n---Creating plots---")
        # Update addData variable to contain newly created average-files
        self._addData = list(self.paths.datadir.glob('Avg_*'))

        # PLOT SAMPLE GROUP WIDTHS
        if Sett.plot_width:
            lg.logprint(LAM_logger, 'Plotting widths', 'i')
            print('  Plotting widths  ...')
            plotting(self).width()
            lg.logprint(LAM_logger, 'width plot done.', 'i')

        # CHANNEL PLOTTING
        if Sett.Create_Channel_Plots:
            lg.logprint(LAM_logger, 'Plotting channels', 'i')
            print('  Plotting channels  ...')
            plotting(self).channels()
            lg.logprint(LAM_logger, 'Channel plots done.', 'i')

        # ADDITIONAL DATA PLOTTING
        if Sett.Create_AddData_Plots:
            lg.logprint(LAM_logger, 'Plotting additional data', 'i')
            print('  Plotting additional data  ...')
            plotting(self).add_data()
            lg.logprint(LAM_logger, 'Additional data plots done.', 'i')

        # CHANNEL MATRIX PLOTTING
        if Sett.Create_Channel_PairPlots:  # Plot pair plot
            lg.logprint(LAM_logger, 'Plotting channel matrix', 'i')
            print('  Plotting channel matrix  ...')
            plotting(self).channel_matrix()
            lg.logprint(LAM_logger, 'Channel matrix done.', 'i')

        # SAMPLE AND SAMPLE GROUP HEATMAPS
        if Sett.Create_Heatmaps:  # Plot channel heatmaps
            lg.logprint(LAM_logger, 'Plotting heatmaps', 'i')
            print('  Plotting heatmaps  ...')
            plotting(self).heatmaps()
            lg.logprint(LAM_logger, 'Heatmaps done.', 'i')

        # CHANNEL VS ADDITIONAL BIVARIATE
        if Sett.Create_ChanVSAdd_Plots:
            lg.logprint(LAM_logger, 'Plotting channel VS additional data', 'i')
            print('  Plotting channel VS additional data  ...')
            plotting(self).chan_bivariate()
            lg.logprint(LAM_logger, 'Channels VS Add Data done.', 'i')

        # ADDITIONAL VS ADDITIONAL BIVARIATE
        if Sett.Create_AddVSAdd_Plots:  # Plot additional data against self
            lg.logprint(LAM_logger, 'Plotting add. data vs add. data', 'i')
            print('  Plotting additional data VS additional data  ...')
            plotting(self).add_bivariate()
            lg.logprint(LAM_logger, 'Add Data VS Add Data done', 'i')

        # CHANNEL AND ADD DISTRIBUTIONS
        if Sett.Create_Distribution_Plots:  # Plot distributions
            lg.logprint(LAM_logger, 'Plotting distributions', 'i')
            print('  Plotting distributions')
            plotting(self).distributions()
            lg.logprint(LAM_logger, 'Distributions done', 'i')

        # CLUSTER PLOTS
        if Sett.Create_Cluster_Plots:  # Plot cluster data
            lg.logprint(LAM_logger, 'Plotting clusters', 'i')
            print('  Plotting clusters  ...')
            plotting(self).clusters()
            lg.logprint(LAM_logger, 'Clusters done', 'i')

        lg.logprint(LAM_logger, 'Plotting completed', 'i')

    def read_channel(self, path, groups, drop=False, name_sep=1):
        """Read channel data and concatenate sample group info into DF."""
        data = system.read_data(path, header=0, test=False)
        read_data = pd.DataFrame()

        # Loop through given groups and give an identification variable for
        # each sample belonging to the group.
        for grp in groups:
            namerreg = re.compile('^{}_'.format(grp), re.I)
            # Get only the samples that belong to the loop's current group
            temp = data.loc[:, data.columns.str.contains(namerreg)].T
            if Sett.Drop_Outliers and drop:  # conditionfull_dfy drop outliers
                temp = DropOutlier(temp)
            temp['Sample Group'] = grp  # Giving of sample group identification
            if read_data.empty:
                read_data = temp
            else:
                read_data = pd.concat([read_data, temp])

        # Finding the name of the data under analysis from its filepath
        name = '_'.join(str(path.stem).split('_')[name_sep:])
        center = self._center  # Getting the bin to which samples are centered
        return read_data, name, center

    def Get_Clusters(self):
        """Gather sample data to compute clusters of cells."""
        print('\n---Finding clusters---')
        lg.logprint(LAM_logger, 'Finding clusters', 'i')

        for grp in self._groups:  # Get one sample group
            lg.logprint(LAM_logger, '-> group {}'.format(grp), 'i')
            print('  {}  ...'.format(grp))
            samplegroup = Group(grp)

            for path in samplegroup.groupPaths:  # Get one sample of the group
                test_sample = Sample(path, samplegroup)
                test_sample.Clusters(Sett.cl_max_dist)  # Find clusters

        lg.logprint(LAM_logger, 'Clusters calculated', 'i')

    def Get_DistanceMean(self):
        """Get sample data and pass for cell-to-cell distance calculation."""
        print('\n---Feature-to-feature distances---')
        lg.logprint(LAM_logger, 'Finding feature-to-feature distances', 'i')

        for grp in self._groups:  # Get one sample group
            lg.logprint(LAM_logger, '-> group {}'.format(grp), 'i')
            print('  {}  ...'.format(grp))
            samplegroup = Group(grp)

            for path in samplegroup.groupPaths:  # Get one sample of the group
                test_sample = Sample(path, samplegroup)
                # Find distances between features within the sample
                test_sample.DistanceMean(Sett.max_dist)

        lg.logprint(LAM_logger, 'Distances calculated', 'i')

    def Get_Statistics(self):
        """Handle data for group-wise statistical analysis."""

        if len(self._groups) <= 1:  # Test whether enough groups for stats
            print("Statistics require multiple sample groups. Stats passed.")
            lg.logprint(LAM_logger, 'Stats passed. Too few groups', 'i')
            return

        # Print info
        lg.logprint(LAM_logger, 'Calculation of statistics', 'i')
        if Sett.Create_Plots and Sett.Create_Statistics_Plots:
            print('\n---Calculating and plotting statistics---')
        else:
            print('\n---Calculating statistics---')

        # VERSUS STATS
        if Sett.stat_versus:
            lg.logprint(LAM_logger, '-> Versus statistics', 'i')
            print('  -Versus-')

            # Finding control and other groups
            control = Sett.cntrlGroup
            ctrl_name = re.compile(control, re.I)
            others = [g for g in self._groups if not ctrl_name.fullmatch(g)]

            # Create all possible combinations of control vs other groups
            grouping = [[control], others]
            pairs = product(*grouping)

            # Loop through all the possible group pairs
            for pair in pairs:
                (__, testgroup) = pair

                # Initiate group-class for both groups
                ctrl = Group(control)
                test_grp = Group(testgroup)
                # Print names of groups under statistical analysis
                print("    {} Vs. {}  ...".format(ctrl.group, test_grp.group))

                # Initiate statistics-class with the two groups
                stats = statistics(ctrl, test_grp)

                # Find stats of cell counts and additional data by looping
                # through each.
                for path in chain(self.paths.datadir.glob('Norm_*'), self.paths.datadir.glob('Avg_*'),
                                  self.paths.datadir.glob('ClNorm_*'), self.paths.datadir.glob('Sample_widths_n*')):
                    stats = stats.MWW_test(path)

                    if stats.error:
                        msg = "Missing or faulty data for {}".format(path.name)
                        lg.logprint(LAM_logger, msg, 'e')
                        continue

                    # If plotting set to True, make plots of current stats
                    if Sett.Create_Statistics_Plots and Sett.Create_Plots:
                        plotting(self).stat_versus(stats, path)
            lg.logprint(LAM_logger, '--> Versus done', 'i')

        # TOTAL STATS
        if Sett.stat_total:
            lg.logprint(LAM_logger, '-> Total statistics', 'i')
            print('  -Totals-')

            # Find the data file, initialize class, and count stats
            datapaths = self.paths.datadir.glob('Total*.csv')
            for path in datapaths:
                total_counts = TotalStats(path, self._groups, self.paths.plotdir, self.paths.statsdir)

                # If error in data, continue to next totals file
                if total_counts.dataerror:
                    continue

                # Calculate stats
                total_counts.stats()

                for key, vals in total_counts.error_vars.items():
                    evars = ', '.join(vals)
                    msg = f"Value Error between control and {key} in {evars}"
                    lg.logprint(LAM_logger, '{} {}'.format(msg, evars), 'e')

                # If wanted, create plots of the stats
                if Sett.Create_Plots and Sett.Create_Statistics_Plots:
                    plotting(self).stat_totals(total_counts, path)

            lg.logprint(LAM_logger, '--> Totals done', 'i')
        lg.logprint(LAM_logger, 'All statistics done', 'i')

    def get_totals(self):
        """Count sample & channel -specific cell totals."""

        def _read_and_sum():
            """Read path and sum cell numbers of bins for each sample."""
            chan_data, __, _ = self.read_channel(path, self._groups, drop=drpb)
            # Get sum of cells for each sample
            ch_sum = chan_data.sum(axis=1, skipna=True, numeric_only=True)
            # Get group of each sample
            groups = chan_data.loc[:, 'Sample Group']
            # Change the sum data into dataframe and add group identifiers
            ch_sum = ch_sum.to_frame().assign(group=groups.values)
            ch_sum.rename(columns={'group': 'Sample Group'}, inplace=True)
            return ch_sum

        lg.logprint(LAM_logger, 'Finding total counts', 'i')
        drpb = Sett.Drop_Outliers  # Find if dropping outliers
        datadir = self.paths.datadir
        full_df = pd.DataFrame()

        # Loop through files containing cell count data, read, and find sums
        for path in datadir.glob('All*'):
            ch_sum = _read_and_sum()
            channel = path.stem.split('_')[1]  # Channel name
            ch_sum = ch_sum.assign(Variable=channel)
            full_df = pd.concat([full_df, ch_sum], ignore_index=False, sort=False)

        # Save dataframe containing sums of each channel for each sample
        system.saveToFile(full_df, datadir, 'Total Counts.csv', append=False, w_index=True)

        # Find totals of additional data
        for channel in [c for c in store.channels if c not in ['MP', 'R45', Sett.MPname]]:
            full_df = pd.DataFrame()

            for path in datadir.glob('Avg_{}_*'.format(channel)):
                chan_data, __, _ = self.read_channel(path, self._groups, drop=drpb)
                # Assign channel identifier
                add_name = path.stem.split('_')[2:]  # Channel name
                chan_data = chan_data.assign(Variable='_'.join(add_name))

                # Concatenate new data to full set
                full_df = pd.concat([full_df, chan_data], ignore_index=False, sort=False)

            if full_df.empty:
                continue

            # Adjust column order so that identifiers are first
            ordered = ['Sample Group', 'Variable']
            cols = ordered + (full_df.columns.drop(ordered).tolist())
            full_df = full_df[cols]

            # Drop samples that have invariant data
            full_df = full_df[full_df.iloc[:, :-3].nunique(axis=1, dropna=True) > 1]

            # Save dataframe containing sums of each channel for each sample
            filename = 'Total {} AddData.csv'.format(channel)
            system.saveToFile(full_df, datadir, filename, append=False, w_index=True)

        # Find totals of data obtained from distance calculations
        full_df = pd.DataFrame()
        for path in chain(datadir.glob('Clusters-*.csv'), datadir.glob('*Distance Means.csv'),
                          datadir.glob('Sample_widths_norm.csv')):
            if 'Clusters-' in path.name:
                name = "{} Clusters".format(path.stem.split('-')[1])
            elif 'Distance Means' in path.name:
                name = "{} Distances".format(path.name.split('_')[1])
            else:
                name = "Widths"
            chan_data, __, _ = self.read_channel(path, self._groups, drop=drpb)

            # Assign data type identifier
            chan_data = chan_data.assign(Variable=name)
            full_df = pd.concat([full_df, chan_data], ignore_index=False, sort=False)

        if not full_df.empty:  # If data obtained
            # Adjust column order so that identifiers are first
            ordered = ['Sample Group', 'Variable']
            cols = ordered + (full_df.columns.drop(ordered).tolist())
            full_df = full_df[cols]

            # Save DF
            filename = 'Total Distance Data.csv'
            system.saveToFile(full_df, datadir, filename, append=False, w_index=True)

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
            self.groupPaths = [p for p in self._samplePaths if namerreg.search(p.name)]
            inds = self._AllMPs.columns.str.contains(self.namer)
            self.MPs = self._AllMPs.loc[:, inds]


class Sample(Group):
    """For sample-specific data and handling sample-related functionalities."""

    def __init__(self, path, grp):
        # Inherit variables from the sample's group
        super().__init__(grp, child=True)
        # Sample's name, path to its directory, and paths to data it has
        self.name = str(path.stem)
        self.path = path
        self.channelPaths = [p for p in path.iterdir() if p.suffix == '.csv'
                             if p.stem not in ['vector', 'MPs', Sett.MPname]]
        # Sample's group-specific color, and it's anchoring bin.
        self.color = grp.color
        self.MP = grp.MPs.loc[0, self.name]

    def count_clusters(self, data, name):
        """Count total clustered cells per bin."""

        # Find bins of the clustered cells to find counts per bin
        idx = data.loc[:, 'ClusterID'].notna().index
        binned_data = data.loc[data.dropna(subset=['ClusterID']).index, 'DistBin']

        # Sort values and then get counts
        bins = binned_data.sort_values().to_numpy()
        unique, counts = np.unique(bins, return_counts=True)
        idx = np.arange(0, Sett.projBins)

        # Create series to store the cell count data
        binned_counts = pd.Series(np.full(len(idx), 0), index=idx, name=self.name)
        binned_counts.loc[unique] = counts
        filename = 'Clusters-{}.csv'.format(name)
        system.saveToFile(binned_counts, self.paths.datadir, filename)

        # Relate the counts to context, i.e. anchor them at the MP
        insert, _ = process.relate_data(binned_counts, self.MP, self._center, self._length)

        # Save the data
        counts_series = pd.Series(data=insert, name=self.name)
        filename = 'ClNorm_Clusters-{}.csv'.format(name)
        system.saveToFile(counts_series, self.paths.datadir, filename)

    def Clusters(self, dist=10):
        """Handle data for finding clusters of cells."""
        kws = {'Dist': dist}  # Maximum distance for considering clustering

        # Listing of paths of channels on which clusters are to be found
        cluster_chans = [p for p in self.channelPaths for t in Sett.Cluster_Channels if t.lower() == p.stem.lower()]
        for path in cluster_chans:  # Loop paths, read file, and find clusters
            try:
                data = system.read_data(path, header=0)
            except (FileNotFoundError, AttributeError):
                msg = "No file for channel {}".format(path.stem)
                lg.logprint(LAM_logger, "{}: {}".format(self.name, msg), 'w')
                print("-> {}".format(msg))

            # Discard earlier versions of found clusters, if present
            data = data.loc[:, ~data.columns.str.contains('ClusterID')]
            data.name = path.stem  # The name of the clustering channel

            # Find clusters
            self.find_distances(data, vol_incl=Sett.cl_inclusion, compare=Sett.cl_incl_type, clusters=True, **kws)

    def find_distances(self, data, vol_incl=200, compare='smfull_dfer', clusters=False, **kws):
        """Calculate cell-to-cell distances or find clusters."""

        def _find_clusters():
            """Find cluster 'seeds' and merge to create full clusters."""

            def __merge(seeds):
                """Merge seeds that share cells."""
                cells = sum(seeds, [])  # List of all cells

                # Create map object containing a set for each cell ID:
                cells = map(lambda x: set([x]), set(cells))

                # Loop through a set of each seed
                for item in map(set, seeds):
                    # For each seed, find IDs from the set of cell
                    # IDs and merge them

                    # ID-sets not in seed:
                    out = [x for x in cells if not x & item]
                    # found ID-sets:
                    m_seeds = [x for x in cells if x & item]

                    # make union of the ID sets that are found
                    m_seeds = set([]).union(*m_seeds)

                    # Reassign cells to contain the newly merged ID-sets
                    cells = out + [m_seeds]
                yield cells

            max_dist = kws.get('Dist')  # max distance to consider clustering
            treedata = xy_pos[['x', 'y', 'z']]

            # Create K-D tree and query for nearest
            tree = KDTree(treedata)
            seed_ids = tree.query_radius(treedata, r=max_dist)

            # Merging of the seeds
            seed_lst = [xy_pos.iloc[a, :].ID.tolist() for a in seed_ids]
            cl_gen = __merge(seed_lst)

            # Change the generator into list of lists and drop clusters of size
            # under/over limits
            all_cl = [list(y) for x in cl_gen for y in x if y and Sett.cl_min <= len(y) <= Sett.cl_max]
            return all_cl

        def _find_nearest():
            """Iterate passed data to determine nearby cells."""

            max_dist = kws.get('Dist')  # distance used for subsetting target

            # If distances are found to features on another channel:
            if 'target_xy' in locals():
                target = target_xy
                comment = Sett.target_chan
                filename = f'Avg_{data.name} VS {comment}_Distance Means.csv'
            else:  # If using the same channel:
                target = xy_pos
                comment = data.name
                filename = f'Avg_{data.name}_Distance Means.csv'

            # Creation of DF to store found data (later concatenated to data)
            cols = [f'Nearest_Dist_{comment}', f'Nearest_ID_{comment}']
            new_data = pd.DataFrame(index=xy_pos.index)

            # KD tree
            treedata = target[['x', 'y', 'z']]
            tree = KDTree(treedata)
            dist, ind = tree.query(xy_pos[['x', 'y', 'z']], k=2)

            col_dict = {cols[0]: dist[:, 1], cols[1]: target.iloc[ind[:, 1]].ID.values}
            new_data = new_data.assign(**col_dict)

            # Concatenate the obtained data with the read data.
            new_data = pd.concat([data, new_data], axis=1)

            # limit data based on max_dist
            new_data[cols] = new_data[cols].where((new_data[cols[0]] <= max_dist))

            # Get bin and distance to nearest cell for each cell, calculate
            # average distance within each bin.
            binned_data = new_data.loc[:, 'DistBin']
            distances = new_data.loc[:, cols[0]].astype('float64')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                means = [np.nanmean(distances[binned_data.values == k]) for k in np.arange(0, Sett.projBins)]
            return new_data, means, filename

        if vol_incl > 0:  # Subsetting of data based on cell volume
            data_ind = subset_data(data, compare, vol_incl, self.name)
            if 'test_data' in kws.keys():  # Obtain target channel if used
                test_data = kws.pop('test_data')
                test_data.name = data.name
                test_ind = subset_data(test_data, compare, vol_incl, self.name)
        elif 'test_data' in kws.keys():
            test_data = kws.pop('test_data')
            test_ind = test_data.index
            data_ind = data.index
        else:
            data_ind = data.index

        # Accessing the data for the analysis via the indexes taken before.
        # Cells for which the nearest cells will be found:
        xy_pos = data.loc[data_ind, ['Position X', 'Position Y', 'Position Z', 'ID', 'DistBin']]
        renames = {'Position X': 'x', 'Position Y': 'y', 'Position Z': 'z'}
        xy_pos.rename(columns=renames, inplace=True)  # rename for dot notation

        if 'test_ind' in locals():  # Get data from target channel, if used
            target_xy = test_data.loc[test_ind, ['Position X', 'Position Y', 'Position Z', 'ID']]
            target_xy.rename(columns=renames, inplace=True)

        if not clusters:  # Finding nearest distances
            new_data, means, filename = _find_nearest()
            means_series = pd.Series(means, name=self.name)
            insert, _ = process.relate_data(means_series, self.MP, self._center, self._length)
            means_insert = pd.Series(data=insert, name=self.name)
            system.saveToFile(means_insert, self.paths.datadir, filename)

        else:  # Finding clusters

            # DBSCAN TEST
            # from sklearn.cluster import DBSCAN
            # model = DBSCAN(eps=Sett.cl_max_dist, min_samples=Sett.cl_min)
            # clusters = model.fit(xy_pos.loc[:, ['x', 'y', 'z']].values)
            # cl_labels = pd.Series(clusters.labels_).replace(-1, np.nan)
            # DBSCAN END TEST

            all_cl = _find_clusters()
            # Create dataframe for storing the obtained data
            cl_data = pd.DataFrame(index=data.index, columns=['ID', 'ClusterID'])
            cl_data = cl_data.assign(ID=data.ID)  # Copy ID column

            # DBSCAN
            # cl_data.loc[cl_data.ID == xy_pos.ID, 'ClusterID'] = cl_labels
            # DBSCAN END

            # Give name from a continuous range to each of the found clusters
            # and add it to cell-specific data (for each belonging cell).
            if all_cl:
                for i, vals in enumerate(all_cl):
                    vals = [int(v) for v in vals]
                    cl_data.loc[cl_data.ID.isin(vals), 'ClusterID'] = i + 1
            else:
                print(f"-> No clusters found for {self.name}.")
                cl_data.loc[:, 'ClusterID'] = np.nan
            # Merge obtained data with the original data
            new_data = data.merge(cl_data, how='outer', copy=False, on=['ID'])
            self.count_clusters(new_data, data.name)

        # Overwrite original sample data with the data containing new columns
        write_name = '{}.csv'.format(data.name)
        system.saveToFile(new_data, self.path, write_name, append=False)

    def DistanceMean(self, dist=25):
        """Prepare and handle data for cell-to-cell distances."""
        kws = {'Dist': dist}  # Maximum distance used to find cells

        # List paths of channels where distances are to be found
        dist_chans = [p for p in self.channelPaths for t in Sett.distance_channels if t.lower() == p.stem.lower()]

        if Sett.use_target:  # If distances are found against other channel:
            target = Sett.target_chan  # Get the name of the target channel
            try:  # Find target's data file, read, and update data to keywords
                file = '{}.csv'.format(target)
                test_namer = re.compile(file, re.I)
                target_path = [p for p in self.channelPaths if test_namer.fullmatch(str(p.name))]
                test_data = system.read_data(target_path[0], header=0)
                kws.update({'test_data': test_data})
            except (FileNotFoundError, IndexError):
                msg = "No file for channel {}".format(target)
                lg.logprint(LAM_logger, "{}: {}".format(self.name, msg), 'w')
                print("-> {}".format(msg))
                return

        # Loop through the channels, read, and find distances
        for path in dist_chans:
            try:
                data = system.read_data(path, header=0)
            except FileNotFoundError:
                msg = "No file for channel {}".format(path.stem)
                lg.logprint(LAM_logger, "{}: {}".format(self.name, msg), 'w')
                print("-> {}".format(msg))
                return
            # Discard earlier versions of calculated distances, if present
            data = data.loc[:, ~data.columns.str.startswith('Nearest_')]
            # Find distances
            data.name = path.stem
            self.find_distances(data, vol_incl=Sett.inclusion, compare=Sett.incl_type, **kws)


def DropOutlier(data):
    """Drop outliers from a dataframe."""
    with warnings.catch_warnings():  # Ignore warnings regarding empty bins
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean = np.nanmean(data.values)
        std = np.nanstd(data.values)
        dropval = Sett.dropSTD * std
        if isinstance(data, pd.DataFrame):
            data = data.applymap(lambda x, dropval=dropval: x if np.abs(x - mean) <= dropval else np.nan)
        elif isinstance(data, pd.Series):
            data = data.apply(lambda x, dropval=dropval: x if np.abs(x - mean) <= dropval else np.nan)
    return data


def subset_data(data, compare, vol_incl, sample):
    """Get indexes of cells based on values in a column."""

    if not isinstance(data, pd.DataFrame):
        lg.logprint(LAM_logger, 'Wrong data type for subset_data()', 'e')
        msg = 'Wrong datatype for find_distance, Has to be pandas DataFrame.'
        print(msg)
        return None

    # Search for the filtering column:
    match_str = re.compile(Sett.incl_col, re.I)
    cols = data.columns.str.match(match_str)

    # If no columns or multiple found:
    if not cols.any():
        e_msg = f"Column '{Sett.incl_col}' not found for {sample} {data.name}."
        print(f"ERROR: {e_msg}\n")
        lg.logprint(LAM_logger, e_msg, 'e')
    elif sum(cols) > 1:
        id_str = f"{sample} {data.name}"
        msg = f"Multiple columns with '{Sett.incl_col}' found for " + id_str
        print(f"WARNING: {msg}. Give specific name for filtering column.\n")

    # Find indices of data to retain:
    if compare.lower() == 'greater':  # Get only cells that are greater value
        sub_ind = data.loc[(data.loc[:, cols].values >= vol_incl), :].index
    else:  # Get only cells that are of lesser value
        sub_ind = data.loc[(data.loc[:, cols].values <= vol_incl), :].index
    return sub_ind


def test_control():
    """Assert that control group exists, and if not, handle it."""
    # If control group is not found:
    if Sett.cntrlGroup in store.samplegroups:
        return True
    lg.logprint(LAM_logger, 'Set control group not found', 'c')

    # Test if entry is due to capitalization error:
    namer = re.compile(r"{}$".format(re.escape(Sett.cntrlGroup)), re.I)
    for group in store.samplegroups:
        if re.match(namer, group):  # If different capitalization:
            msg = "Control group-setting is case-sensitive!"
            print(f"WARNING: {msg}")

            # Change control to found group
            Sett.cntrlGroup = group
            msg = "Control group has been changed to"
            print("{} '{}'\n".format(msg, group))
            lg.logprint(LAM_logger, f"-> Changed to {group}", 'i')
            return True

    # If control not found at all:
    msg = "Control group NOT found in sample groups!"
    print("\nWARNING: {}\n".format(msg))
    if Sett.force_dialog:
        lg.logprint(LAM_logger, msg, 'e')
        Sett.statistics = False
        return False
    ask_control()
    return True


def ask_control():
    """Ask new control group if one not found."""
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
            print(f"Control group set as '{Sett.cntrlGroup}'.\n")
            flag = 0
        else:
            print('Command not understood.')

    msg = f"-> Changed to group '{Sett.cntrlGroup}' by user"
    lg.logprint(LAM_logger, msg, 'i')


def Get_Widths(samplesdir, datadir):
    """Find widths of samples along their vectors."""
    msg = "Necessary files for width approximation not found for "

    for path in [p for p in samplesdir.iterdir() if p.is_dir()]:
        # Find necessary data files:
        files = [p for p in path.iterdir() if p.is_file()]

        # Search terms
        vreg = re.compile('^vector.', re.I)  # vector
        dreg = re.compile(f'^{Sett.vectChannel}.csv', re.I)  # channel data

        try:  # Match terms to found paths
            vect_path = [p for p in files if vreg.match(p.name)]
            data_path = [p for p in files if dreg.match(p.name)]
            # Read found paths
            vector_data = system.read_data(vect_path[0], header=0, test=False)
            data = system.read_data(data_path[0], header=0)

        # Error handling
        except (StopIteration, IndexError):
            name = path.name
            full_msg = msg + name
            print(f"WARNING: {full_msg}")
            if 'vector_data' not in locals():  # if vector not found
                print("-> Could not read vector data.")
                continue
            if 'data' not in locals():  # if channel data not found
                print("Could not read channel data")
                print("Make sure channel is set right (vector channel)\n")
                continue
            lg.logprint(LAM_logger, full_msg, 'w')

        # Compute widths
        process.DefineWidths(data, vector_data, path, datadir)
