# -*- coding: utf-8 -*-
"""
LAM-module for controlling system paths and reading / writing.

Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen

"""
# Standard libraries
import inspect
import re
import shutil
from tkinter import simpledialog as sd
import warnings

# Other packages
import numpy as np
import pandas as pd
import pathlib as pl
import shapely.geometry as gm

# LAM modules
import src.logger as lg
import src.plot as plot
from src.settings import Store, Settings as Sett

LAM_logger = None


class Paths:
    """Handle required system paths."""

    def __init__(self, workdir: pl.Path, only_vectors: bool = False):
        """Creation of output folders."""
        # Create path-variables necessary for the analysis
        self.outputdir = workdir.joinpath('Analysis Data')
        self.datadir = pl.Path(self.outputdir / 'Data Files')
        self.plotdir = pl.Path(self.outputdir / 'Plots')
        self.samplesdir = pl.Path(self.outputdir / 'Samples')
        self.statsdir = pl.Path(self.outputdir / 'Statistics')

        # If samples are to be processed and output data directory exists,
        # the directory will be removed with all files as not to interfere
        # with analysis.
        if not only_vectors:
            self.clear_analysis()

        # Create output directories
        pl.Path.mkdir(self.outputdir, exist_ok=True)
        pl.Path.mkdir(self.plotdir, exist_ok=True)
        pl.Path.mkdir(self.samplesdir, exist_ok=True)
        pl.Path.mkdir(self.datadir, exist_ok=True)
        pl.Path.mkdir(self.statsdir, exist_ok=True)
        if LAM_logger is not None:
            lg.logprint(LAM_logger, 'Directories successfully created.', 'i')

    def clear_analysis(self):
        """Clear existing analysis data files."""
        if self.datadir.exists() and Sett.process_counts:
            if Sett.force_dialog:
                shutil.rmtree(self.datadir)
            files = list(self.datadir.glob('*'))
            if files:
                flag = 1
                msg = "Data Files-folder will be cleared. Continue? [y/n]"
                print('\a')
                while flag:
                    ans = ask_user(msg)
                    if ans in ("y", "Y"):
                        flag = 0
                        shutil.rmtree(self.datadir)
                    elif ans in ("n", "N"):
                        print('Analysis terminated')
                        raise KeyboardInterrupt
                    else:
                        print('Command not understood.')

    def save_analysis_info(self, samples, groups, channels):
        """For saving information of all analyzed samples."""
        with open(str(self.outputdir.joinpath('Analysis_info.txt')), 'w') as f:
            f.write('GROUPS:\t')
            f.write(', '.join(groups))
            f.write('\nSAMPLES:\t')
            f.write(', '.join(samples))
            f.write('\nCHANNELS:\t')
            f.write(', '.join(channels))
        lg.logprint(LAM_logger, 'Analysis info successfully saved.', 'i')


class DataHandler:
    """
    Handle data for plotting.

    Data will be passed to plot.MakePlot-class
    """

    def __init__(self, samplegroups, in_paths, savepath=None):
        if savepath is None:
            self.savepath = samplegroups.paths.plotdir
        else:
            self.savepath = savepath
        self.palette = samplegroups.grp_palette
        self.center = samplegroups.center_bin
        self.total_length = samplegroups.bin_length
        self.MPs = samplegroups.sample_mps
        self.paths = in_paths

    def get_data(self, *args, **kws):
        """Collect data from files and modify."""

        melt = False

        # Create a DF for holding data of all samples, add identifiers, format:
        all_data = pd.DataFrame()
        for path in self.paths:
            data = read_data(path, header=0, test=False)
            if 'IDs' in kws.keys():  # Identifiers
                data = plot.identifiers(data, path, kws.get('IDs'))
            if 'melt' in kws.keys():  # Data to long format
                m_kws = kws.get('melt')
                if 'path_id' in args:  # Get ID from filepath
                    id_sep = kws.get('id_sep')
                    try:
                        id_var = path.stem.split('_')[id_sep]
                        m_kws.update({'value_name': id_var})
                    except IndexError:
                        msg = 'Faulty list index. Incorrect file names?'
                        print('ERROR: {}'.format(msg))
                        lg.logprint(LAM_logger, msg, 'e')
                data = data.T.melt(id_vars=m_kws.get('id_vars'), value_vars=m_kws.get('value_vars'),
                                   var_name=m_kws.get('var_name'), value_name=m_kws.get('value_name'))
                data = data.dropna(subset=[m_kws.get('value_name')])
                melt = True
            else:
                data = data.T
            if 'merge' in args:  # Merge data with data from other paths
                if all_data.empty:
                    all_data = data
                else:
                    all_data = all_data.merge(data, how='outer', copy=False, on=kws.get('merge_on'))
                continue

            # If not merging, concatenate the data with others
            all_data = pd.concat([all_data, data], sort=True)

        # Give proper index to full data
        all_data.index = pd.RangeIndex(stop=all_data.shape[0])

        # Drop outliers
        if 'drop_outlier' in args and Sett.Drop_Outliers:
            all_data = drop_outliers(all_data, melt, **kws)

        # Determine column data types
        all_data = all_data.infer_objects()
        return all_data

    def get_sample_data(self, col_ids, *args, **kws):
        """Collect data from channel-specific sample files."""
        melt = False
        error = False

        all_data = pd.DataFrame()  # Holds full data
        for path in self.paths:
            data = read_data(path, header=0, test=False)
            col_list = ['DistBin']
            for key in col_ids:
                col_list.extend([c for c in data.columns if key in c])
                # temp = data.loc[:, data.columns.str.contains(key)]
            sub_data = data.loc[:, col_list].sort_values('DistBin')

            if 'no_var' not in args:
                # Test for missing variables:
                for col in sub_data.columns:
                    # If no variance, drop data
                    if sub_data.loc[:, col].nunique() == 1:
                        sub_data.drop(col, axis=1, inplace=True)
            # Add identifier columns and melt data
            sub_data.loc[:, 'Channel'] = path.stem
            sub_data.loc[:, 'Sample Group'] = str(path.parent.name).split('_')[0]

            if 'melt' in kws.keys():
                mkws = kws.get('melt')
                try:
                    sub_data = sub_data.melt(id_vars=mkws.get('id_vars'), value_vars=mkws.get('value_vars'),
                                             var_name=mkws.get('var_name'), value_name=mkws.get('value_name'))
                except KeyError:
                    error = True
                melt = True

            # Concatenate new data to full set
            if not error:
                all_data = pd.concat([all_data, sub_data], sort=True)

        # Drop outliers if needed
        if 'drop_outlier' in args and Sett.Drop_Outliers and not all_data.empty:
            all_data = drop_outliers(all_data, melt, **kws)

        # Infer datatype of columns
        all_data = all_data.infer_objects()
        return all_data


def read_data(filepath, header=Sett.header_row, test=True, index_col=False):
    """Read csv-data."""

    try:  # Read data
        data = pd.read_csv(filepath, header=header, index_col=index_col)
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        if test:  # Test that the data contains ID columns
            try:
                data.loc[:, 'ID']
            except KeyError:
                msg = f'Column label test failed: ID not present at {filepath}'
                lg.logprint(LAM_logger, msg, 'ex')
                print(f'WARNING: read_data() call from {inspect.stack()[1][1]} line {inspect.stack()[1][2]}')
                print("Key 'ID' not found. Verify header row setting.\n\n")
                print(f"Path: {filepath}\n")

    except FileNotFoundError:
        lg.logprint(LAM_logger, 'File not found at {}'.format(filepath), 'e')
        print(f'WARNING: read_data() call from {inspect.stack()[1][1]} line {inspect.stack()[1][2]}')
        print(f'File {filepath.name} not found at {str(filepath.parent)}')
        return None

    except (AttributeError, pd.errors.EmptyDataError) as err:
        if isinstance(err, pd.errors.EmptyDataError):
            msg = "{} is empty. Skipped.".format(filepath.name)
            print("ERROR: {}".format(msg))
            lg.logprint(LAM_logger, msg, 'e')
            return None
        msg = f"Data or columns may be faulty in {filepath.name}"
        print("WARNING: {}".format(msg))
        lg.logprint(LAM_logger, msg, 'w')
        return data

    except pd.errors.ParserError:
        msg = f"{filepath} cannot be read."
        print(f"ERROR: {msg}")
        print("\nWrong header row?")
        lg.logprint(LAM_logger, msg, 'ex')
    return data


def save_to_file(data, directory, filename, append=True, w_index=False):
    """Save series or DF to a file."""
    path = directory.joinpath(filename)

    if not append:  # If saving full data, just save it
        if isinstance(data, pd.DataFrame):
            data.to_csv(str(path), index=w_index)
        else:
            data.to_frame().to_csv(str(path), index=w_index)

    elif path.exists():  # Add the data into an existing file
        file = pd.read_csv(str(path), index_col=w_index)
        if data.name not in file.columns:
            file = pd.concat([file, data], axis=1)
        else:
            file.loc[:, data.name] = data
        file = file.sort_index(axis=1)
        file.to_csv(str(path), index=w_index)

    else:  # If path doesn't yet exist, create it
        if isinstance(data, pd.DataFrame):
            data.to_csv(str(path), index=w_index)
        else:
            data.to_frame().to_csv(str(path), index=w_index)


def start(test_vectors=True, only_vectors=False):
    """Check that everything is OK when starting a run."""
    # If workdir variable isn't pathlib.Path, make it so
    if not isinstance(Sett.workdir, pl.Path):
        Sett.workdir = pl.Path(Sett.workdir)

    # Check that at least one primary setting is True
    if not any([Sett.process_samples, Sett.process_counts, Sett.Create_Plots, Sett.process_dists, Sett.statistics]):
        lg.logprint(LAM_logger, 'All primary settings are False', 'e')
        print("\nAll primary settings are set to False.\n\nExiting ...")
        raise SystemExit

    # Otherwise create paths and directories
    paths = Paths(Sett.workdir, only_vectors=only_vectors)

    if not test_vectors:
        return paths

    # Check that vector channel data are found
    if Sett.process_samples or (Sett.measure_width and Sett.process_counts):
        samples = [p for p in Sett.workdir.iterdir() if p.is_dir() and p.name != 'Analysis Data']
        failed = []
        for sample in samples:
            try:
                next(sample.glob(f'*_{Sett.vectChannel}_*'))
            except StopIteration:
                failed.append(sample.name)
        if failed:
            msg = f"Vector channel data not found for {', '.join(failed)}"
            print(f'ERROR: {msg}')
            print('Check vector channel setting or data.')
            lg.logprint(LAM_logger, msg, 'e')
            raise SystemExit

    # Find and store all sample names
    Store.samples = [p.name for p in paths.samplesdir.iterdir() if p.is_dir()]
    return paths


def test_vector_ext(dir_path):
    """Test if vectors exist and ask permission to remove."""

    if Sett.force_dialog:  # If forcing no dialog
        return

    # Get all existing sample output folders
    samples = iter([p for p in dir_path.iterdir() if p.is_dir()])
    if samples is None:  # if no pre-made samples found, continue analysis
        return

    # Loop samples and find any vector file:
    for smpl in samples:
        test = any([re.match(re.compile(".*vector.*", re.I), str(p.name)) for p in smpl.glob('*')])
        # If a vector file is found, ask permission to remove:
        if test:
            flag = 1
            print('\a')
            msg = "Pre-existing vectors will be cleared. Continue? [y/n]"
            while flag:
                ans = ask_user(msg)
                if ans in ("y", "Y"):
                    return
                if ans in ("n", "N"):
                    print('Analysis terminated')
                    raise KeyboardInterrupt
                print('Command not understood.')


def drop_outliers(all_data, melted=False, raw=False, **kws):
    """Drop outliers from full dataset."""

    def drop(data, col):
        """Drop outliers from a dataframe."""
        # Get mean and std of input data
        if raw:
            values = data
        else:
            values = data.loc[:, col].sort_values(ascending=False)
        with warnings.catch_warnings():  # Ignore empty bin warnings
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mean = np.nanmean(values.astype('float'))
            std = np.nanstd(values.astype('float'))
        drop_val = Sett.dropSTD * std
        if raw:  # If data is not melted, replace outliers with NaN
            data.where(np.abs(values - mean) <= drop_val, other=np.nan, inplace=True)
        else:  # If data is melted and sorted, find indexes until val < drop
            idx = []
            for ind, val in values.iteritems():
                if np.abs(val - mean) < drop_val:
                    break
                idx.append(ind)
            # Select data that fills criteria for validity
            data = data.loc[(data.index.difference(idx)), :]
        return data

    if raw:
        all_data = drop(all_data, col=None)
        return all_data

    # Handle data for dropping
    if 'drop_grouper' in kws.keys():
        grouper = kws.get('drop_grouper')
    else:
        grouper = 'Sample Group'

    grp_data = all_data.groupby(by=grouper)

    if melted:
        names = kws['melt'].get('value_name')
    else:
        names = all_data.loc[:, all_data.columns != grouper].columns
    all_data = grp_data.apply(lambda grp: drop(grp, col=names))

    if isinstance(all_data.index, pd.MultiIndex):
        all_data = all_data.droplevel(grouper)
    return all_data


def ask_user(dlg, dlgtype='string', use_gui=Sett.GUI):
    """Create input dialog."""
    if use_gui:
        ans = None
        if dlgtype == 'string':
            ans = sd.askstring(title="Dialog", prompt=dlg)
        elif dlgtype == 'integer':
            ans = sd.askinteger(title="Dialog", prompt=dlg)
    else:
        ans = input(dlg)
    return ans


def read_vector(vector_paths):
    """Read vector and transform to LineString."""

    # Read depending on file type
    path = [path for path in vector_paths if path.suffix == '.txt']
    if path:
        vector_df = pd.read_csv(path[0], sep="\t", header=None)
        vector_df.columns = ["X", "Y"]
    else:
        vector_df = pd.read_csv(vector_paths[0])

    # Reformat data and change to LineString
    vector_coords = list(zip(vector_df.loc[:, 'X'].astype('float'),
                    vector_df.loc[:, 'Y'].astype('float')))
    vector = gm.LineString(vector_coords)
    return vector
