# -*- coding: utf-8 -*-
"""
LAM-module for vector creation, and data collection and projection.

Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen

"""
# Standard libraries
from decimal import Decimal
import inspect
import math
import re
import warnings

# Other packages
import numpy as np
import pandas as pd
import pathlib as pl
import shapely.geometry as gm
from shapely.ops import polygonize
from scipy.ndimage import morphology as mp
from skimage.morphology import medial_axis
from skimage.filters import gaussian
from skimage.transform import resize as resize_arr
from skimage.measure import find_contours

# LAM modules
from src.settings import Store, Settings as Sett
import src.plotfuncs as pfunc
import src.logger as lg
import src.system as system

LAM_logger = None


class GetSample:
    """Collect sample data and process for analysis."""

    def __init__(self, sample_path: pl.Path, paths: system.Paths, process=True, projection=False):
        self.name = sample_path.stem
        self.sampledir = paths.samplesdir.joinpath(self.name)
        self.group = self.name.split('_')[0]
        # Add sample and group to storing variables
        if self.name not in Store.samples:
            Store.samples.append(self.name)
            Store.samples = sorted(Store.samples)
        if self.group not in Store.samplegroups:
            Store.samplegroups.append(self.group)
            Store.samplegroups = sorted(Store.samplegroups)
        # Make folder for storing data and find data-containing files
        if not self.sampledir.exists():
            pl.Path.mkdir(self.sampledir)
        self.channelpaths = list([p for p in sample_path.iterdir() if p.is_dir()])
        self.channels = [str(p).split('_')[(-2)] for p in self.channelpaths]
        self.vect_data = None
        self.MP = None
        self.data = None
        self.vector = None
        self.vector_length = None

        if process is False and projection is True:
            for channel in self.channels:  # Store all found channel names
                if (channel.lower() not in [c.lower() for c in Store.channels] and
                        channel.lower() != Sett.MPname.lower()):
                    Store.channels.append(channel)
            self.find_sample_vector(paths.datadir)

    def find_sample_vector(self, path):  # path = data directory
        """Find sample's vector data."""
        try:  # Find sample's vector file
            paths = list(self.sampledir.glob('Vector.*'))
            self.vector = system.read_vector(paths)
            self.vector_length = self.vector.length
            length_series = pd.Series(self.vector_length, name=self.name)
            system.save_to_file(length_series, path, 'Length.csv')

        # If vector file not found
        except (FileNotFoundError, IndexError):
            msg = f'Vector-file NOT found for {self.name}'
            lg.logprint(LAM_logger, msg, 'e')
            print(f'ERROR: {msg}')
        except (AttributeError, ValueError):  # If vector file is faulty
            msg = f'Faulty vector for {self.name}'
            lg.logprint(LAM_logger, msg, 'c')
            print(f'CRITICAL: {msg}')

    def get_vect_data(self, channel):
        """Get channel data that is used for vector creation."""
        try:
            # Search string:
            namer = str("_{}_".format(channel))
            namerreg = re.compile(namer, re.I)
            # Search found paths with string
            dir_path = [self.channelpaths[i] for i, s in enumerate(self.channelpaths)
                        if namerreg.search(str(s))][0]
            vect_path = next(dir_path.glob('*Position.csv'))
            vect_data = system.read_data(vect_path, header=Sett.header_row)  # Read data
        except (FileNotFoundError, IndexError):  # If data file not found
            msg = 'No valid datafile for vector creation.'
            if LAM_logger is not None:
                lg.logprint(LAM_logger, msg, 'w')
            print('-> {}'.format(msg))
            vect_data = None
        return vect_data

    def create_skeleton(self):
        # Extract point coordinates of the vector:
        positions = self.vect_data
        x, y = positions.loc[:, 'Position X'], positions.loc[:, 'Position Y']
        coord_df, bin_array, skeleton = self.binarize_coords(x, y, Sett.SkeletonResize,
                                                            Sett.BDiter, Sett.SigmaGauss)
        line_df = self.skeleton_vector(coord_df)
        if line_df is not None and not line_df.empty:
            system.save_to_file(line_df, self.sampledir, 'Vector.csv', append=False)
        pfunc.skeleton_plot(self.sampledir, self.name, bin_array, skeleton)

    def create_median(self):
        # Extract point coordinates of the vector:
        positions = self.vect_data
        x, y = positions.loc[:, 'Position X'], positions.loc[:, 'Position Y']
        line_df = self.median_vector(x, y, Sett.medianBins)
        if line_df is not None and not line_df.empty:
            system.save_to_file(line_df, self.sampledir, 'Vector.csv', append=False)

    def binarize_coords(self, x_values, y_values, resize: float, bd_iter: int, sigma_gauss: float):
        """Create binary image from cell coordinates."""

        def _binarize():
            """Transform XY into binary image and perform operations on it."""
            # Create DF indices (X&Y-coords) with a buffer for operations:
            buffer = 1000 * resize
            # Get needed axis related variables:
            x_max, x_min = round(max(x_values) + buffer), round(min(x_values) - buffer)
            y_max, y_min = round(max(y_values) + buffer), round(min(y_values) - buffer)
            y_size = round(y_max - y_min)
            x_size = round(x_max - x_min)

            # Create binary array
            binary_arr = np.zeros((y_size, x_size))
            for coord in coords:  # Set cell locations in array to True
                binary_arr[round(coord[1] - y_min),
                           round(coord[0] - x_min)] = 1
            if resize != 1:
                y_size = round(y_size * resize)
                x_size = round(x_size * resize)
                binary_arr = resize_arr(binary_arr, (y_size, x_size))
                binary_arr = np.where(binary_arr > 0, 1, 0)
            # Create Series to store real coordinate labels
            x_lbl = pd.Series(np.linspace(x_min, x_max, x_size), index=pd.RangeIndex(binary_arr.shape[1]))
            y_lbl = pd.Series(np.linspace(y_min, y_max, y_size), index=pd.RangeIndex(binary_arr.shape[0]))

            # BINARY DILATION
            if bd_iter > 0:
                struct = mp.generate_binary_structure(2, 2)
                binary_arr = mp.binary_dilation(binary_arr, iterations=int(bd_iter), structure=struct)

            # SMOOTHING
            if sigma_gauss > 0:  # Gaussian smoothing
                binary_arr = gaussian(binary_arr, sigma=sigma_gauss)

            # FIND CONTOURS
            contours = find_contours(binary_arr, 0.5)
            continuous = contours
            try:
                pol = gm.MultiPolygon(polygonize(continuous))
            except (TypeError, NotImplementedError):
                pol = gm.Polygon(continuous)

            # CREATE ARRAY WITH FILLED SAMPLE OUTLINE
            segm = _intersection(binary_arr.shape, pol)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                bool_bin_arr = mp.binary_fill_holes(segm)
            return bool_bin_arr, y_lbl, x_lbl

        def _intersection(shape: tuple, pols: gm.multipolygon) -> np.ndarray:
            """Create binary array by filling given polygons."""
            segment_array = np.zeros(shape)

            # Loop rows of array
            for ind in np.arange(shape[0]+1):
                # Create LineString from row and see where it intersects with polygons
                row = gm.LineString([(ind, 0), (ind, shape[1]+1)])
                section = row.intersection(pols)
                if section.is_empty:
                    continue

                # Test the datatype of intersection:
                if isinstance(section, gm.LineString):
                    _, miny, _, maxy = section.bounds
                    # Assign True to elements that fall within the polygons
                    segment_array[ind, round(miny):round(maxy)] = 1
                # If the results gave collections of objects:
                elif isinstance(section, (gm.MultiLineString, gm.collection.GeometryCollection)):
                    # Assign values from each object
                    for geom in section.geoms:
                        _, miny, _, maxy = geom.bounds
                        segment_array[ind, round(miny):round(maxy)] = 1
            return segment_array

        coords = list(zip(x_values, y_values))
        # Transform to binary
        bin_array, bin_arr_ind, bin_arr_cols = _binarize()
        # Make skeleton and get coordinates of skeleton pixels
        # skeleton = skeletonize(bin_array)
        skeleton = medial_axis(bin_array)
        skel_values = [(bin_arr_ind.iat[y], bin_arr_cols.iat[x]) for y, x in zip(*np.where(skeleton == 1))]
        # Dataframe from skeleton coords
        coord_df = pd.DataFrame(skel_values, columns=['Y', 'X']).infer_objects()
        return coord_df, bin_array, skeleton

    def skeleton_vector(self, coord_df):
        """Create vector by skeletonization of image-transformed positions."""

        def _score_nearest(test_point):
            # DataFrame for storing relevant info on pixel coordinates
            score = pd.DataFrame(np.zeros((nearest.size, 6)), index=nearest,
                                 columns=['rads', 'dist_test', 'dist_vect', 'penalty', 'X', 'Y'])
            score.X, score.Y = coord_df.X, coord_df.Y

            # Get direction of test point
            test_x, test_y = test_point.x - last_point.x, test_point.y - last_point.y
            test_rad = math.atan2(test_y, test_x)
            # Calculate scoring variables
            shifts = pd.DataFrame(data=(score.X - last_point.x, score.Y - last_point.y), columns=nearest).T
            score.rads = shifts.apply(lambda p, r=test_rad: abs(math.atan2(p.iat[1], p.iat[0]) - r), axis=1)
            score.dist_vect = score.apply(lambda p, t=last_point: t.distance(gm.Point(p.X, p.Y)), axis=1)
            score.dist_test = score.apply(lambda p, t=test_point: t.distance(gm.Point(p.X, p.Y)), axis=1)
            score.penalty = score.dist_vect + score.dist_test + (score.rads * 10)
            # Drop values that would turn the vector to move backwards
            score.penalty.loc[score.rads > 1.9] = np.nan
            #print(sum(abs(shifts.loc[score.penalty.idxmin(), :].values)))
            return score

        def _find_pixel(line, s_x, s_y, flag=False):
            # Establish the vector's direction and project a forward point:
            test_point = define_scoring_point(line, s_x, s_y)
            # Calculate scoring of pixels
            scores = _score_nearest(test_point)
            # Get the pixels that are behind current vector coord
            forfeit = scores.loc[((scores.dist_test > scores.dist_vect) & scores.rads > 1.3) |
                                 (scores.penalty == np.nan)].index

            # Find the pixel with the smallest penalty and add to vector:
            try:
                best = scores.penalty.idxmin()
                x_2, y_2 = coord_df.X.at[best], coord_df.Y.at[best]

                # Drop used pixel and pixels falling behind vector
                forfeit = forfeit.append(pd.Index([best], dtype='int64'))
                coord_df.drop(forfeit, inplace=True)

                # Set found pixel for the next loop
                line.append((x_2, y_2))
                s_x, s_y = x_2, y_2

            except (ValueError, KeyError):
                flag = True

            return line, s_x, s_y, flag


        # BEGIN CREATION OF VECTOR FROM SKELETON COORDS
        finder = Sett.find_dist  # Distance for detection of nearby XY
        line = []  # For storing vector
        # Start from mean coordinates of pixels with smallest x-coords
        start = coord_df.nsmallest(5, 'X').idxmin()
        s_x, s_y = coord_df.loc[start, 'X'].mean(), coord_df.loc[start, 'Y'].mean()
        line.append((s_x, s_y))
        coord_df.drop(start, inplace=True)  # Drop the start coordinates from data
        s_x, s_y = s_x + finder / 4, s_y

        # Continue finding next pixels until flagged ready:
        flag = False
        while not flag:
            # Current end point of line:
            last_point = gm.Point(line[-1])

            # Find pixels near to the current coordinate
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                nearest = coord_df[(abs(coord_df.X - s_x) <= finder) &
                                   (abs(coord_df.Y - s_y) <= finder)].index

            if nearest.size >= 1:
                line, s_x, s_y, flag = _find_pixel(line, s_x, s_y)
            else:
                flag = True

        # Create LineString-object from finished vector:
        try:
            xy_coord = gm.LineString(line).simplify(Sett.simplifyTol).xy
            linedf = pd.DataFrame(data=list(zip(xy_coord[0], xy_coord[1])), columns=['X', 'Y'])

        # If something went wrong with creation, warn
        except (ValueError, AttributeError):
            linedf = pd.DataFrame().assign(X=[line[0][0]], Y=[line[0][1]])
            msg = 'Faulty vector for {}'.format(self.name)
            if LAM_logger is not None:
                lg.logprint(LAM_logger, msg, 'e')
            print("WARNING: Faulty vector. Try different settings")

        return linedf

    def median_vector(self, x_values, y_values, creation_bins):
        """Create vector by calculating median coordinates."""
        # Divide sample to equidistant points between min & max X-coord:
        bins = np.linspace(x_values.min(), x_values.max(), creation_bins)
        idx = np.digitize(x_values, bins, right=True)
        y_median = np.zeros(creation_bins)
        # Find median Y-coord at first bin:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            startval = np.nanmean(y_values[(idx == 1)])
        y_median[0] = startval
        # Then find median for the rest of the bins
        for b in range(1, creation_bins):
            cells = y_values[idx == b]
            if cells.size == 0:  # If no cells at bin, copy previous Y-coord
                y_median[b] = y_median[b - 1]
            else:
                y_median[b] = y_values[idx == b].min() + (y_values[idx == b].max() - y_values[idx == b].min()) / 2
        # Change bins and their medians into XY-coordinates
        xy_median = [p for p in tuple(np.stack((bins, y_median), axis=1)) if ~np.isnan(p).any()]
        # Create LineString-object from finished vector, simplify, and get new coords
        xy_coord = gm.LineString(xy_median).simplify(Sett.simplifyTol).xy
        linedf = pd.DataFrame(data=list(zip(xy_coord[0], xy_coord[1])), columns=['X', 'Y'])
        return linedf

    def get_mps(self, mp_name: str, use_mp: bool, datadir: pl.Path) -> pd.Series:
        """Collect MPs for sample anchoring."""
        if use_mp:
            try:  # Get measurement point for anchoring
                mp_dir_path = next(self.channelpaths.pop(i) for i, s in enumerate(self.channelpaths) if
                                   str('_' + mp_name + '_') in str(s))
                mp_path = next(mp_dir_path.glob("*Position.csv"))
                mp_data = system.read_data(mp_path, header=Sett.header_row, test=False)
                mp_data = mp_data.loc[:, ['Position X', 'Position Y']]
                if not mp_data.empty:
                    mp_bin = self.project_mps(mp_data, datadir, filename="MPs.csv")
                    mp_df = pd.DataFrame({'MP': mp_bin.values.codes})
                    mp_df.to_csv(self.sampledir.joinpath("MPs.csv"), index=False)
            except (StopIteration, ValueError, UnboundLocalError):
                mp_bin = None
                msg = f'could not find MP position for {self.name}'
                lg.logprint(LAM_logger, msg, 'e')
                print("    -> Failed to find MP position data.")
        else:  # Sets measurement point values to zero when MP's are not used
            mp_bin = pd.Series(0, name=self.name)
            system.save_to_file(mp_bin, datadir, "MPs.csv")
            system.save_to_file(mp_bin, self.sampledir, "MPs.csv", append=False)
        return mp_bin

    def project_mps(self, positions, datadir, filename="some.csv"):
        """For the projection of spot coordinates onto the vector."""
        xy_positions = list(zip(positions['Position X'], positions['Position Y']))
        # The shapely packages reguires transformation into Multipoints for the
        # projection.
        points = gm.MultiPoint(xy_positions)
        # Find point of projection on the vector.
        positions["VectPoint"] = [self.vector.interpolate(self.vector.project(gm.Point(x))) for x in points]
        # Find normalized distance (0->1)
        positions["NormDist"] = [self.vector.project(x, normalized=True) for x in positions["VectPoint"]]
        # Find the bins that the points fall into
        # Determine bins of each feature
        edges = np.linspace(0, 1, Sett.projBins+1)
        labels = np.arange(0, Sett.projBins)
        positions["DistBin"] = pd.cut(positions["NormDist"], edges, labels=labels)
        mp_bin = pd.Series(positions.loc[:, "DistBin"], name=self.name)
        self.data = positions
        self.test_projection(Sett.MPname)
        # Save the obtained data:
        system.save_to_file(mp_bin.astype(int), datadir, filename)
        return mp_bin

    def project_channel(self, channel):
        """For projecting coordinates onto the vector."""
        data = channel.data
        xy_positions = list(zip(data['Position X'], data['Position Y']))
        # Transformation into Multipoints required for projection:
        points = gm.MultiPoint(xy_positions)
        # Find projection distance on the vector.
        proj_vector_dist = [self.vector.project(x) for x in points]
        # Find the exact point of projection
        proj_points = [self.vector.interpolate(p) for p in proj_vector_dist]
        # Find distance between feature and the point of projection
        proj_dist = [p.distance(proj_points[i]) for i, p in enumerate(points)]
        # Find normalized distance (0->1)
        data["NormDist"] = [d / self.vector_length for d in proj_vector_dist]
        # Determine bins of each feature
        edges = np.linspace(0, 1, Sett.projBins+1)
        labels = np.arange(0, Sett.projBins)
        data["DistBin"] = pd.cut(data["NormDist"], labels=labels, bins=edges, include_lowest=True).astype('int')

        # Assign data to DF and save the dataframe:
        data["VectPoint"] = [(round(p.x, 3), round(p.y, 3)) for p in proj_points]
        data["ProjDist"] = proj_dist
        self.data = data
        self.test_projection(channel.name)
        channel_string = f'{channel.name}.csv'
        system.save_to_file(data, self.sampledir, channel_string, append=False)
        return data

    def find_counts(self, channel_name, datadir):
        """Gather projected features and find bin counts."""
        counts = np.bincount(self.data['DistBin'], minlength=Sett.projBins)
        counts = pd.Series(np.nan_to_num(counts), name=self.name)
        channel_string = f'All_{channel_name}.csv'
        system.save_to_file(counts, datadir, channel_string)
        if channel_name == Sett.vectChannel:
            test_count_projection(counts, self.name)

    def test_projection(self, name):
        if self.data["DistBin"].isna().any():
            msg = "All features were not projected. Check vector and data."
            print(f"   -> {name}: {msg}")


class VectorError(Exception):
    """Exception when missing sample vectors."""
    def __init__(self, samples, message='CRITICAL: vectors not found for all samples.'):
        self.samples = samples
        self.message = message
        super().__init__(self.message)


class GetChannel:
    """Find and read channel data plus additional data."""

    def __init__(self, path, sample, data_keys, datadir):
        self.sample = sample
        self.datafail = []
        self.datadir = datadir
        self.name = str(path.stem).split('_')[-2]
        self.path = path
        self.pospath = next(self.path.glob("*Position.csv"))
        self.data = self.read_channel(self.pospath)
        self.read_additional(data_keys)
        if 'ClusterID' in self.data.columns:
            Store.clusterPaths.append(self.path)

    def read_channel(self, path):
        """Read channel data into a dataframe."""
        try:
            data = system.read_data(str(path), header=Sett.header_row)
            channel = self.name
            if channel.lower() not in [c.lower() for c in Store.channels] and channel.lower() != Sett.MPname.lower():
                Store.channels.append(self.name)
            return data
        except ValueError:
            lg.logprint(LAM_logger, 'Cannot read channel path {}'.format(path),
                        'ex')

    def read_additional(self, data_keys):
        """Read relevant additional data of channel."""

        def _test_variance(data):
            """Test if additional data column contains variance."""
            for column in data.columns.difference(['ID']):
                test = data.loc[:, column].dropna()
                test = (test - test.min()) / test.max()
                if test.std() < 0.01:
                    self.datafail.append(column)
                    data.loc[:, column] = np.nan
            return data

        def _rename_id(data):
            """Rename filename identification of channel."""
            # I.e. as defined by settings.channelID
            for column in data.columns:
                id_str = str(column).split('_')[-1]
                if id_str in Sett.channelID.keys():
                    new_id = Sett.channelID.get(id_str)
                    data.rename(columns={column: column.replace(f'_{id_str}', f'-{new_id}')}, inplace=True)
            return data

        add_data = pd.DataFrame(self.data.loc[:, 'ID'])
        for key, values in data_keys.items():
            paths = list(self.path.glob(f'*{values[0]}*'))
            if not paths:
                print(f"-> {self.name} {key} file not found")
                continue
            if len(paths) == 1:
                namer = re.compile(f'^{key}', re.I)
                if paths[0] == self.pospath and any(self.data.columns.str.contains(namer)):
                    continue
                if paths[0] == self.pospath and not any(self.data.columns.str.contains(namer)):
                    print(f"'{key}' not in {self.pospath.name} of {self.sample.name} on channel {self.name}")
                temp_data = system.read_data(str(paths[0]), header=Sett.header_row)
                cols = temp_data.columns.map(lambda x, matcher=namer: bool(re.match(matcher, x)) or x == 'ID')
                temp_data = temp_data.loc[:, cols]
                add_data = pd.merge(add_data, temp_data, on='ID')
            else:  # If multiple files, e.g. intensity, get all
                for path in paths:
                    # Search identifier for column from filename
                    strings = str(path.stem).split(f'{values[0]}_')
                    id_string = strings[1].split('_')[0]
                    # Locate columns
                    temp_data = system.read_data(str(path), header=Sett.header_row)
                    temp_data = temp_data.loc[:, [key, 'ID']]
                    for col in [c for c in temp_data.columns if c != 'ID']:
                        rename = str(col + '_' + id_string)
                        temp_data.rename(columns={key: rename}, inplace=True)
                    add_data = pd.merge(add_data, temp_data, on='ID')
        # Drop invariant data
        add_data = _test_variance(add_data)
        if Sett.replaceID:
            add_data = _rename_id(add_data)
        self.data = pd.merge(self.data, add_data, on='ID')


class Normalize:
    """Anchor sample data into dataframe with all samples."""

    def __init__(self, path):
        self.path = pl.Path(path)
        self.channel = str(self.path.stem).split('_')[1]
        self.counts = system.read_data(path, header=0, test=False)
        self.starts = None

    def averages(self, norm_counts: pd.DataFrame):
        """Find bin averages of channels."""
        # Find groups of each sample based on samplenames
        samples = norm_counts.columns.tolist()
        groups = set({s.casefold(): s.split('_')[0] for s in samples}.values())
        cols = ["{}_All".format(g) for g in groups]
        averages = pd.DataFrame(index=norm_counts.index, columns=cols)
        for grp in groups:  # For each group found in data
            namer = "{}_".format(grp)
            group_data = norm_counts.loc[:, norm_counts.columns.str.startswith(namer)]
            # Calculate group averages
            averages.loc[:, "{}_All".format(grp)] = group_data.mean(axis=1)
        # Save average data
        filename = str('ChanAvg_{}.csv'.format(self.channel))
        system.save_to_file(averages, self.path.parent, filename, append=False)

    def avg_add_data(self, paths: system.Paths, data_names: dict, total_len: int):
        """Find bin averages of additional data."""
        samples = self.starts.index
        for sample in samples:
            sample_dir = paths.samplesdir.joinpath(sample)
            data_file = sample_dir.glob(str(self.channel + '.csv'))
            data = system.read_data(next(data_file), header=0)
            for data_type in data_names.keys():
                sample_data = data.loc[:, data.columns.str.contains(str(data_type))]
                if sample_data.empty:
                    continue
                binned_data = data.loc[:, 'DistBin']
                bins = np.arange(0, Sett.projBins)
                for col in sample_data:
                    avg_s = pd.Series(np.full(total_len, np.nan), name=sample)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        insert = [np.nanmean(sample_data.loc[binned_data == i, col]) for i in bins]
                        insert = [0 if np.isnan(v) else v for v in insert]
                    start = int(self.starts.at[sample])
                    end = int(start + Sett.projBins)
                    avg_s[start:end] = insert
                    filename = str('Avg_{}_{}.csv'.format(self.channel, col))
                    system.save_to_file(avg_s, paths.datadir, filename)

    def normalize_samples(self, mps, array_length, center, name=None):
        """For inserting sample data into larger matrix, centered with MP."""
        # Create empty data array => insert in DF
        cols = self.counts.columns
        arr = np.full((array_length, len(cols)), np.nan)
        data = pd.DataFrame(arr, columns=cols)

        # Create empty series for holding each sample's starting index
        sample_start = pd.Series(np.full(len(cols), np.nan), index=cols)
        for col in self.counts.columns:
            handle = self.counts[col].values
            mp_bin = mps.at[0, col]
            # Insert sample's count data into larger, anchored dataframe:
            insert, insx = relate_data(handle, mp_bin, center, array_length)
            data[col] = insert
            # Save starting index of the sample
            sample_start.at[col] = insx

        check_anchor_quality(sample_start)
        # Save anchored data
        if name is None:
            name = f'Norm_{self.channel}'
        filename = f'{name}.csv'
        data = data.sort_index(axis=1)
        system.save_to_file(data, self.path.parent, filename, append=False)
        return sample_start, data


class DefineWidths:
    """Find widths of samples along the vector."""

    def __init__(self, data, vector, path, datadir):
        self.name = path.name
        self.sampledir = path
        self.data = data
        self.vector = vector

        # Determine width:
        self.data = self.point_handedness()
        self.average_width(datadir)

    def point_handedness(self):
        """
        Find handedness of projected points compared to vector.
        Returns DF with added column 'hand', with possible values [-1, 0, 1]
        that correspond to [right side, on vector, left side] respectively.
        """

        def _get_sign(arr, p1_x, p1_y, p2_x, p2_y):
            """Find which side of vector a feature is."""
            x_val, y_val = arr[0], arr[1]
            val = math.copysign(1, (p2_x - p1_x) * (y_val - p1_y) - (p2_y - p1_y) * (x_val - p1_x))
            return val

        # Define bin edges
        edges, edge_points = self.get_vector_edges(multip=2)
        data = self.data.sort_values(by='NormDist')
        # Find features in every bin and define hand-side
        for ind, point1 in enumerate(edge_points[:-1]):
            point2 = edge_points[ind+1]
            p1x, p1y = point1.x, point1.y
            p2x, p2y = point2.x, point2.y
            d_index = data.loc[(data.NormDist >= edges[ind]) & (data.NormDist < edges[ind+1])].index
            points = data.loc[d_index, ['Position X', 'Position Y']]
            # Assign hand-side of features
            data.loc[d_index, 'hand'] = points.apply(_get_sign, args=(p1x, p1y, p2x, p2y), axis=1, raw=True
                                                     ).replace(np.nan, 0)
        data = data.sort_index()
        # Save calculated data
        channel_string = str('{}.csv'.format(Sett.vectChannel))
        system.save_to_file(data, self.sampledir, channel_string, append=False)
        return data

    def get_vector_edges(self, multip=1, points=True):
        """Divide vector to segments."""
        edges = np.linspace(0, 1, Sett.projBins*multip)
        if points:
            edge_points = [self.vector.interpolate(d, normalized=True) for d in edges]
            return edges, edge_points
        return edges

    def average_width(self, datadir):
        """Calculate width based on feature distance and side."""

        def _get_approx_width(sub_data):
            """Approximate sample's width at bin."""
            width = 0
            for val in [-1, 1]:
                distances = sub_data.loc[(sub_data.hand == val)].ProjDist
                if not distances.empty:
                    temp = distances.groupby(pd.qcut(distances, 10, duplicates='drop')).mean()
                    if not temp.empty:
                        width += temp.tolist()[-1]
            return width

        edges = self.get_vector_edges(multip=2, points=False)
        cols = ['NormDist', 'ProjDist', 'hand']
        data = self.data.sort_values(by='NormDist').loc[:, cols]
        # Create series to hold width results
        res = pd.Series(name=self.name, index=pd.RangeIndex(stop=len(edges)))
        # Loop segments and get widths:
        for ind, _ in enumerate(edges[:-1]):
            d_index = data.loc[(data.NormDist >= edges[ind]) & (data.NormDist < edges[ind+1])].index
            res.iat[ind] = _get_approx_width(data.loc[d_index, :])
        filename = 'Sample_widths.csv'
        system.save_to_file(res, datadir, filename)


def create_samples(paths: system.Paths):
    """Create vectors for the samples."""
    lg.logprint(LAM_logger, 'Begin vector creation.', 'i')
    print("---Processing samples---")
    # Test that resize-setting is in step of 0.1:
    if Sett.SkeletonVector:
        check_resize_step(Sett.SkeletonResize)
    # Loop Through samples to create vectors
    for path in [p for p in Sett.workdir.iterdir() if p.is_dir() and p.stem != 'Analysis Data']:
        sample = GetSample(path, paths)
        print("{}  ...".format(sample.name))
        sample.vect_data = sample.get_vect_data(Sett.vectChannel)
        # Creation of vector for projection
        if Sett.SkeletonVector:
            sample.create_skeleton()
        else:
            sample.create_median()
    sample_dirs = [p for p in paths.samplesdir.iterdir() if p.is_dir()]
    pfunc.create_vector_plots(Sett.workdir, paths.samplesdir, sample_dirs)
    lg.logprint(LAM_logger, 'Vectors created.', 'i')


def find_existing(paths: system.Paths):
    """Get MPs and count old projections when not projecting during 'Count'."""
    msg = 'Collecting pre-existing data.'
    print(msg)
    lg.logprint(LAM_logger, msg, 'i')
    mps = pd.DataFrame(columns=Store.samples)
    for smpl in Store.samples:
        smplpath = paths.samplesdir.joinpath(smpl)
        # FIND MP
        if Sett.useMP:
            try:
                mp_df = pd.read_csv(smplpath.joinpath('MPs.csv'))
                mp_bin = mp_df.iat[0, 0]
            except FileNotFoundError:
                msg = "MP-data not found."
                add = "Provide MP-data or set useMP to False."
                print(f"ERROR: {msg}\n{add}")
                raise SystemExit
        else:
            mp_bin = 0
        mps.loc[0, smpl] = mp_bin
        # FIND CHANNEL COUNTS
        for path in [p for p in smplpath.iterdir() if p.suffix == '.csv' and p.stem not in ['Vector', 'MPs',
                                                                                            Sett.MPname]]:
            data = pd.read_csv(path)
            try:
                counts = np.bincount(data['DistBin'], minlength=Sett.projBins)
                counts = pd.Series(np.nan_to_num(counts), name=smpl)
                channel_string = str(f'All_{path.stem}.csv')
                system.save_to_file(counts, paths.datadir, channel_string)
            except ValueError:  # If channel has not been projected
                print(f"Missing projection data: {path.stem} - {smpl}")
                print("-> Set project=True and perform Count")
                continue
    mps.to_csv(paths.datadir.joinpath('MPs.csv'))
    samples = mps.columns.tolist()
    groups = set({s.casefold(): s.split('_')[0] for s in samples}.values())
    Store.samplegroups = sorted(groups)


def get_counts(paths):
    """Handle data to anchor samples and find cell counts."""
    try:  # Test that MPs are found for the sample
        mps = system.read_data(next(paths.datadir.glob('MPs.csv')), header=0, test=False)
    except (FileNotFoundError, StopIteration):
        msg = "MPs.csv NOT found!"
        print("ERROR: {}".format(msg))
        lg.logprint(LAM_logger, msg, 'c')
        msg = "-> Perform 'Count' before continuing.\n"
        print("{}".format(msg))
        lg.logprint(LAM_logger, msg, 'i')
        raise SystemExit

    # Find the smallest and largest anchor bin-number of the dataset
    mp_max, mp_min = mps.max(axis=1).values[0], mps.min(axis=1).values[0]
    # Store the bin number of the row onto which samples are anchored to
    Store.center = mp_max
    # Find the size of needed dataframe, i.e. so that all anchored samples fit
    mp_diff = mp_max - mp_min

    if not any([Sett.process_counts, Sett.process_samples]):
        # Find all sample groups in the analysis from the found MPs.
        found_samples = [p for p in paths.samplesdir.iterdir() if p.is_dir()]
        samples = mps.columns.tolist()
        if len(found_samples) != len(samples):  # Test whether sample numbers match
            msg = "Mismatch of sample N between MPs.csv and sample folders"
            print('WARNING: {}'.format(msg))
            lg.logprint(LAM_logger, msg, 'w')
        groups = set({s.casefold(): s.split('_')[0] for s in samples}.values())
        Store.samplegroups = sorted(groups)
        Store.channels = [c.stem.split('_')[1] for c in paths.datadir.glob("All_*.csv")]
        try:  # If required lengths of matrices haven't been defined because
            # Process and Count are both False, get the sizes from files.
            chan = Sett.vectChannel
            path = paths.datadir.joinpath("Norm_{}.csv".format(chan))
            temp = system.read_data(path, test=False, header=0)
            Store.totalLength = temp.shape[0]  # Length of anchored matrices
            path = paths.datadir.joinpath("All_{}.csv".format(chan))
            temp = system.read_data(path, test=False, header=0)
            Sett.projBins = temp.shape[0]
        except AttributeError:
            msg = "Cannot determine length of sample matrix\n-> Must perform 'Count' before continuing."
            lg.logprint(LAM_logger, msg, 'c')
            print("ERROR: {}".format(msg))
        return

    # The total length of needed matrix when using 'Count'
    Store.totalLength = int(Sett.projBins + mp_diff)

    # Counting and anchoring of data:
    if Sett.process_counts:
        lg.logprint(LAM_logger, 'Begin normalization of channels.', 'i')
        print('\n---Normalizing sample data---')
        # Get combined channel files of all samples
        countpaths = paths.datadir.glob('All_*')
        for path in countpaths:
            name = str(path.stem).split('_')[1]
            print('  {}  ...'.format(name))
            # Anchor sample's data to the full data matrix
            ch_counts = Normalize(path)
            ch_counts.starts, norm_counts = ch_counts.normalize_samples(mps, Store.totalLength, Store.center)
            # Get average bin counts
            ch_counts.averages(norm_counts)
            # Get averages of additional data per bin
            ch_counts.avg_add_data(paths, Sett.AddData, Store.totalLength)

        # Approximate width of sample
        if Sett.measure_width:
            print('  Width  ...')
            width_path = paths.datadir.joinpath('Sample_widths.csv')
            width_counts = Normalize(width_path)
            _, _ = width_counts.normalize_samples(mps * 2, Store.totalLength * 2, Store.center * 2,
                                                  name='Sample_widths_norm')
        lg.logprint(LAM_logger, 'Channels normalized.', 'i')


def project(paths):
    """Project features onto the vector."""
    lg.logprint(LAM_logger, 'Begin channel projection and counting.', 'i')
    print("\n---Projecting and counting channels---")
    # Loop through all directories in the root directory
    for path in [p for p in Sett.workdir.iterdir() if p.is_dir() and p.stem != 'Analysis Data']:
        # Initialize sample variables
        sample = GetSample(path, paths, process=False, projection=True)
        print(f"  {sample.name}  ...")
        # Find anchoring point of the sample
        sample.MP = sample.get_mps(Sett.MPname, Sett.useMP, paths.datadir)
        # Collection of data for each channel of the sample
        for path2 in [p for p in sample.channelpaths if Sett.MPname.lower() != str(p).split('_')[-2].lower()]:
            channel = GetChannel(path2, sample, Sett.AddData, paths.datadir)
            # If no variance in found additional data, it is discarded.
            if channel.datafail:
                datatypes = ', '.join(channel.datafail)
                info = "Invariant data discarded"
                msg = f"   -> {info} - {channel.name}: {datatypes}"
                print(msg)
            # Project features of channel onto vector
            sample.data = sample.project_channel(channel)
            if channel.name == Sett.vectChannel and Sett.measure_width:
                DefineWidths(sample.data, sample.vector, sample.sampledir, paths.datadir)
            # Count occurrences in each bin
            if channel.name not in ["MPs"]:
                sample.find_counts(channel.name, paths.datadir)
    lg.logprint(LAM_logger, 'All channels projected and counted.', 'i')


def relate_data(data, mp_bin=0, center=50, total_length=100):
    """Place sample data in context of all data, i.e. anchoring."""
    try:
        length = data.shape[0]
    except AttributeError:
        length = len(data)
    if np.isnan(mp_bin):
        msg = "Missing MP-projection(s). See 'Analysis Data/MPs.csv'."
        print(f"CRITICAL: {msg}")
        lg.logprint(LAM_logger, msg, 'c')
        raise SystemExit
    # Insert smaller input data into larger DF defined by TotalLength
    insx = int(center - mp_bin)
    end = int(insx + length)
    insert = np.full(total_length, np.nan)  # Bins outside input data are NaN
    data = np.where(data == np.nan, 0, data)  # Set all NaN in input to 0
    try:  # Insertion
        insert[insx:end] = data
    except ValueError:
        msg = "relate_data() call from {} line {}".format(inspect.stack()[1][1], inspect.stack()[1][2])
        print('ERROR: {}'.format(msg))
        lg.logprint(LAM_logger, f'Failed {msg}\n', 'ex')
        msg = "If not using MPs, remove MPs.csv from 'Data Files'."
        if insert[insx:end].size - length == mp_bin:
            lg.logprint(LAM_logger, msg, 'i')
        raise SystemExit
    return insert, insx


def vector_test(path):
    """Test that vector-files are found."""
    paths = [p for p in path.iterdir() if p.is_dir()]
    miss_vector = []
    for samplepath in paths:
        try:
            _ = next(samplepath.glob("Vector.*"))
        except StopIteration:
            miss_vector.append(samplepath.name)
            continue
    if not miss_vector:
        return
    raise VectorError(miss_vector)


def test_count_projection(counts, name):
    if (counts == 0).sum() > counts.size / 3:
        print("\n")
        print('WARNING: Uneven projection <- vector may be faulty!')
        print("\n")
        print('\a')
        lg.logprint(LAM_logger, f'Uneven projection for {name}. Check vector quality.', 'w')


def check_resize_step(resize, log=True):
    if Sett.SkeletonVector and Decimal(str(resize)) % Decimal(str(0.10)) != Decimal('0.0'):
        msg = 'Resizing not in step of 0.1'
        print("WARNING: {}".format(msg))
        # Round setting down to nearest 0.1.
        Sett.SkeletonResize = math.floor(resize*10) / 10
        msg2 = 'SkeletonResize changed to {}'.format(Sett.SkeletonResize)
        print("-> {}".format(msg2))
        if log:
            lg.logprint(LAM_logger, msg, 'w')
            lg.logprint(LAM_logger, msg2, 'i')


def define_scoring_point(line, s_x, s_y):
    points, point2 = line[-3:-2], line[-1]
    if not points:
        point1 = (s_x, s_y)
    elif len(points) == 1:
        point1 = points[0]
    else:
        p1, p2 = points[0], points[1]
        point1 = ((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)

    # Create a test point (used to score nearby pixels)
    shiftx = point2[0] - point1[0]  # shift in x for test point
    shifty = point2[1] - point1[1]  # shift in y for test point

    return gm.Point(s_x + shiftx, s_y + shifty)


def check_anchor_quality(sample_start):
    mean = np.mean(sample_start.astype('float'))
    if mean == 0:
        return
    std = np.std(sample_start.astype('float'))
    threshold = 2.5 * std
    outliers = sample_start[np.abs(sample_start - mean) >= threshold]
    if not outliers.empty:
        print(f"WARNING: Samples with outlying anchoring. Check anchoring and vector of:" +
              f"\n - {', '.join(outliers.index)}")