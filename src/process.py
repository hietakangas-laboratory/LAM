# -*- coding: utf-8 -*-
"""
LAM-module for vector creation, and data collection and projection.

Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen

"""
# Standard libraries
import decimal as dl
import inspect
import math
import re
import warnings
# Other packages
import numpy as np
import pandas as pd
import pathlib as pl
from scipy.ndimage import morphology as mp
import shapely.geometry as gm
from skimage.morphology import skeletonize
from skimage.filters import gaussian
# LAM modules
from settings import store as store, settings as Sett
from plot import plotter
import logger as lg
import system
try:
    LAM_logger = lg.get_logger(__name__)
except AttributeError:
    print('Cannot get logger')


def Create_Samples(PATHS):
    """Create vectors for the samples."""
    lg.logprint(LAM_logger, 'Begin vector creation.', 'i')
    # Test that resize-setting is in step of 0.1:
    resize = Sett.SkeletonResize
    if Sett.SkeletonVector and dl.Decimal(str(resize)) % dl.Decimal(str(0.10))\
            != dl.Decimal('0.0'):
        msg = 'Resizing not in step of 0.1'
        print("WARNING: {}".format(msg))
        # Round setting down to nearest 0.1.
        Sett.SkeletonResize = math.floor(resize*10) / 10
        msg2 = 'SkeletonResize changed to {}'.format(Sett.SkeletonResize)
        print("-> {}".format(msg2))
        lg.logprint(LAM_logger, msg, 'w')
        lg.logprint(LAM_logger, msg2, 'i')
    # Loop Through samples to create vectors
    print("---Processing samples---")
    for path in [p for p in Sett.workdir.iterdir() if p.is_dir() and p.stem
                 != 'Analysis Data']:
        sample = get_sample(path, PATHS)
        print("{}  ...".format(sample.name))
        sample.vectData = sample.get_vectData(Sett.vectChannel)
        # Creation of vector for projection
        sample.create_vector(Sett.medianBins, PATHS.datadir,
                             Sett.SkeletonVector, Sett.SkeletonResize,
                             Sett.BDiter, Sett.SigmaGauss)
    lg.logprint(LAM_logger, 'Vectors created.', 'i')
    
    
def vector_test(path):
    paths = [p for p in path.iterdir() if (p.is_dir and p.glob("vector.csv"))]
    vectors = [p.name for p in paths for v in p.iterdir() if v.name == "vector.csv"]
    if len(vectors) == 0:
        raise AssertionError


def Project(PATHS):
    """Project features onto the vector."""
    lg.logprint(LAM_logger, 'Begin channel projection and counting.', 'i')
    print("\n---Projecting and counting channels---")
    # Loop through all directories in the root directory
    for path in [p for p in Sett.workdir.iterdir() if p.is_dir() and p.stem
                 != 'Analysis Data']:
        # Initialize sample variables
        sample = get_sample(path, PATHS, process=False, project=True)
        print("{}  ...".format(sample.name))
        # Find anchoring point of the sample
        sample.MP = sample.get_MPs(Sett.MPname, Sett.useMP, PATHS.datadir)
        # Collection of data for each channel of the sample
        for path2 in [p for p in sample.channelpaths if Sett.MPname
                      != str(p).split('_')[-2]]:
            channel = get_channel(path2, sample, Sett.AddData, PATHS.datadir)
            # If no variance in found additional data, it is discarded.
            if channel.datafail:
                datatypes = ', '.join(channel.datafail)
                info = "No variance, data discarded"
                msg = "-> {} - {}: {}".format(info, channel.name, datatypes)
                print(msg)
            # Project features of channel onto vector
            sample.data = sample.project_channel(channel)
            channelName = str(path2.stem)
            # Count occurrences in each bin
            if channelName not in ["MPs"]:
                sample.find_counts(channel.name, PATHS.datadir)
    lg.logprint(LAM_logger, 'All channels projected and counted.', 'i')


def Get_Counts(PATHS):
    """Handle data to anchor samples and find cell counts."""
    try:  # Test that MPs are found for the sample
        MPs = system.read_data(next(PATHS.datadir.glob('MPs.csv')),
                               header=0, test=False)
    except FileNotFoundError:
        msg = "MPs.csv NOT found!"
        print("ERROR: {}".format(msg))
        lg.logprint(LAM_logger, msg, 'c')
        msg = "-> Perform 'Count' before continuing.\n"
        print("{}".format(msg))
        lg.logprint(LAM_logger, msg, 'i')
        raise SystemExit
    # Find the smallest and largest bin-number of the dataset
    MPmax, MPmin = MPs.max(axis=1).values[0], MPs.min(axis=1).values[0]
    # Store the bin number of the row onto which samples are anchored to
    store.center = MPmax
    # Find the size of needed dataframe, i.e. so that all anchored samples fit
    MPdiff = MPmax - MPmin
    if not any([Sett.process_counts, Sett.process_samples]):
        # Find all sample groups in the analysis from the found MPs.
        FSamples = [p for p in PATHS.samplesdir.iterdir() if p.is_dir()]
        samples = MPs.columns.tolist()
        if len(FSamples) != len(samples):  # Test whether sample numbers match
            msg = "Mismatch of sample N between MPs.csv and sample folders"
            print('WARNING: {}'.format(msg))
            lg.logprint(LAM_logger, msg, 'w')
        Groups = set({s.casefold(): s.split('_')[0] for s in samples}.values())
        store.samplegroups = sorted(Groups)
        store.channels = [c.stem.split('_')[1] for c in
                          PATHS.datadir.glob("All_*.csv")]
        try:  # If required lengths of matrices haven't been defined because
            # Process and Count are both False, get the sizes from files.
            chan = Sett.vectChannel
            path = PATHS.datadir.joinpath("Norm_{}.csv".format(chan))
            temp = system.read_data(path, test=False, header=0)
            store.totalLength = temp.shape[0]  # Length of anchored matrices
            path = PATHS.datadir.joinpath("All_{}.csv".format(chan))
            temp = system.read_data(path, test=False, header=0)
            store.binNum = temp.shape[0]  # Length of sample matrix
        except AttributeError:
            msg = "Cannot determine length of sample matrix\n" +\
                    "-> Must perform 'Count' before continuing."
            lg.logprint(LAM_logger, msg, 'c')
            print("ERROR: {}".format(msg))
        return
    # The total length of needed matrix when using 'Count'
    store.totalLength = int(len(Sett.projBins) + MPdiff)
    if Sett.process_counts:  # Begin anchoring of data
        lg.logprint(LAM_logger, 'Begin normalization of channels.', 'i')
        print('\n---Normalizing sample data---')
        # Get combined channel files of all samples
        countpaths = PATHS.datadir.glob('All_*')
        for path in countpaths:
            name = str(path.stem).split('_')[1]
            print('{}  ...'.format(name))
            # Aforementionad data is used to create dataframes onto which each
            # sample's MP is anchored to one row, with bin-respective (index)
            # cell counts in each element of a sample (column) to allow
            # relative comparison.
            ChCounts = normalize(path)
            ChCounts.starts, NormCounts = ChCounts.normalize_samples(
                MPs, store.totalLength)
            ChCounts.averages(NormCounts)
            ChCounts.Avg_AddData(PATHS, Sett.AddData, store.totalLength)
        lg.logprint(LAM_logger, 'Channels normalized.', 'i')


class get_sample:
    """Collect sample data and process for analysis."""

    def __init__(self, path, PATHS, process=True, project=False):
        self.name = str(path.stem)
        self.sampledir = PATHS.samplesdir.joinpath(self.name)
        self.group = self.name.split('_')[0]
        # Add sample and group to storing variables
        if self.name not in store.samples:
            store.samples.append(self.name)
            store.samples = sorted(store.samples)
        if self.group not in store.samplegroups:
            store.samplegroups.append(self.group)
            store.samplegroups = sorted(store.samplegroups)
        # Make folder for storing data and find data-containing files
        if self.sampledir.exists() is False:
            pl.Path.mkdir(self.sampledir)
        self.channelpaths = list([p for p in path.iterdir() if p.is_dir()])
        self.channels = [str(p).split('_')[(-2)] for p in self.channelpaths]
        self.vectData = None
        self.MP = None
        self.data = None
        if process is False and project is True:
            # If the samples are not to be processed, the vector data is
            # gathered from the csv-file in the sample's directory
            # ("./Analysis Data/Samples/")
            for channel in self.channels:  # Store all found channel names
                if channel.lower() not in [c.lower() for c in store.channels]:
                    store.channels.append(channel)
            try:  # Find sample's vector file and read it
                tempVect = pd.read_csv(self.sampledir.joinpath('Vector.csv'))
                Vect = list(zip(tempVect.loc[:, 'X'], tempVect.loc[:, 'Y']))
                self.vector = gm.LineString(Vect)
                self.vectorLength = self.vector.length
                lenS = pd.Series(self.vectorLength, name=self.name)
                system.saveToFile(lenS, PATHS.datadir, 'Length.csv')
            except FileNotFoundError:  # If vector file not found
                msg = 'Vector.csv NOT found for {}'.format(self.name)
                lg.logprint(LAM_logger, msg, 'e')
                print('ERROR: {}'.format(msg))
            except AttributeError:  # If vector file is faulty
                msg = 'Faulty vector for {}'.format(self.name)
                lg.logprint(LAM_logger, msg, 'w')
                print('ERROR: Faulty vector for {}'.format(msg))

    def get_vectData(self, channel):
        """Get channel data that is used for vector creation."""
        try:
            namer = str("_{}_".format(channel))
            namerreg = re.compile(namer, re.I)
            dirPath = [self.channelpaths[i] for i, s in
                       enumerate(self.channelpaths) 
                       if namerreg.search(str(s))][0]
            vectPath = next(dirPath.glob('*Position.csv'))
            vectData = system.read_data(vectPath)
        except FileNotFoundError:
            msg = 'No valid file for vector creation.'
            lg.logprint(LAM_logger, msg, 'w')
            print('-> {}'.format(msg))
            vectData = None
        return vectData

    def create_vector(self, creationBins, datadir, Skeletonize, resize, BDiter,
                      SigmaGauss):
        """Handle data for vector creation."""
        # Extract point coordinates of the vector:
        positions = self.vectData
        X, Y = positions.loc[:, 'Position X'], positions.loc[:, 'Position Y']
        if Skeletonize:  # Create skeleton vector
            vector, binaryArray, skeleton, lineDF = self.SkeletonVector(
                X, Y, resize, BDiter, SigmaGauss)
            if vector is None:
                return
        else:  # Alternatively create median vector
            vector, lineDF = self.MedianVector(X, Y, creationBins)
            binaryArray, skeleton = None, None
        # Simplification of vector points
        vector = vector.simplify(Sett.simplifyTol)
        # Save total length of vector
        length = pd.Series(vector.length, name=self.name)
        system.saveToFile(length, datadir, 'Length.csv')
        # Save vector file
        system.saveToFile(lineDF, self.sampledir, 'Vector.csv', append=False)
        # Create plots of created vector
        create_plot = plotter(self, self.sampledir)
        create_plot.vector(self.name, vector, X, Y, binaryArray, skeleton)

    def SkeletonVector(self, X, Y, resize, BDiter, SigmaGauss):
        """Create vector by skeletonization of image-transformed positions."""
        def resize_minmax(minv, maxv):
            """
            Rounds x- and y-coordinates for binarization.
            
            Takes min and max values found in feature coordinates and rounds
            them appropriately to provide indexes for the smaller, resized
            binary array. The new indexes allow changing array values from
            zeroes to ones based on feature coords.
            """
            rminv = math.floor(minv * resize / 10) * 10
            rmaxv = math.ceil(maxv * resize / 10) * 10
            return rminv, rmaxv

        def _binarize():
            """Transform XY into binary image and perform editing."""
            # Create DF indices (X&Y-coords) with a buffer for operations:
            buffer = 500 * resize
            ylbl = np.arange(int(rminy - buffer),
                             int(rmaxy + (buffer + 1)), 10)
            xlbl = np.arange(int(rminx - buffer),
                             int(rmaxx + (buffer + 1)), 10)
            # Create binary array with the created indices
            BA = pd.DataFrame(np.zeros((len(ylbl), len(xlbl))),
                              index=np.flip(ylbl, 0), columns=xlbl)
            for coord in coords:  # Transform coords into binary array
                y = round(coord[1] * resize / 10) * 10
                x = (round(coord[0] * resize / 10) * 10)
                BA.at[y, x] = 1
            if BDiter > 0:  # binary dilations, if used
                struct1 = mp.generate_binary_structure(2, 2)
                try:
                    BA = mp.binary_dilation(BA, structure=struct1,
                                            iterations=BDiter)
                except TypeError:
                    msg = 'BDiter in settings has to be an integer.'
                    lg.logprint(LAM_logger, msg, 'e')
                    print("TypeError: {}".format(msg))
            if SigmaGauss > 0:  # Gaussian smoothing
                BA = gaussian(BA, sigma=SigmaGauss)
                BA[BA > 0] = True  # Re-binarization
            # Fill holes in the sample, e.g. ruptured locations with no micro-
            # scopy features.
            BA = mp.binary_fill_holes(BA)
            return BA, list(ylbl), list(xlbl)

        coords = list(zip(X, Y))
        rminy, rmaxy = resize_minmax(Y.min(), Y.max())
        rminx, rmaxx = resize_minmax(X.min(), X.max())
        # Transform to binary
        binaryArray, BAindex, BAcols = _binarize()
        # Make skeleton and get coordinates of skeleton pixels
        skeleton = skeletonize(binaryArray)
        skelValues = [(BAindex[y], BAcols[x]) for y, x in zip(
            *np.where(skeleton == 1))]
        # Dataframe from skeleton coords
        coordDF = pd.DataFrame(skelValues, columns=['Y', 'X'])
        # BEGIN CREATION OF VECTOR FROM SKELETON COORDS
        finder = Sett.find_dist  # Distance for detection of nearby XY
        line = []  # For storing vector
        start = coordDF.X.idxmin()  # Start from smallest x-coord
        sx, sy = coordDF.loc[start, 'X'], coordDF.loc[start, 'Y']
        div = 5
        flag = False
        # Determining starting point of vector from as near to the end of
        # sample as possible:
        while not flag:
            nearStart = coordDF[(abs(coordDF.X - sx) <= finder/div) &
                                (abs(coordDF.Y - sy) <= finder)].index
            if nearStart.size < 3:
                div -= 0.1
            else:
                flag = True
        # Take mean coordinates of cells near the end to be the starting point
        sx, sy = coordDF.loc[nearStart, 'X'].mean(), coordDF.loc[
            nearStart, 'Y'].mean()
        line.append((sx / resize, sy / resize))
        # Drop the used coordinates
        coordDF.drop(nearStart, inplace=True)
        # Continue finding next points until flagged ready:
        flag = False
        while not flag:
            point = gm.Point(sx, sy)
            # Find pixels near to the current coordinate
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                nearest = coordDF[(abs(coordDF.X - sx) <= finder) &
                                  (abs(coordDF.Y - sy) <= finder)].index
            if nearest.size == 0:  # If none near, extend search distance once
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=UserWarning)
                    nearest = coordDF[(abs(coordDF.X - sx) <= finder * 2) &
                                      (abs(coordDF.Y - sy) <= finder * 2)
                                      ].index
            if nearest.size > 0:
                # If pixels are found, establish the vector's direction:
                try:
                    point1, point2 = line[-3], line[-1]
                except IndexError:
                    point1, point2 = (sx, sy), (sx + (50*resize), sy)
                # Create a test point (used to score nearby pixels)
                shiftx = point2[0] - point1[0]  # shift in x for test
                shifty = point2[1] - point1[1]  # shift in y for test
                testP = gm.Point(point2[0]+1.25*shiftx, point2[1]+shifty)
                # DataFrame for storing relevant info on pixel coordinates
                distances = pd.DataFrame(np.zeros((nearest.size, 5)),
                                         index=nearest,
                                         columns=['dist', 'distOG', 'penalty',
                                                  'X', 'Y'])
                # Create scores for each nearby pixel:
                for index, __ in coordDF.loc[nearest, :].iterrows():
                    x, y = coordDF.X.at[index], coordDF.Y.at[index]
                    point3 = gm.Point(x, y)
                    dist = testP.distance(point3)  # distance to a testpoint
                    distOg = point.distance(point3)  # dist to current coord
                    penalty = (0.75 * distOg) + dist
                    distances.loc[index, :] = [dist, distOg, penalty, x, y]
                # Find the pixel with the smallest penalty and add to vector:
                best = distances.penalty.idxmin()
                x2, y2 = coordDF.X.at[best], coordDF.Y.at[best]
                line.append((x2 / resize, y2 / resize))
                # Drop the pixels that are behind current vector coord
                forfeit = distances[(distances.dist >= distances.distOG)
                                    ].dropna().index
                best = pd.Index([best], dtype='int64')
                forfeit = forfeit.append(best)
                coordDF.drop(forfeit, inplace=True)
                # Set current location for the next loop
                sx, sy = x2, y2
            else:  # If none are found nearby after widening search, finished
                flag = True
        try:  # Create LineString-object from finished vector
            vector = gm.LineString(line)
            linedf = pd.DataFrame(line, columns=['X', 'Y'])
            return vector, binaryArray, skeleton, linedf
        except ValueError:  # If something went wrong with creation, warn
            msg = 'Faulty vector for {}'.format(self.name)
            lg.logprint(LAM_logger, msg, 'e')
            print("WARNING: Faulty vector. Try different settings")
            return None, None, None, None

    def MedianVector(self, X, Y, creationBins):
        """Create vector by calculating median coordinates."""
        # Divide sample to equidistant points between min & max X-coord:
        bins = np.linspace(X.min(), X.max(), creationBins)
        idx = np.digitize(X, bins, right=True)
        Ymedian = np.zeros(creationBins)
        # Find median Y-coord at first bin:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            startval = np.nanmean(Y[(idx == 1)])
        Ymedian[0] = startval
        # Then for the rest of the bins
        for b in range(1, creationBins):
            cells = Y[idx == b]
            if cells.size == 0:  # If no cells at bin, copy previous Y-coord
                Ymedian[b] = Ymedian[b - 1]
            else:
                Ymedian[b] = Y[idx == b].min() + (Y[idx == b].max() -
                                                  Y[idx == b].min()) / 2
        # Make into XY-coordinates
        XYmedian = [p for p in tuple(np.stack((bins, Ymedian), axis=1)) if
                    ~np.isnan(p).any()]
        # Create LineString-object from finished vector
        vector = gm.LineString(XYmedian)
        linedf = pd.DataFrame(XYmedian, columns=['X', 'Y'])
        return vector, linedf

    def get_MPs(self, MPname, useMP, datadir):
        """Collect MPs for sample anchoring."""
        if useMP:
            try:  # Get primary MP
                MPdirPath = next(self.channelpaths.pop(i) for i, s in
                                 enumerate(self.channelpaths) if
                                 str('_' + MPname + '_') in str(s))
                MPpath = next(MPdirPath.glob("*Position.csv"))
                MPdata = system.read_data(MPpath)
                MPdata = MPdata.loc[:, ['Position X', 'Position Y']]
            except (StopIteration, ValueError):
                msg = 'could not find MP position for {}'.format(self.name)
                lg.logprint(LAM_logger, msg, 'e')
                print("-> Failed to find MP positions")
            finally:
                MPbin = None
                MPs = pd.DataFrame()
                if not MPdata.empty:
                    MPbin = self.project_MPs(MPdata, self.vector, datadir,
                                             filename="MPs.csv")
                    MPs = pd.concat([MPs, pd.Series(MPbin.iat[0], name='MP')])
                    # MP = pd.Series(MPbin, name="MP")
                    # MPs = pd.concat([MPs, MP], axis=1)
                # MPs.to_csv(self.sampledir.joinpath("MPs.csv"), index=False)
        else:  # Sets measurement point values to zero when MP's are not used
            MPbin = pd.Series(0, name=self.name)
            system.saveToFile(MPbin, datadir, "MPs.csv")
            system.saveToFile(MPbin, self.sampledir, "MPs.csv", append=False)
        return MPbin

    def project_MPs(self, Positions, vector, datadir, filename="some.csv"):
        """For the projection of spot coordinates onto the vector."""
        XYpos = list(zip(Positions['Position X'], Positions['Position Y']))
        # The shapely packages reguires transformation into Multipoints for the
        # projection.
        points = gm.MultiPoint(XYpos)
        # Find point of projection on the vector.
        Positions["VectPoint"] = [vector.interpolate(
            vector.project(gm.Point(x))) for x in points]
        # Find normalized distance (0->1)
        Positions["NormDist"] = [vector.project(x, normalized=True) for x in
                                 Positions["VectPoint"]]
        # Find the bins that the points fall into
        Positions["DistBin"] = np.digitize(Positions.loc[:, "NormDist"],
                                           Sett.projBins, right=True)
        MPbin = pd.Series(Positions.loc[:, "DistBin"], name=self.name)
        # Save the obtained data:
        system.saveToFile(MPbin, datadir, filename)
        return MPbin

    def project_channel(self, channel):
        """For projecting coordinates onto the vector."""
        Positions = channel.data
        XYpos = list(zip(Positions['Position X'], Positions['Position Y']))
        # The shapely packages reguires transformation into Multipoints for the
        # projection.
        points = gm.MultiPoint(XYpos)
        # Find point of projection on the vector.
        Positions["VectPoint"] = [self.vector.interpolate(self.vector.project(
                                gm.Point(x))) for x in points]
        # Find normalized distance (0->1)
        Positions["NormDist"] = [self.vector.project(x, normalized=True) for x
                                 in Positions["VectPoint"]]
        # Find the bins that the points fall into
        Positions["DistBin"] = np.digitize(Positions.loc[:, "NormDist"],
                                           Sett.projBins, right=True)
        # Save the obtained data:
        ChString = str('{}.csv'.format(channel.name))
        system.saveToFile(Positions, self.sampledir, ChString, append=False)
        return Positions

    def find_counts(self, channelName, datadir):
        """Gather projected features and find bin counts."""
        counts = np.bincount(self.data['DistBin'],
                             minlength=len(Sett.projBins))
        counts = pd.Series(np.nan_to_num(counts), name=self.name)
        ChString = str('All_{}.csv'.format(channelName))
        system.saveToFile(counts, datadir, ChString)


class get_channel:
    """Find and read channel data plus additional data."""

    def __init__(self, path, sample, dataKeys, datadir):
        self.sample = sample
        self.datafail = []
        self.datadir = datadir
        self.name = str(path.stem).split('_')[-2]
        self.path = path
        self.pospath = next(self.path.glob("*Position.csv"))
        self.data = self.read_channel(self.pospath)
        self.read_additional(dataKeys)
        if 'ClusterID' in self.data.columns:
            store.clusterPaths.append(self.path)

    def read_channel(self, path):
        """Read channel data into a dataframe."""
        try:
            data = system.read_data(str(path))
            channel = self.name
            if channel.lower() not in [c.lower() for c in store.channels]:
                store.channels.append(self.name)
            return data
        except ValueError:
            lg.logprint(LAM_logger, 'Cannot read channel path {}'.format(path),
                        'ex')

    def read_additional(self, dataKeys):
        """Read relevant additional data of channel."""
        def _testVariance(data):
            """Test if additional data column contains any variance."""
            for col in data.columns:
                if data.loc[:, col].nunique() == 1:
                    data.loc[:, col] = np.nan
                    self.datafail.append(col)

        def _rename_ID():
            """Rename filename identification of channel."""
            # I.e. as defined by settings.channelID
            rename = None
            strings = str(col).split('_')
            if len(strings) > 1:
                IDstring = strings[-1]
                key = '_'.join(strings[:-1])
                if Sett.replaceID:
                    temp = Sett.channelID.get(IDstring)
                    if temp is not None:
                        IDstring = temp
                rename = str(key + '-' + IDstring)
            return rename

        for key in dataKeys:
            fstring = dataKeys.get(key)[0]
            finder = str('*{}*'.format(fstring))
            paths = list(self.path.glob(finder))
            addData = pd.DataFrame(self.data.loc[:, 'ID'])
            if not paths:
                print("-> {} {} file not found".format(self.name, key))
                continue
            elif len(paths) == 1:
                namer = re.compile('^{}'.format(key), re.I)
                if (paths[0] == self.pospath and
                        any(self.data.columns.str.contains(namer))):
                    continue
                elif (paths[0] == self.pospath and
                      not any(self.data.columns.str.contains(namer))):
                    print("'{}' not in AddData-file of {} on channel {}"
                          .format(key, self.sample.name, self.name))
                tmpData = system.read_data(str(paths[0]))
                cols = tmpData.columns.map(lambda x, namer=namer: bool(
                    re.match(namer, x)) or x == 'ID')
                tmpData = tmpData.loc[:, cols]
                addData = pd.merge(addData, tmpData, on='ID')
            else:  # If multiple files, e.g. intensity, get all
                for path in paths:
                    # Search identifier for column from filename
                    strings = str(path.stem).split(fstring)
                    IDstring = strings[1].split('_')[1]
                    # Locate columns
                    tmpData = system.read_data(str(path))
                    tmpData = tmpData.loc[:, [key, 'ID']]
                    for col in [c for c in tmpData.columns if c != 'ID']:
                        rename = str(col + '_' + IDstring)
                        tmpData.rename(columns={key: rename}, inplace=True)
                    addData = pd.merge(addData, tmpData, on='ID')
            # Go through all columns and drop invariant data
            for col in [c for c in addData.columns if c != 'ID']:
                _testVariance(tmpData)
                rename = _rename_ID()  # Rename columns if wanted
                if rename is not None:
                    addData.rename(columns={col: rename}, inplace=True)
            self.data = pd.merge(self.data, addData, on='ID')


class normalize:
    """Anchor sample data into dataframe with all samples."""

    def __init__(self, path):
        self.path = pl.Path(path)
        self.channel = str(self.path.stem).split('_')[1]
        self.counts = system.read_data(path, header=0, test=False)
        self.starts = None

    def normalize_samples(self, MPs, arrayLength):
        """For inserting sample data into larger matrix, centered with MP."""
        cols = self.counts.columns
        data = pd.DataFrame(np.zeros((arrayLength, len(cols))), columns=cols)
        # Create empty series for holding each sample's starting index
        SampleStart = pd.Series(np.full(len(cols), np.nan), index=cols)
        for col in self.counts.columns:
            handle = self.counts.loc[:, col].values
            MP = MPs.loc[0, col]
            # Insert sample's count data into larger, anchored dataframe:
            insert, insx = relate_data(handle, MP, store.center, arrayLength)
            data[col] = insert
            # Save starting index of the sample
            SampleStart.at[col] = insx
        # Save anchored data
        filename = str('Norm_{}.csv'.format(self.channel))
        data = data.sort_index(axis=1)
        system.saveToFile(data, self.path.parent, filename, append=False)
        return SampleStart, data

    def averages(self, NormCounts):
        """Find bin averages of channels."""
        samples = NormCounts.columns.tolist()
        Groups = set({s.casefold(): s.split('_')[0] for s in samples}.values())
        cols = ["{}_All".format(g) for g in Groups]
        Avgs = pd.DataFrame(index=NormCounts.index, columns=cols)
        for grp in Groups:
            namer = "{}_".format(grp)
            grpData = NormCounts.loc[:, NormCounts.columns.str.startswith(
                namer)]
            # Calculate group averages
            Avgs.loc[:, "{}_All".format(grp)] = grpData.mean(axis=1)
        # Save average data
        filename = str('ChanAvg_{}.csv'.format(self.channel))
        system.saveToFile(Avgs, self.path.parent, filename, append=False)

    def Avg_AddData(self, PATHS, dataNames, TotalLen):
        """Find bin averages of additional data."""
        samples = self.starts.index
        for sample in samples:
            sampleDir = PATHS.samplesdir.joinpath(sample)
            dataFile = sampleDir.glob(str(self.channel + '.csv'))
            data = system.read_data(next(dataFile), header=0)
            for dataType in dataNames.keys():
                sampleData = data.loc[:, data.columns.str.contains(
                    str(dataType))]
                binnedData = data.loc[:, 'DistBin']
                bins = np.arange(1, len(Sett.projBins) + 1)
                for col in sampleData:
                    avgS = pd.Series(np.full(TotalLen, np.nan), name=sample)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',
                                              category=RuntimeWarning)
                        insert = [np.nanmean(sampleData.loc[
                            binnedData == i, col]) for i in bins]
                        insert = [0 if np.isnan(v) else v for v in insert]
                    strt = int(self.starts.at[sample])
                    end = int(strt + len(Sett.projBins))
                    avgS[strt:end] = insert
                    filename = str('Avg_{}_{}.csv'.format(self.channel, col))
                    system.saveToFile(avgS, PATHS.datadir, filename)


def relate_data(data, MP=0, center=50, TotalLength=100):
    """Place sample data in context of all data, i.e. anchoring."""
    try:
        length = data.shape[0]
    except AttributeError:
        length = len(data)
    # Insert smaller input data into larger DF defined by TotalLength
    insx = int(center - MP)
    end = int(insx + length)
    insert = np.full(TotalLength, np.nan)  # Bins outside input data are NaN
    data = np.where(data == np.nan, 0, data)  # Set all NaN in input to 0
    try:  # Insertion
        insert[insx:end] = data
    except ValueError:
        msg = "relate_data() call from {} line {}".format(
            inspect.stack()[1][1], inspect.stack()[1][2])
        print('ERROR: {}'.format(msg))
        lg.logprint(LAM_logger, 'Failed {}\n'.format(msg), 'ex')
        msg = "If not using MPs, remove MPs.csv from 'Data Files'."
        if insert[insx:end].size - length == MP:
            lg.logprint(LAM_logger, msg, 'i')
        raise SystemExit
    return insert, insx
