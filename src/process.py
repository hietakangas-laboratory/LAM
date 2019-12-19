# -*- coding: utf-8 -*-
"""
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
from settings import settings as Sett
from plot import plotter
from system import store
import logger as lg, system
try:
    LAM_logger = lg.get_logger(__name__)
except AttributeError:
    print('Cannot get logger')

def Create_Samples(PATHS):
    lg.logprint(LAM_logger, 'Begin vector creation.', 'i')
    resize = Sett.SkeletonResize
    if Sett.SkeletonVector and dl.Decimal(str(resize)) \
                                % dl.Decimal(str(0.10)) != dl.Decimal('0.0'):
        msg = 'Resizing not in step of 0.1'
        print("WARNING: {}".format(msg))
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

def Project(PATHS):
    lg.logprint(LAM_logger, 'Begin channel projection and counting.', 'i')
    print("\n---Projecting and counting channels---")
    for path in [p for p in Sett.workdir.iterdir() if p.is_dir() and p.stem 
             != 'Analysis Data']:
        sample = get_sample(path, PATHS, process=False, project=True)
        print("{}  ...".format(sample.name))
        sample.MP, sample.secMP = sample.get_MPs(Sett.MPname, Sett.useMP, 
                                                 Sett.useSecMP, 
                                                 Sett.secMP, PATHS.datadir)
        # Collection of data for each channel
        for path2 in [p for p in sample.channelpaths if Sett.MPname \
                      != str(p).split('_')[-2]]:
            channel = get_channel(path2, sample, Sett.AddData, PATHS.datadir)
            sample.data = sample.project_channel(channel, PATHS.datadir)
            channelName = str(path2.stem)
            if channelName not in ["MPs"]:
                sample.find_counts(channel.name, PATHS.datadir)
    lg.logprint(LAM_logger, 'All channels projected and counted.', 'i')

def Get_Counts(PATHS):
#    if not Sett.process_counts:
    try:
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
        if len(FSamples) != len(samples): # Test whether sample numbers match
            msg = "Mismatch of sample N between MPs.csv and sample folders"
            print('WARNING: {}'.format(msg))
            lg.logprint(LAM_logger, msg, 'w')
        Groups = set({s.casefold(): s.split('_')[0] for s in samples}.values())
        store.samplegroups = sorted(Groups)
        try: # If required lengths of matrices haven't been defined because
            # Process and Count are both False, get the sizes from files.
            temp = system.read_data(PATHS.datadir.joinpath("Norm_{}.csv".format(
                                Sett.vectChannel)), test=False, header=0)
            store.totalLength = temp.shape[0] # Length of anchored matrices
            temp = system.read_data(PATHS.datadir.joinpath("All_{}.csv".format(
                                Sett.vectChannel)), test=False, header=0)
            store.binNum = temp.shape[0] # Length of sample matrix
        except AttributeError:
            msg = "Cannot determine length of sample matrix\n"+\
                    "-> Must perform 'Count' before continuing."
            lg.logprint(LAM_logger, msg, 'c')
            print("ERROR: {}".format(msg))         
        return
    else:
        store.totalLength = int(len(Sett.projBins) + MPdiff)
    if Sett.process_counts:
        lg.logprint(LAM_logger, 'Begin normalization of channels.', 'i')
        print('\n---Normalizing sample data---')
        countpaths = PATHS.datadir.glob('All_*')
        for path in countpaths:
            name = str(path.stem).split('_')[1]
            print('{}  ...'.format(name))
            # Aforementionad data is used to create dataframes onto which each 
            # sample's MP is anchored to one row, with bin-respective (index) 
            # cell counts in each element of a sample (column) to allow 
            # relative comparison.
            ChCounts = normalize(path)
            ChCounts.starts, NormCounts = ChCounts.normalize_samples(MPs, 
                                                             store.totalLength)
            ChCounts.averages(NormCounts)
            ChCounts.Avg_AddData(PATHS, Sett.AddData, store.totalLength)
        lg.logprint(LAM_logger, 'Channels normalized.', 'i')


class get_sample:
    def __init__(self, path, PATHS, process=True, project=False):
        self.name = str(path.stem)
        self.sampledir = PATHS.samplesdir.joinpath(self.name)
        self.group = self.name.split('_')[0]
        if self.name not in store.samples:
            store.samples.append(self.name)
        if self.group not in store.samplegroups:
            store.samplegroups.append(self.group)
            store.samplegroups = sorted(store.samplegroups)
        if self.sampledir.exists() == False:
            pl.Path.mkdir(self.sampledir)
        self.channelpaths = list([p for p in path.iterdir() if p.is_dir()])
        self.channels = [str(p).split('_')[(-2)] for p in self.channelpaths]
        if process == False and project == True:
        # If the samples are not to be processed, the vector data is gathered
        # from the csv-file in the sample's directory ("./Analysis Data/Samples/")
            for channel in self.channels:
                if channel.lower() not in [c.lower() for c in store.channels]:
                    store.channels.append(channel)
            try:
                tempVect = pd.read_csv(self.sampledir.joinpath('Vector.csv'))
                Vect = list(zip(tempVect.loc[:, 'X'], tempVect.loc[:, 'Y']))
                self.vector = gm.LineString(Vect)
                self.vectorLength = self.vector.length
                lenS = pd.Series(self.vectorLength, name=self.name)
                system.saveToFile(lenS, PATHS.datadir, 'Length.csv')
            except FileNotFoundError:
                msg = 'Vector.csv NOT found for {}'.format(self.name)
                lg.logprint(LAM_logger, msg, 'e')
                print('ERROR: {}'.format(msg))
            except AttributeError:
                msg = 'Faulty vector for {}'.format(self.name)
                lg.logprint(LAM_logger, msg, 'w')
                print('ERROR: Faulty vector for {}'.format(msg))

    def get_vectData(self, channel):
        try:
            namer = str("_{}_".format(channel))
            namerreg = re.compile(namer, re.I)
            dirPath = [self.channelpaths[i] for i, s in enumerate(
                            self.channelpaths) if namerreg.search(str(s))][0]
            vectPath = next(dirPath.glob('*Position.csv'))
            vectData = system.read_data(vectPath)
        except:
            msg = 'No valid file for vector creation.'
            lg.logprint(LAM_logger, msg, 'w')
            print('-> {}'.format(msg))
            vectData = None
        finally:
            return vectData

    def create_vector(self, creationBins, datadir, Skeletonize, resize, BDiter, 
                      SigmaGauss):
        """For creating the vector from the running median of the DAPI-positions.
        """
        positions = self.vectData
        X, Y = positions.loc[:, 'Position X'], positions.loc[:, 'Position Y']
        if Skeletonize:
            vector, binaryArray, skeleton, lineDF = self.SkeletonVector(X, Y, 
                                                    resize, BDiter, SigmaGauss)
            if vector is None:
                return
        else:
            vector, lineDF = self.MedianVector(X, Y, creationBins)
            binaryArray, skeleton = None, None
        vector = vector.simplify(Sett.simplifyTol)
        length = pd.Series(vector.length, name=self.name)
        system.saveToFile(length, datadir, 'Length.csv')
        system.saveToFile(lineDF, self.sampledir, 'Vector.csv', append=False)
        create_plot = plotter(self, self.sampledir)
        create_plot.vector(self.name, vector, X, Y, binaryArray, skeleton)

    def SkeletonVector(self, X, Y, resize, BDiter, SigmaGauss):
        def resize_minmax(minv, maxv, axis, resize):
            rminv = math.floor(minv * resize / 10) * 10
            rmaxv = math.ceil(maxv * resize / 10) * 10
            return rminv, rmaxv
        
        def _binarize():
            """Transformation of coordinates into binary image and subsequent
            dilation and smoothing."""
            buffer = 500 * resize
            ylbl = np.arange(int(rminy) - buffer, int(rmaxy + (buffer + 1)),10)
            xlbl = np.arange(int(rminx) - buffer, int(rmaxx + (buffer + 1)),10)
            BA = pd.DataFrame(np.zeros((len(ylbl), len(xlbl))), 
                          index=np.flip(ylbl, 0), columns=xlbl)
            BAind, BAcol = BA.index, BA.columns
            coords = list(zip(X, Y))
            for coord in coords: # Transform coords into binary array
                BA.at[(round(coord[1] * resize / 10) * 10, 
                       round(coord[0] * resize / 10) * 10)] = 1
            if BDiter > 0: # binary dilations
                struct1 = mp.generate_binary_structure(2, 2)
                try:
                    BA = mp.binary_dilation(BA, structure=struct1, 
                                            iterations=BDiter)
                except TypeError:
                    msg = 'BDiter in settings has to be an integer.'
                    lg.logprint(LAM_logger, msg, 'e')
                    print("TypeError: {}".format(msg))
            if SigmaGauss > 0: # Smoothing
                BA = gaussian(BA, sigma=SigmaGauss)
                BA[BA > 0] = True
            BA = mp.binary_fill_holes(BA)
            return BA, BAind, BAcol
        
        rminy, rmaxy = resize_minmax(Y.min(), Y.max(), 'y', resize)
        rminx, rmaxx = resize_minmax(X.min(), X.max(), 'x', resize)
        # Transform to binary
        binaryArray, BAindex, BAcols = _binarize()
        # Make skeleton and get coordinates of skeleton pixels
        skeleton = skeletonize(binaryArray)
        skelValues = [(BAindex[y], BAcols[x]) for y, x in zip(
                *np.where(skeleton == True))]
        # Dataframe from skeleton coords
        coordDF = pd.DataFrame(skelValues, columns=['Y', 'X'])
        # BEGIN CREATION OF VECTOR FROM SKELETON COORDS
        finder = Sett.find_dist # Variable for detection of nearby XY
        line = [] # For storing vector
        start = coordDF.X.idxmin()
        sx, sy = coordDF.loc[start, 'X'], coordDF.loc[start, 'Y']
        nearStart = coordDF[(abs(coordDF.X - sx) <= finder/3) & 
                            (abs(coordDF.Y - sy) <= finder)].index
        sx, sy = coordDF.loc[nearStart, 'X'].mean(), coordDF.loc[nearStart, 'Y'
                            ].mean()
        line.append((sx / resize, sy / resize))
        coordDF.drop(nearStart, inplace=True)
        flag = False
        while not flag:
            point = gm.Point(sx, sy)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                nearest = coordDF[(abs(coordDF.X - sx) <= finder) & 
                                 (abs(coordDF.Y - sy) <= finder)].index
            if nearest.size > 0:
                try:
                    point1, point2 = line[-3], line[-1]
                except:
                    point1, point2 = (sx, sy), (sx + (50*resize), sy)
                shiftx = point2[0] - point1[0]
                shifty = point2[1] - point1[1]
                testP = gm.Point(point2[0]+1.5*shiftx, point2[1]+1.5*shifty)
                distances = pd.DataFrame(np.zeros((nearest.size, 5)), 
                                         index=nearest, columns=['dist', 
                                         'distOG', 'score', 'X', 'Y'])
                for index, row in coordDF.loc[nearest, :].iterrows():
                    x, y = coordDF.X.at[index], coordDF.Y.at[index]
                    point3 = gm.Point(x, y)
                    dist = testP.distance(point3)
                    distOg = point.distance(point3)
                    score = (0.75 * distOg) + dist
                    distances.at[index, :] = [dist, distOg, score, x, y]
                best = distances.score.idxmin()
                x2, y2 = coordDF.X.at[best], coordDF.Y.at[best]
                line.append((x2 / resize, y2 / resize))
                forfeit = distances[(distances.dist >= distances.distOG)
                                    ].dropna().index
                coordDF.drop(forfeit, inplace=True)
                sx, sy = x2, y2
            else:
                flag = True
        try:
            vector = gm.LineString(line)
            vector = vector.simplify(Sett.simplifyTol)
            linedf = pd.DataFrame(line, columns=['X', 'Y'])
            return vector, binaryArray, skeleton, linedf
        except ValueError:
            msg = 'Faulty vector for {}'.format(self.name)
            lg.logprint(LAM_logger, msg, 'e')
            print("WARNING: Faulty vector. Try different Sett.")
            return None, None, None, None

    def MedianVector(self, X, Y, creationBins):
        bins = np.linspace(X.min(), X.max(), creationBins)
        idx = np.digitize(X, bins, right=True)
        Ymedian = np.zeros(creationBins)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            startval = np.nanmean(Y[(idx == 1)])
        Ymedian[0] = startval
        for b in range(1, creationBins):
            cells = Y[idx == b]
            if cells.size == 0:
                Ymedian[b] = Ymedian[b - 1]
            else:
                Ymedian[b] = Y[idx==b].min() + (Y[idx==b].max() - Y[idx==b
                       ].min())/2

        XYmedian = [p for p in tuple(np.stack((bins, Ymedian), axis=1)) if 
                    ~np.isnan(p).any()]
        vector = gm.LineString(XYmedian)
        linedf = pd.DataFrame(XYmedian, columns=['X', 'Y'])
        return vector, linedf

    def get_MPs(self, MPname, useMP, useSecMP, secMPname, datadir):
        MPdata, secMPdata = pd.DataFrame(), pd.DataFrame()
        if useSecMP: # If True, get secondary MP
            try:
                secMPdirpath = next(self.channelpaths.pop(i) for i, s in 
                                 enumerate(self.channelpaths) if 
                                 str('_'+secMPname+'_') in str(s))
                secMPpath = next(secMPdirpath.glob("*Position.csv"))
                secMPdata = system.read_data(secMPpath)
                self.secMPdata = secMPdata.loc[:,['Position X', 'Position Y']]
            except:
                print("-> Failed to find secondary MP positions")
        else: # If samplefolder contains unused secondary MP data, remove path
            secMPbin = None
            try:
                rmv = next(s for s in self.channelpaths if \
                           str('_'+secMPname+'_') in str(s))
                self.channelpaths.remove(rmv)  
            except: pass
        if useMP:
            try: # Get primary MP
                MPdirPath = next(self.channelpaths.pop(i) for i, s in enumerate(
                        self.channelpaths) if str('_'+MPname+'_') in str(s))
                MPpath = next(MPdirPath.glob("*Position.csv"))
                MPdata = system.read_data(MPpath)
                self.MPdata = MPdata.loc[:,['Position X', 'Position Y']]
            except:
                msg = 'could not find MP position for {}'.format(self.name)
                lg.logprint(LAM_logger, msg, 'e')
                print("-> Failed to find MP positions")
            finally:
                MPbin, secMPbin = None, None
                MPs = pd.DataFrame()
                if not self.MPdata.empty:
                    MPbin = self.project_MPs(self.MPdata, self.vector, datadir, 
                                             filename="MPs.csv")
                    MP = pd.Series(MPbin, name="MP")
                    MPs = pd.concat([MPs, MP], axis=1)
                if useSecMP and hasattr(self, "secMPdata"):
                    secMPbin = self.project_MPs(self.secMPdata, self.vector, 
                                                datadir, filename="secMPs.csv")
                    secMP = pd.Series(secMPbin, name = "secMP")
                    MPs = pd.concat([MPs, secMP], axis=1)
                MPs.to_csv(self.sampledir.joinpath("MPs.csv"), index=False)
        else: # Sets measurement point values to zero when MP's are not used
            MPbin = pd.Series(0, name = self.name)
            system.saveToFile(MPbin, datadir, "MPs.csv")
            system.saveToFile(MPbin, self.sampledir, "MPs.csv", append=False)
        return MPbin, secMPbin

    def project_MPs(self, Positions, vector, datadir, filename="some.csv"):
        """For the projection of spot coordinates onto the vector."""
        XYpos = list(zip(Positions['Position X'],Positions['Position Y']))
        # The shapely packages reguires transformation into Multipoints for the 
        # projection.
        points = gm.MultiPoint(XYpos) 
        # Find point of projection on the vector.
        Positions["VectPoint"] = [vector.interpolate(vector.project(gm.Point(x))) 
                                  for x in points]
        # Find normalized distance (0->1)
        Positions["NormDist"] = [vector.project(x, normalized=True) for x in
                 Positions["VectPoint"]]
        # Find the bins that the points fall into
        Positions["DistBin"]=np.digitize(Positions.loc[:,"NormDist"],
                 Sett.projBins, right=True)
        MPbin = pd.Series(Positions.loc[:,"DistBin"], name = self.name)
        # Save the obtained data:
        system.saveToFile(MPbin, datadir, filename)
        return MPbin
    
    def project_channel(self, channel, datadir):
        """For projecting coordinates onto the vector."""
        Positions = channel.data
        XYpos = list(zip(Positions['Position X'],Positions['Position Y']))
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
        Positions["DistBin"] = np.digitize(Positions.loc[:,"NormDist"],
                                 Sett.projBins, right=True)
        # Save the obtained data:
        ChString = str('{}.csv'.format(channel.name))
        system.saveToFile(Positions, self.sampledir, ChString, append=False)
        return Positions

    def find_counts(self, channelName, datadir):
        counts = np.bincount(self.data['DistBin'], minlength=len(Sett.projBins))
        counts = pd.Series(np.nan_to_num(counts), name=self.name)
        ChString = str('All_{}.csv'.format(channelName))
        system.saveToFile(counts, datadir, ChString)


class get_channel:
    def __init__(self, path, sample, dataKeys, datadir):
        self.datadir = datadir
        self.name = str(path.stem).split('_')[-2]
        self.path = path
        pospath = next(self.path.glob("*Position.csv"))
        self.data = self.read_channel(pospath)
        self.data = self.read_additional(dataKeys)
        if 'ClusterID' in self.data.columns:
            store.clusterPaths.append(self.path)

    def read_channel(self, path):
        try:
            data = system.read_data(str(path))
            channel = self.name
            if channel.lower() not in [c.lower() for c in store.channels]:
                store.channels.append(self.name)
            return data
        except ValueError:
            lg.logprint(LAM_logger,'Cannot read channel path {}'.format(path),'ex')

    def read_additional(self, dataKeys):
        newData = self.data
        for key in dataKeys:
            fstring = dataKeys.get(key)[0]
            finder = str('*{}*'.format(fstring))
            paths = list(self.path.glob(finder))
            for path in paths:
                addData = system.read_data(str(path))
                addData = addData.loc[:, [key, 'ID']]
                if len(paths) > 1:
                    # If multiple files found, search identifier from filename
                    strings = str(path.stem).split(fstring)
                    IDstring = strings[1].split('_')[1]
                    if Sett.replaceID:
                        try:
                            temp = Sett.channelID.get(IDstring)
                            if temp != None:
                                IDstring = temp
                        except: pass
                    rename = str(key + '-' + IDstring)
                    addData.rename(columns={key: rename}, inplace=True)
                newData = pd.merge(newData, addData, on='ID')
        return newData


class normalize:
    def __init__(self, path):
        self.path = pl.Path(path)
        self.channel = str(self.path.stem).split('_')[1]
        self.counts = system.read_data(path, header=0, test=False)

    def normalize_samples(self, MPs, arrayLength):
        """ For inserting sample data into larger matrix, centered with MP."""
        cols = self.counts.columns
        data = pd.DataFrame(np.zeros((arrayLength, len(cols))), columns=cols)
        SampleStart = pd.Series(np.full(len(cols), np.nan), index=cols)
        for col in self.counts.columns:
            handle = self.counts.loc[:, col].values
            mp = MPs.loc[0, col]
            insert, insx = relate_data(handle, mp, store.center, arrayLength)
            data[col] = insert
            SampleStart.at[col] = insx
        filename = str('Norm_{}.csv'.format(self.channel))
        data = data.sort_index(axis=1)
        system.saveToFile(data, self.path.parent, filename, append=False)
        return SampleStart, data
    
    def averages(self, NormCounts):
        samples = NormCounts.columns.tolist()
        Groups = set({s.casefold(): s.split('_')[0] for s in samples}.values())
        cols = ["{}_All".format(g) for g in Groups]
        Avgs = pd.DataFrame(index=NormCounts.index, columns=cols)
        for grp in Groups:
            namer = "{}_".format(grp)
            grpData = NormCounts.loc[:,(NormCounts.columns.str.startswith(namer))]
            Avgs.loc[:, "{}_All".format(grp)] = grpData.mean(axis=1)
        filename = str('ChanAvg_{}.csv'.format(self.channel))
        system.saveToFile(Avgs, self.path.parent, filename, append=False)
        

    def Avg_AddData(self, PATHS, dataNames, TotalLen):
    # TODO Make area and volume plots only for some channels + datafile naming !!!
        samples = self.starts.index
        for sample in samples:
            sampleDir = PATHS.samplesdir.joinpath(sample)
            dataFile = sampleDir.glob(str(self.channel + '.csv'))
            data = system.read_data(next(dataFile), header=0)
            for dataType in dataNames.keys():
                sampleData = data.loc[:,data.columns.str.contains(str(dataType))]
                binnedData = data.loc[:, 'DistBin']
                bins = np.arange(1, len(Sett.projBins) + 1)
                for col in sampleData:
                    avgS = pd.Series(np.full(TotalLen, np.nan), name=sample)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        insert = [np.nanmean(sampleData.loc[binnedData==i, 
                                                        col]) for i in bins]
                        insert = [0 if np.isnan(v) else v for v in insert]
                    strt = int(self.starts.at[sample])
                    end = int(strt + len(Sett.projBins))
                    avgS[strt:end] = insert
                    filename = str('Avg_{}_{}.csv'.format(self.channel, col))
                    system.saveToFile(avgS, PATHS.datadir, filename)

def relate_data(data, MP=0, center=50, TotalLength=100):
    """Sets the passed data into the context of all samples, i.e. places the
    data into an empty array with the exact length required to fit all 
    samples"""
    try:
        length = data.shape[0]
    except:
        length = len(data)
    insx = int(center - MP)
    end = int(insx + length)
    insert = np.full(TotalLength, np.nan)            
    data = np.where(data==np.nan, 0, data)
    try:
        insert[insx:end] = data
    except ValueError:
        msg = "relate_data() call from {} line {}".format(inspect.stack()[1][1], 
                                                       inspect.stack()[1][2])
        print('ERROR: {}'.format(msg))
        lg.logprint(LAM_logger, 'Failed {}\n'.format(msg), 'ex')
        msg="If not using MPs, remove MPs.csv from'./Analysis Data/Data Files/'"
        if insert[insx:end].size - length == MP:
            lg.logprint(LAM_logger, msg, 'i')
        raise SystemExit
    return insert, insx
