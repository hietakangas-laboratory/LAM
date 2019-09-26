# -*- coding: utf-8 -*-
from settings import settings
from plot import plotter
import system
from system import store as store
import pandas as pd
import numpy as np
import warnings
import shapely.geometry as gm
import pathlib as pl
import math
from scipy.ndimage import morphology as mp
from skimage.morphology import skeletonize
from skimage.filters import gaussian
import re

def Create_Samples(PATHS):
    # Loop Through samples and collect relevant data
    for path in [p for p in settings.workdir.iterdir() if p.is_dir() and p.stem 
                 != 'Analysis Data']:
        sample = get_sample(path, PATHS.samplesdir, process=True)
        print("Processing {}  ...".format(sample.name))
        sample.vectData = sample.get_vectData(settings.vectChannel)
        # Creation of vector for projection
        sample.vector = sample.create_vector(settings.medianBins, PATHS.datadir, 
                                             settings.SkeletonVector, settings.SkeletonResize, 
                                             settings.BDiter, settings.SigmaGauss)
        # Finding measurement points for normalization between samples
        sample.MP, sample.secMP = sample.get_MPs(settings.MPname, settings.useMP, settings.useSecMP, 
                                                 settings.secMP, PATHS.datadir)
        # Collection of data for each channel
        for path2 in sample.channelpaths:
            channel = get_channel(path2, sample, settings.AddData)
            sample.data = sample.project_channel(channel, PATHS.datadir)
            sample.find_counts(channel.name, PATHS.datadir)
    print("\nAll samples processed")

def Gather_Samples(PATHS):
# When samples are not to be processed, the data is gathered from 
    # "./Analysis Data/Samples".
    print("\nGathering sample data  ...")
    # For each sample, the data is collected, and cell numbers are quantified
    # for each channel.
    for path in [p for p in PATHS.samplesdir.iterdir() if p.is_dir()]:
        sample = get_sample(path, PATHS.samplesdir, process=False)
        # Looping through every channel found in the sample's directory
        for channelPath in sample.channelpaths:
            channelName = str(channelPath.stem)
            if channelName != "MPs": # Collecting microscopy channel relevant data
                sample.data = system.read_data(channelPath, header = 0)
                sample.find_counts(channelName, PATHS.datadir)
            else: # Collecting measurement point data for anchoring of samples
                if hasattr(sample, "MP"):
                    system.saveToFile(sample.MP.rename(sample.name), 
                                      PATHS.datadir, "MPs.csv")
                if hasattr(sample, "secMP"):
                    system.saveToFile(sample.secMP.rename(sample.name), 
                                      PATHS.datadir, "secMPs.csv")

def Get_Counts(PATHS):
    MPs = system.read_data(next(PATHS.datadir.glob("MPS.csv")), header = 0, test=False)
    # Find the smallest and largest bin-number of the dataset
    MPmax, MPmin = MPs.max(axis=1).item(), MPs.min(axis=1).item()
    # Find the size of needed dataframe, i.e. so that all anchored samples fit
    MPdiff = MPmax - MPmin
    store.totalLength = len(settings.projBins) + MPdiff
    # Store the bin number of the row onto which samples are anchored to
    store.centerpoint = MPmax
    if settings.process_counts == False and settings.process_samples == False:
        return
    print("\nCounting and normalizing sample data ...")
    countpaths = PATHS.datadir.glob("All_*")
    for path in countpaths:
        name = str(path.stem).split('_')[1]
        print("{}  ...".format(name))
        # Aforementionad data is used to create dataframes onto which each sample's
        # MP is anchored to one row, with bin-respective (index) cell counts in 
        # each element of a sample (column) to allow relative comparison.
        ChCounts = normalize(path)            
        ChCounts.starts = ChCounts.normalize_samples(MPs, store.totalLength)
        ChCounts.Avg_AddData(PATHS, settings.AddData, store.totalLength)

class get_sample:   
    def __init__(self, path, samplesdir, process=True):
        self.name = str(path.stem)
        self.sampledir = samplesdir.joinpath(self.name)
        self.group = self.name.split('_')[0]
        if self.name not in store.samples:
            store.samples.append(self.name)
        if self.group not in store.samplegroups:
            store.samplegroups.append(self.group)
        if self.sampledir.exists() == False:
            pl.Path.mkdir(self.sampledir)
        if process == True:
            self.channelpaths = list([p for p in path.iterdir() if p.is_dir()])
            self.channels = [str(p).split('_')[(-2)] for p in self.channelpaths]
        else: # If the samples are not to be processed, the data is only gathered
              # from the csv-files in the sample's directory ("./Analysis Data/Samples/")
            self.channelpaths = list([p for p in path.iterdir() if ".csv" in p.name and 
                                      "Vector.csv" not in p.name])
            self.channels = [p.stem for p in self.channelpaths]
            for channel in self.channels:
                if channel.lower() not in [c.lower() for c in store.channels]:
                    store.channels.append(channel)
            tempVect = pd.read_csv(self.sampledir.joinpath("Vector.csv"))
            Vect = list(zip(tempVect.loc[:,"X"], tempVect.loc[:,"Y"]))
            self.vector = gm.LineString(Vect)
            try:
                MPs = pd.read_csv(self.sampledir.joinpath("MPs.csv"))
                if settings.useMP:
                    self.MP = MPs.loc[:, "MP"]
                try:
                    if settings.useSecMP:
                        self.secMP = MPs.loc[:, "secMP"]
                except KeyError:
                    pass
            except FileNotFoundError:
                print("MPs.csv NOT found for sample {}!".format(self.name))
            except KeyError:
                print("Measurement point for sample {} NOT found in MPs.csv!".format(self.name))
        
    def get_vectData(self, channel):
        try:
            namer = str('_'+channel+'_')
            namerreg = re.compile(namer, re.I)
            dirPath = [self.channelpaths[i] for i, s in enumerate(self.channelpaths) \
                       if namerreg.search(str(s))][0]
            vectPath = next(dirPath.glob("*_Position.csv"))
            vectData = system.read_data(vectPath)
        except:
            print("Sample {} has no valid file for vector creation".format(self.name))
            vectData = None
        finally:
            return vectData
    
    def create_vector(self, creationBins, datadir, Skeletonize, resize, BDiter, SigmaGauss):
        """For creating the vector from the running median of the DAPI-positions."""
        positions = self.vectData
        X, Y = positions.loc[:,'Position X'], positions.loc[:,'Position Y']
        if Skeletonize:
            vector, binaryArray, skeleton, lineDF = self.SkeletonVector(X, Y, resize, BDiter, SigmaGauss)            
        else:
            vector, lineDF = self.MedianVector(X, Y, creationBins)
            binaryArray, skeleton = None, None
        # The next ones are for storing the length of the vector, which should be appr. the real length of the midgut/sub-region
        length = pd.Series(vector.length, name=self.name)
        system.saveToFile(length,datadir,"Length.csv")
        system.saveToFile(lineDF, self.sampledir, "Vector.csv", append=False)
        create_plot = plotter(self, self.sampledir)
        create_plot.vector(self.name, vector, X, Y, binaryArray, skeleton)
        return vector   
    
    def SkeletonVector(self, X, Y, resize, BDiter, SigmaGauss):
        def resize_minmax(minv, maxv, axis, resize):
            rminv =  math.floor((minv * resize) / 10) * 10
            rmaxv =  math.ceil((maxv * resize) / 10) * 10
            return rminv, rmaxv
        
        buffer = 100 * resize
        coords = list(zip(X,Y))  
        
        miny, maxy = Y.min(), Y.max()
        minx, maxx = X.min(), X.max()
        rminy, rmaxy = resize_minmax(miny, maxy, "y", resize)
        rminx, rmaxx = resize_minmax(minx, maxx, "x", resize)
        
        ylabels = np.arange(int(rminy)-buffer, int(rmaxy+(buffer+1)), 10)
        xlabels = np.arange(int(rminx)-buffer, int(rmaxx+(buffer+1)), 10)
        ylen, xlen = len(ylabels), len(xlabels)
        BA = pd.DataFrame(np.zeros((ylen,xlen)),index=np.flip(ylabels,0),columns=xlabels)
        BAind, BAcol = BA.index, BA.columns
        for coord in coords:
            y = round((coord[1] * resize) / 10) * 10
            x = round((coord[0] * resize) / 10) * 10
            BA.at[y, x] = 1
        if BDiter > 0:
            struct1 = mp.generate_binary_structure(2, 2)
            BA = mp.binary_dilation(BA, structure=struct1, iterations=BDiter)
        if SigmaGauss > 0:
            BA = gaussian(BA, sigma=SigmaGauss)
            BA[BA > 0] = True
        BA = mp.binary_fill_holes(BA)
        
        skeleton = skeletonize(BA)
        skelDF = pd.DataFrame(skeleton, index=BAind, columns=BAcol)
        skelValues = [(skelDF.index[y], skelDF.columns[x]) for y, x in zip(*np.where(skelDF.values == True))]
        coordDF = pd.DataFrame(np.zeros((len(skelValues),2)), columns=["X", "Y"])
        for i, coord in enumerate(skelValues):
            coordDF.loc[i, ["X","Y"]] = [coord[1], coord[0]]
        ### FINDING NEAREST POINTS
        line = []
        start = coordDF.loc[:,"X"].idxmin()
        sx, sy = coordDF.loc[start,"X"], coordDF.loc[start,"Y"]
        line.append((sx/resize, sy/resize))
        coordDF.drop(start, inplace = True)        
        flag = False
        finder = 200 * resize
        while not flag:
            point = gm.Point(sx,sy)
            nearpixels = coordDF[(abs(coordDF.loc[:,"X"] - sx) <= finder) & 
                                (abs(coordDF.loc[:,"Y"] - sy) <= finder)]
            points = pd.Series(np.zeros(nearpixels.shape[0]), index = nearpixels.index)
            for i, row in nearpixels.iterrows():
                coord = row.loc['X':'Y'].tolist()
                point2 = gm.Point(coord[0], coord[1])
                dist = point.distance(point2)
                points.at[i] = dist
            minv = points.min()
            nearest = points.where(points <= (1.5 * minv)).dropna().index
            if nearest.size == 1:
                x2, y2 = coordDF.loc[nearest,"X"].item(), coordDF.loc[nearest,"Y"].item()
                line.append((x2/resize, y2/resize))
                coordDF.drop(nearest, inplace = True)
            elif nearest.size > 1:
                try: point1, point2 = line[-3], line[-1]
                except: point1, point2 = (sx,sy), (sx+(resize * 100),sy)
                x1, y1 = point1[0], point1[1]
                x2, y2 = point2[0], point2[1]
                shiftx = x2 - x1
                shifty = y2 - y1
                x3, y3 = x2+(2*shiftx), y2+(2*shifty)
                testpoint = gm.Point(x3,y3)
                distances = pd.DataFrame(np.zeros((nearest.size, 3)), index = nearest, columns = ["dist","X","Y"])
                for index, row in coordDF.loc[nearest,:].iterrows():
                    x4, y4 = coordDF.loc[index,"X"].item(), coordDF.loc[index,"Y"].item()
                    point3 = gm.Point(x4,y4)
                    dist = testpoint.distance(point3)
                    distances.at[index, :] = [dist, x4, y4]
                nearest = distances.dist.idxmin()
                x2, y2 = coordDF.loc[nearest,"X"].item(), coordDF.loc[nearest,"Y"].item()
                forfeit = distances.where(distances.X < x2).dropna().index
                line.append((x2/resize, y2/resize))
                coordDF.drop(forfeit, inplace = True)
            else:
                flag = True
            sx, sy = x2, y2
        vector = gm.LineString(line)
        vector = vector.simplify(settings.simplifyTol)
        linedf = pd.DataFrame(line, columns =['X', 'Y']) 
        return vector, BA, skeleton, linedf
           
    def MedianVector(self, X, Y, creationBins):
        # Find the minimal and maximum values of X in the cell positions, and 
        # divide their separation into number of bins equal to creationBins.
        bins = np.linspace(X.min(),X.max(), creationBins)
        idx  = np.digitize(X,bins, right=True)
        Ymedian = np.zeros(creationBins)
        # Determines the median for the first bin from cells belonging to the 
        # second bin, because the first bin typically contains no cells.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            startval = np.nanmean(Y[idx==1])
        Ymedian[0] = startval
        # Loops and finds medians for rest of the bins.
        for b in range(1,creationBins):
            cells = Y[idx==b]
            if cells.size == 0: # In case of an empty bin, value is copied from the earlier bin.
                Ymedian[b] = Ymedian[b-1]       
            else: # Otherwise median is the median coordinate between highest and lowest y-value in bin.
                Ymedian[b] = (Y[idx==b].min()+((Y[idx==b].max()-Y[idx==b].min())/2))
        # Make coordinate pairs from y-medians and creation bin x-coordinates.
        XYmedian = [p for p in tuple(np.stack((bins, Ymedian), axis=1)) if 
                    ~np.isnan(p).any()]
        vector = gm.LineString(XYmedian) # Create vector from the XY-coordinates
        linedf = pd.DataFrame(XYmedian, columns =['X', 'Y']) 
        return vector, linedf
        
        
    def get_MPs(self, MPname, useMP, useSecMP, secMPname, datadir):
        MPdata, secMPdata = pd.DataFrame(), pd.DataFrame()
        if useSecMP: # If True, get secondary MP
            try:
                secMPdirpath = next(self.channelpaths.pop(i) for i, s in 
                                 enumerate(self.channelpaths) if str('_'+secMPname+'_') in str(s))
                secMPpath = next(secMPdirpath.glob("*_Position.csv"))
                secMPdata = system.read_data(secMPpath)
                self.secMPdata = secMPdata.loc[:,['Position X', 'Position Y']]
            except:
                print("Failed to find secondary MP positions for sample {}".format(self.name))
        else: # If samplefolder contains unused secondary MP data, remove path
            secMPbin = None
            try:
                rmv = next(s for s in self.channelpaths if str('_'+secMPname+'_') in str(s))
                self.channelpaths.remove(rmv)  
            except: pass
        if useMP:
            try: # Get primary MP
                MPdirPath = next(self.channelpaths.pop(i) for i, s in enumerate(self.channelpaths) \
                                  if str('_'+MPname+'_') in str(s))
                MPpath = next(MPdirPath.glob("*_Position.csv"))
                MPdata = system.read_data(MPpath)
                self.MPdata = MPdata.loc[:,['Position X', 'Position Y']]
            except:
                print("Failed to find MP positions for sample {}".format(self.name))
            finally:
                MPbin, secMPbin = None, None
                MPs = pd.DataFrame()
                if not self.MPdata.empty:
                    MPbin = self.project_MPs(self.MPdata, self.vector, datadir, 
                                             filename="MPs.csv")
                    MP = pd.Series(MPbin, name = "MP")
                    MPs = pd.concat([MPs, MP], axis=1)
                if useSecMP and hasattr(self, "secMPdata"):
                    secMPbin = self.project_MPs(self.secMPdata, self.vector, datadir,
                                                 filename="secMPs.csv")
                    secMP = pd.Series(secMPbin, name = "secMP")
                    MPs = pd.concat([MPs, secMP], axis=1)
                MPs.to_csv(self.sampledir.joinpath("MPs.csv"), index=False)
        else: MPbin = 0        
        return MPbin, secMPbin
        
    def project_MPs(self, Positions, vector, datadir, filename="some.csv"):
        """For projecting coordinates onto the vector."""
        XYpos = list(zip(Positions['Position X'],Positions['Position Y']))
        points = gm.MultiPoint(XYpos) # The shapely packages reguires transformation into Multipoints for the projection.
        Positions["VectPoint"] = [vector.interpolate(vector.project(gm.Point(x))) # Find point of projection on the vector.
                                  for x in points]
        Positions["NormDist"] = [vector.project(x, normalized=True) for x in # Find normalized distance (0->1)
                 Positions["VectPoint"]]
        # Find the bins that the points fall into
        Positions["DistBin"]=np.digitize(Positions.loc[:,"NormDist"],settings.projBins, right=True)
        MPbin = pd.Series(Positions.loc[:,"DistBin"], name = self.name)
        # Save the obtained data:
        system.saveToFile(MPbin, datadir, filename)
        return MPbin
    
    def project_channel(self, channel, datadir):
        """For projecting coordinates onto the vector."""
        Positions = channel.data
        XYpos = list(zip(Positions['Position X'],Positions['Position Y']))
        points = gm.MultiPoint(XYpos) # The shapely packages reguires transformation into Multipoints for the projection.
        Positions["VectPoint"] = [self.vector.interpolate(self.vector.project(gm.Point(x))) # Find point of projection on the vector.
                                  for x in points]
        Positions["NormDist"] = [self.vector.project(x, normalized=True) for x in # Find normalized distance (0->1)
                 Positions["VectPoint"]]
        # Find the bins that the points fall into
        Positions["DistBin"]=np.digitize(Positions.loc[:,"NormDist"],settings.projBins, right=True)
        # Save the obtained data:
        ChString = str('{}.csv'.format(channel.name))
        system.saveToFile(Positions, self.sampledir, ChString, append=False)
        return Positions
    
    def find_counts(self, channelName, datadir):
        counts = np.bincount(self.data["DistBin"],minlength=len(settings.projBins))
        counts = pd.Series(np.nan_to_num(counts), name = self.name)
        ChString = str('All_{}.csv'.format(channelName))
        system.saveToFile(counts, datadir, ChString)

class get_channel:
    def __init__(self, path, sample, dataKeys):
        self.name = str(path.stem).split('_')[(-2)]
        self.path = path
        namer = str('*{}_Position*'.format(self.name))
        pospath = next(self.path.glob(namer))
        self.data = self.read_channel(pospath)
        self.data = self.read_additional(dataKeys)

    def read_channel(self, path):
        data = system.read_data(str(path))
        channel = self.name
        if channel.lower() not in [c.lower() for c in store.channels]:
            store.channels.append(self.name)
        return data
    
    def read_additional(self, dataKeys):
        # Loop through pre-defined keys and find respective data files, e.g. Area data
        newData = self.data
        for key in dataKeys:
            fstring = dataKeys.get(key)[0]
            finder = str("*{}*".format(fstring))
            paths = list(self.path.glob(finder))
            for path in paths:
                addData = system.read_data(str(path))
                addData = addData.loc[:, [key, "ID"]]
                if len(paths) > 1:
                    # If multiple files found, search identifier from filename
                    strings = str(path.stem).split(fstring)
                    IDstring = strings[1].split('_')[1]
                    if settings.replaceID:
                        try: 
                            temp = settings.channelID.get(IDstring)
                            IDstring = temp
                        except: pass
                    rename = str(key + '_' + IDstring)
                    addData.rename(columns={key: rename}, inplace=True)
                newData = pd.merge(newData, addData, on="ID")
        return newData

class normalize:
    def __init__(self, path):
        self.path = pl.Path(path)
        self.channel = str(self.path.stem).split('_')[1]
        self.counts = system.read_data(path, header = 0, test=False)
        
    def normalize_samples(self, MPs, arrayLength):
        """ For inserting sample data into larger matrix, centered with MP."""
        cols = self.counts.columns
        data = pd.DataFrame(np.zeros((arrayLength, len(cols))), columns = cols)
        SampleStart = pd.Series(np.full(len(cols), np.nan), index = cols)
        for col in self.counts.columns:
            handle = self.counts.loc[:, col].values
            mp = MPs.loc[0,col]
            # Creation of dataframe insert
            insert, insx = relate_data(handle,mp,store.centerpoint,arrayLength)
            data[col] = insert
            SampleStart.at[col] = insx
        filename = str("Norm_{}.csv".format(self.channel))
        data = data.sort_index(axis=1)
        system.saveToFile(data, self.path.parent, filename, append = False)
        return SampleStart
    
    def Avg_AddData(self, PATHS, dataNames, TotalLen):
        # TODO Add try / exceptions
        samples = self.starts.index
        for sample in samples:
            sampleDir = PATHS.samplesdir.joinpath(sample)
            dataFile = sampleDir.glob(str(self.channel + ".csv"))
#            try:
            data = system.read_data(next(dataFile), header = 0)
            for dataType in dataNames.keys():
                sampleData = data.loc[:,data.columns.str.contains(str(dataType))]
                binnedData = data.loc[:, "DistBin"]
                bins = np.arange(1, len(settings.projBins) + 1)
                for col in sampleData:
                    avgS = pd.Series(np.full(TotalLen, np.nan), name = sample)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        insert = [np.nanmean(sampleData.loc[binnedData==i, 
                                    col]) for i in bins]
                        insert = [0 if np.isnan(v) else v for v in insert]
                    strt = int(self.starts.at[sample])
                    end = int(strt + len(settings.projBins))
                    avgS[strt:end] = insert
                    filename = str("Avg_{}.csv".format(col))
                    system.saveToFile(avgS, PATHS.datadir, filename)
#            except TypeError:
#                print("{}: {} not found on {}".format(self.channel, dataType, sample))
#            except StopIteration:
#                pass
                    
def relate_data(data, MP=0, center=50, TotalLength=100, start = False):
    """Sets the passed data into the context of all samples, ie. places the
    data into an empty array with the exact length required to fit all 
    samples"""
    try: length = data.shape[0]
    except: length = len(data)
    insx = int(center - MP)
    end = int(insx+length)
    insert = np.full(TotalLength, np.nan)
    insert[insx:end] = data
    return insert, insx
            
        