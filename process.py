# -*- coding: utf-8 -*-
from settings import settings
from plot import plotter
import system, math
from system import store
import pandas as pd, numpy as np, warnings, shapely.geometry as gm
import pathlib as pl
from scipy.ndimage import morphology as mp
from skimage.morphology import skeletonize
from skimage.filters import gaussian
import re
import logger
LAM_logger = logger.get_logger()

def Create_Samples(PATHS):
    logger.log_print(LAM_logger, 'Begin vector creation.', 'i')
    # Loop Through samples to create vectors
    print("---Processing samples---")
    for path in [p for p in settings.workdir.iterdir() if p.is_dir() and p.stem 
                 != 'Analysis Data']:
        sample = get_sample(path, PATHS)
        print("{}  ...".format(sample.name))
        sample.vectData = sample.get_vectData(settings.vectChannel)
        # Creation of vector for projection
        sample.create_vector(settings.medianBins, PATHS.datadir, 
                             settings.SkeletonVector, settings.SkeletonResize, 
                             settings.BDiter, settings.SigmaGauss)
    logger.log_print(LAM_logger, 'Vectors created.', 'i')

def Project(PATHS):
    logger.log_print(LAM_logger, 'Begin channel projection and counting.', 'i')
    print("\n---Projecting and count channels---")
    for path in [p for p in settings.workdir.iterdir() if p.is_dir() and p.stem 
             != 'Analysis Data']:
        sample = get_sample(path, PATHS, process=False, project=True)
        print("{}  ...".format(sample.name))
        sample.MP, sample.secMP = sample.get_MPs(settings.MPname, settings.useMP, 
                                                 settings.useSecMP, 
                                                 settings.secMP, PATHS.datadir)
        # Collection of data for each channel
        for path2 in [p for p in sample.channelpaths if settings.MPname \
                      != str(p).split('_')[-2]]:
            channel = get_channel(path2, sample, settings.AddData)
            sample.data = sample.project_channel(channel, PATHS.datadir)
            channelName = str(path2.stem)
            if channelName not in ["MPs"]:
                sample.find_counts(channel.name, PATHS.datadir)
    logger.log_print(LAM_logger, 'All channels projected and counted.', 'i')

def Get_Counts(PATHS):
    try:
        MPs = system.read_data(next(PATHS.datadir.glob('MPs.csv')), header=0, 
                               test=False)
    except:
        if not settings.useMP:
            MPs = pd.DataFrame(np.zeros((1, len(store.samples))), 
                            columns=store.samples)
            system.saveToFile(MPs, PATHS.datadir, 'MPs.csv', append=False)
        else:
            logger.log_print(LAM_logger, 'MPs.csv NOT found!', 'c')
            print("ERROR: 'MPs.csv' NOT found!")
    # Find the smallest and largest bin-number of the dataset
    MPmax, MPmin = MPs.max(axis=1).values[0], MPs.min(axis=1).values[0]
    # Store the bin number of the row onto which samples are anchored to
    store.center = MPmax
    # Find the size of needed dataframe, i.e. so that all anchored samples fit
    MPdiff = MPmax - MPmin
    if settings.process_counts == False and settings.process_samples == False:
        # Find all sample groups in the analysis from the found MPs.
        samples = MPs.columns.tolist()
        Groups = set({s.casefold(): s.split('_')[0] for s in samples}.values())
        store.samplegroups = sorted(Groups)
        try:
            temp = system.read_data(PATHS.datadir.joinpath("All_{}.csv".format(
                                settings.vectChannel)), test=False, header=0)
            store.totalLength = temp.shape[0]
        except AttributeError:
            msg = "Cannot determine length of sample matrix\
                    -> Must perform 'Count' before continuing."
            logger.log_print(LAM_logger, msg, 'c')
            print("ERROR: {}".format(msg))
            return            
        return
    else:
        store.totalLength = int(len(settings.projBins) + MPdiff)
    if settings.process_counts:
        logger.log_print(LAM_logger, 'Begin normalization of channels.', 'i')
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
            ChCounts.Avg_AddData(PATHS, settings.AddData, store.totalLength)
        logger.log_print(LAM_logger, 'Channels normalized.', 'i')


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
                logger.log_print(LAM_logger, msg, 'e')
                print('ERROR: {}'.format(msg))
            except AttributeError:
                msg = 'Faulty vector for {}'.format(self.name)
                logger.log_print(LAM_logger, msg, 'w')
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
            logger.log_print(LAM_logger, msg, 'w')
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
        vector = vector.simplify(settings.simplifyTol)
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

        buffer = 200 * resize
        coords = list(zip(X, Y))
        miny, maxy = Y.min(), Y.max()
        minx, maxx = X.min(), X.max()
        rminy, rmaxy = resize_minmax(miny, maxy, 'y', resize)
        rminx, rmaxx = resize_minmax(minx, maxx, 'x', resize)
        ylabels = np.arange(int(rminy) - buffer, int(rmaxy + (buffer + 1)), 10)
        xlabels = np.arange(int(rminx) - buffer, int(rmaxx + (buffer + 1)), 10)
        ylen, xlen = len(ylabels), len(xlabels)
        BA = pd.DataFrame(np.zeros((ylen, xlen)), index=np.flip(ylabels, 0), 
                          columns=xlabels)
        BAind, BAcol = BA.index, BA.columns
        for coord in coords:
            y = round(coord[1] * resize / 10) * 10
            x = round(coord[0] * resize / 10) * 10
            BA.at[(y, x)] = 1

        if BDiter > 0:
            struct1 = mp.generate_binary_structure(2, 2)
            try:
                BA = mp.binary_dilation(BA, structure=struct1, iterations=BDiter)
            except TypeError:
                msg = 'BDiter in settings has to be an integer.'
                logger.log_print(LAM_logger, msg, 'e')
                print("TypeError: {}".format(msg))
        if SigmaGauss > 0:
            BA = gaussian(BA, sigma=SigmaGauss)
            BA[BA > 0] = True
        BA = mp.binary_fill_holes(BA)
        skeleton = skeletonize(BA)
        skelDF = pd.DataFrame(skeleton, index=BAind, columns=BAcol)
        skelValues = [(skelDF.index[y], skelDF.columns[x]) for y, x in zip(
                *np.where(skelDF.values == True))]
        coordDF = pd.DataFrame(np.zeros((len(skelValues),2)),columns=['X', 'Y'])
        for i, coord in enumerate(skelValues):
            coordDF.loc[i, ['X', 'Y']] = [coord[1], coord[0]]
        finder = settings.find_dist
        line = []
        start = coordDF.loc[:, 'X'].idxmin()
        sx, sy = coordDF.loc[start, 'X'], coordDF.loc[start, 'Y']
        nearStart = coordDF[(abs(coordDF.loc[:, 'X'] - sx) <= finder/2) & 
                            (abs(coordDF.loc[:, 'Y'] - sy) <= finder/1)].index
        sx, sy = coordDF.loc[nearStart, 'X'].mean(), coordDF.loc[nearStart, 'Y'
                            ].mean()
        line.append((sx / resize, sy / resize))
        coordDF.drop(nearStart, inplace=True)
        flag = False
        while not flag:
            # TODO try creating median estimation?
            point = gm.Point(sx, sy)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                nearest = coordDF[(abs(coordDF.loc[:, 'X'] - sx) <= finder) & 
                                 (abs(coordDF.loc[:, 'Y'] - sy) <= finder)
                                 ].index
#            points = pd.Series(np.zeros(nearpixels.size), index=nearpixels)
#            for i, row in coordDF.loc[nearpixels, :].iterrows():
#                coord = row.loc['X':'Y'].tolist()
#                point2 = gm.Point(coord[0], coord[1])
#                dist = point.distance(point2)
#                points.at[i] = dist
#            minv = points.min()
#            nearest = points.where(points <= 2 * minv).dropna().index
#            if nearest.size == 1:
#                x2, y2 = coordDF.loc[nearest, 'X'], coordDF.loc[nearest, 'Y']
#                line.append((x2 / resize, y2 / resize))
#                sx, sy = x2, y2
#                coordDF.drop(nearest, inplace=True)
            if nearest.size > 0:
                try:
                    point1, point2 = line[-3], line[-1]
                except:
                    point1, point2 = (sx, sy), (sx + (50 / resize), sy)

                x1, y1 = point1[0], point1[1]
                x2, y2 = point2[0], point2[1]
                shiftx = x2 - x1
                shifty = y2 - y1
                x3, y3 = x2 + 2 * shiftx, y2 + 2 * shifty
                testpoint = gm.Point(x3, y3)
                distances = pd.DataFrame(np.zeros((nearest.size, 5)), 
                                         index=nearest, columns=['dist', 
                                         'distOG', 'score', 'X', 'Y'])
                for index, row in coordDF.loc[nearest, :].iterrows():
                    x4, y4 = coordDF.loc[index, 'X'], coordDF.loc[index, 'Y']
                    point3 = gm.Point(x4, y4)
                    dist = testpoint.distance(point3)
                    distOg = point.distance(point3)
                    score = (0.75 * distOg) + dist
                    distances.at[index, :] = [dist, distOg, score, x4, y4]
                nearest = distances.score.idxmin()
                x2, y2 = coordDF.loc[nearest, 'X'], coordDF.loc[nearest, 'Y']
                line.append((x2 / resize, y2 / resize))
                dropdist = distances.loc[nearest, 'distOG']
                dropdist2 = distances.loc[nearest, 'dist']
                # TODO alter dropping. Nearby, but only with smaller x than point?
#                huh = distances.at[nearest, "dist"]
#                print(distances)
#                print(huh, drop_dist * huh)
                forfeit = distances[(distances.dist >= dropdist) & 
                              (distances.distOG <= dropdist2)].dropna().index
#                print(type(nearest))
                forfeit.union([nearest])
                coordDF.drop(forfeit, inplace=True)
                sx, sy = x2, y2
#                print(coordDF)
            else:
                flag = True
            
#            print(sx, sy)
#            print(type(sy))
        try:
            vector = gm.LineString(line)
            vector = vector.simplify(settings.simplifyTol)
            linedf = pd.DataFrame(line, columns=['X', 'Y'])
            return vector, BA, skeleton, linedf
        except ValueError:
            msg = 'Faulty vector for {}'.format(self.name)
            logger.log_print(LAM_logger, msg, 'e')
            print("WARNING: Faulty vector. Try different settings.")
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
                rmv = next(s for s in self.channelpaths if str('_'+secMPname+'_')\
                           in str(s))
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
                logger.log_print(LAM_logger, msg, 'e')
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
                 settings.projBins, right=True)
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
                                 settings.projBins, right=True)
        # Save the obtained data:
        ChString = str('{}.csv'.format(channel.name))
        system.saveToFile(Positions, self.sampledir, ChString, append=False)
        return Positions

    def find_counts(self, channelName, datadir):
        counts = np.bincount(self.data['DistBin'], minlength=len(
                                                            settings.projBins))
        counts = pd.Series(np.nan_to_num(counts), name=self.name)
        ChString = str('All_{}.csv'.format(channelName))
        system.saveToFile(counts, datadir, ChString)


class get_channel:
    def __init__(self, path, sample, dataKeys):
        self.name = str(path.stem).split('_')[(-2)]
        self.path = path
        pospath = next(self.path.glob("*Position.csv"))
        self.data = self.read_channel(pospath)
        self.data = self.read_additional(dataKeys)

    def read_channel(self, path):
        data = system.read_data(str(path))
        channel = self.name
        if channel.lower() not in [c.lower() for c in store.channels]:
            store.channels.append(self.name)
        return data

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
                    if settings.replaceID:
                        try:
                            temp = settings.channelID.get(IDstring)
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
            grpData = NormCounts.loc[:,(NormCounts.columns.str.contains(namer))]
            Avgs.loc[:, "{}_All".format(grp)] = grpData.mean(axis=1)
        filename = str('ChanAvg_{}.csv'.format(self.channel))
        system.saveToFile(Avgs, self.path.parent, filename, append=False)
        

    def Avg_AddData(self, PATHS, dataNames, TotalLen):
        # TODO Add try / exceptions
        # TODO Make area and volume plots only for some channels + datafile naming !!!
        samples = self.starts.index
        for sample in samples:
            sampleDir = PATHS.samplesdir.joinpath(sample)
            dataFile = sampleDir.glob(str(self.channel + '.csv'))
#            try:
            data = system.read_data(next(dataFile), header=0)
            for dataType in dataNames.keys():
                sampleData = data.loc[:, data.columns.str.contains(str(dataType))]
                binnedData = data.loc[:, 'DistBin']
                bins = np.arange(1, len(settings.projBins) + 1)
                for col in sampleData:
                    avgS = pd.Series(np.full(TotalLen, np.nan), name=sample)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        insert = [np.nanmean(sampleData.loc[binnedData==i, 
                                                        col]) for i in bins]
                        insert = [0 if np.isnan(v) else v for v in insert]
                    strt = int(self.starts.at[sample])
                    end = int(strt + len(settings.projBins))
                    avgS[strt:end] = insert
                    filename = str('Avg_{}_{}.csv'.format(self.channel, col))
                    system.saveToFile(avgS, PATHS.datadir, filename)
#            except TypeError:
#                print("{}: {} not found on {}".format(self.channel, dataType, 
#                       sample))
#            except StopIteration:
#                pass

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
    insert[insx:end] = data
    return insert, insx
