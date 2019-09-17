# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:42:28 2019

Script for longitudinal analysis of Drosophila midgut images. To run the script,
have the LAM.py file at a directory containing directories of each individual sample. 
The sample directories should be named followingly: "samplegroup_xyz_samplename", 
where xyz can be anything. Within the sample directories, cell positions and other
data should be in channel-specific directories, e.g. GFP or GFP+Pros, named as 
"xyz_channel_xyz". Within these channel directories the data has to be contained 
in csv's in similar format to statistics exported from Imaris. Most importantly 
"xyz_Position.csv" with column labels "Position X", "Position Y", and "ID". The 
cell ID should be the same between other data files, such as "xyz_Area.csv"

The script first creates a vector based on one channel ("vectChannel", typically 
DAPI), in order to approximate the midgut along its length. Positions on other 
channel can then be projected onto the vector, and cell numbers can be quantified 
along the midgut. If using a whole midgut for analysis, a measurement point (MP) "channel"
directory should also be created for the normalization of samples. This directory 
should contain position csv for one coordinate, typically in the middle of R3-region
so that the samples have a point to anchor for group-wise analysis.

Dependencies: Anaconda-included packages (Python 3.7), Shapely

@author: Arto Viitanen
"""
import system, analysis, process
from system import store
from settings import settings as Sett
import pathlib as pl

def main():
    PATHS = system.paths(Sett.workdir)
    # If sample processing set to True, collect data etc. Otherwise continue to
    # plotting and group-wise operations.
    if Sett.process_samples:
        # Loop Through samples and collect relevant data
        for path in [p for p in Sett.workdir.iterdir() if p.is_dir() and p.stem 
                     != 'Analysis Data']:
            sample = process.get_sample(path, PATHS.samplesdir, process=True)
            print("Processing {}  ...".format(sample.name))
            sample.vectData = sample.get_vectData(Sett.vectChannel)
            # Creation of vector for projection
            sample.vector = sample.create_vector(Sett.medianBins, PATHS.datadir, 
                                                 Sett.SkeletonVector, Sett.SkeletonResize, 
                                                 Sett.BDiter, Sett.SigmaGauss)
            # Finding measurement points for normalization between samples
            sample.MP, sample.secMP = sample.get_MPs(Sett.MPname, Sett.useMP, Sett.useSecMP, 
                                                     Sett.secMP, PATHS.datadir)
            # Collection of data for each channel
            for path2 in sample.channelpaths:
                channel = process.get_channel(path2, sample, Sett.AddData)
                sample.data = sample.project_channel(channel, PATHS.datadir)
                sample.find_counts(channel.name, PATHS.datadir)
    else:
        # When samples are not to be processed, the data is gathered from 
        # "./Analysis Data/Samples".
        print("\nGathering sample data  ...")
        # For each sample, the data is collected, and cell numbers are quantified
        # for each channel.
        for path in [p for p in PATHS.samplesdir.iterdir() if p.is_dir()]:
            sample = process.get_sample(path, PATHS.samplesdir, process=False)
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
                        
    # After all samples have been collected, find their respective MP bins and
    # normalize (anchor) cell count data. If MP's are not used, the samples are
    # anchored at bin == 0.
    print("\nNormalizing sample data ...")
    MPs = system.read_data(next(PATHS.datadir.glob("MPS.csv")), header = 0, test=False)
    # Find the smallest and largest bin-number of the dataset
    MPmax, MPmin = MPs.max(axis=1).item(), MPs.min(axis=1).item()
    # Find the size of needed dataframe, i.e. so that all anchored samples fit
    MPdiff = MPmax - MPmin
    store.totalLength = len(Sett.projBins) + MPdiff
    # Store the bin number of the row onto which samples are anchored to
    store.centerpoint = MPmax
    countpaths = PATHS.datadir.glob("All_*")
    for path in countpaths:
        # Aforementionad data is used to create dataframes onto which each sample's
        # MP is anchored to one row, with bin-respective (index) cell counts in 
        # each element of a sample (column) to allow relative comparison.
        ChCounts= process.normalize(path)            
        ChCounts.starts = ChCounts.normalize_samples(MPs, store.totalLength)
        ChCounts.Avg_AddData(PATHS, Sett.AddData, store.totalLength)
    # Storing of descriptive data of analysis, i.e. channels/samples/groups
    PATHS.save_AnalysisInfo(store.samples, store.samplegroups, store.channels)
    
    # After samples have been normalized, 
    ### TODO add plotting and group-wise operations
    SampleGrps = analysis.Samplegroups(store.samplegroups, PATHS)
    SampleGrps.create_plots()
#    for chanPath in SampleGrps._chanPaths:
#        # TODO create facetgrid for channel plots
#        continue
#    for addPath in SampleGrps._addData:
#        # TODO create facetgrid for addData plots
#        continue
    
    Grp = analysis.Group("Holidic")
#    print(Grp.groups)
#    print(Grp.name)
#    print(Grp2.addData)
    print('ANALYSIS COMPLETED')


if __name__ == '__main__':
    main()
