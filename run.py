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
import system, analysis
from system import store as store
from settings import settings as Sett

def main():
    PATHS = system.paths(Sett.workdir)
    # If sample processing set to True, collect data etc.
    if Sett.process_samples:
        # Loop Through samples and collect relevant data
        for path in [p for p in Sett.workdir.iterdir() if p.is_dir() and p.stem 
                     != 'Analysis Data']:
            sample = analysis.get_sample(path, PATHS.samplesdir)
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
                channel = analysis.get_channel(path2, sample, Sett.AddData)
                sample.data = sample.project_channel(channel, PATHS.datadir)
                sample.find_counts(channel.name, PATHS.datadir)
        # After all samples have been collected, find their respective MP bins
        # and go through each channel
        print("\nNormalizing sampledata ...")
        MPs = system.read_data(next(PATHS.datadir.glob("MPS.csv")), header = 0)
        MPmax, MPmin = MPs.max(axis=1).item(), MPs.min(axis=1).item()
        MPdiff = MPmax - MPmin
        store.totalLength = len(Sett.projBins) + MPdiff
        store.centerpoint = MPmax
        countpaths = PATHS.datadir.glob("All_*")
        for path in countpaths:
            # Aforementionad data is used to create arrays onto which each sample's
            # MP is anchored to the same row, allowing relative comparison
            channelcounts = analysis.normalize(path)            
            channelcounts.normalize_samples(MPs, store.totalLength)

    print('ANALYSIS COMPLETED')


if __name__ == '__main__':
    main()
