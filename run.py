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

Dependencies: Anaconda-included packages (Python 3.7), Shapely, pycg3d

@author: Arto Viitanen
"""
import system, analysis, process
from system import store
from settings import settings

def main():
    systemPaths = system.start()
    # If sample processing set to True, collect data etc. Otherwise continue to
    # plotting and group-wise operations.
    if settings.process_samples:
        process.Create_Samples(systemPaths)
    else:
        process.Gather_Samples(systemPaths)                        
    # After all samples have been collected/created, find their respective MP bins and
    # normalize (anchor) cell count data. If MP's are not used, the samples are
    # anchored at bin == 0.
    process.Get_Counts(systemPaths)
    # Storing of descriptive data of analysis, i.e. channels/samples/groups
    systemPaths.save_AnalysisInfo(store.samples, store.samplegroups, store.channels)
    
    # After samples have been counted and normalized
    SampleGroups = analysis.Samplegroups(store.samplegroups, store.channels,
                                        store.totalLength, store.center, systemPaths)
    # Computing cell numbers for each sample's each bin
    if settings.process_counts:
        SampleGroups.Get_Totals()
    # Finding of nearest cells and distances
    if settings.Find_Distances and settings.process_dists:
        SampleGroups.Get_DistanceMean()
    # Finding clustered cells
    if settings.Find_Clusters and settings.process_dists:
        SampleGroups.Get_Clusters()  
    # Calculation of MWW-statistics for cell counts and other data
    if settings.statistics:
        SampleGroups.Get_Statistics()
    # Creation of plots from various data (excluding statistical plots)
    if settings.Create_Plots:
        SampleGroups.create_plots()
    print('\nANALYSIS COMPLETED')


if __name__ == '__main__':
    main()
