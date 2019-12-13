# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen
Dependencies: Anaconda-included packages (Python 3.7), Shapely, pycg3d

Dependencies:
    1. install Anaconda3 distribution (https://www.anaconda.com/distribution/)
    2. add Shapely-package:
        Windows: 
            get Shapely .whl from "https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely"
            then write following command(s) in Anaconda prompt:
                (0.) pip install wheel          (should be in Anaconda)
                1. pip install <path-to-the-downloaded-whl-file>
        OS X & Linux:
            open Anaconda prompt and write following command:
                pip install shapely
    3. add pycg3d-package: 
        Open Anaconda Prompt, write command:
            pip install pycg3d
    
Script for longitudinal analysis of Drosophila midgut images. To run the script,
change the work directory in either Sett.py or the GUI to the directory 
containing the directories for each individual sample. 
The sample directories should be named as "<samplegroup>_xyz_<samplename>", 
where xyz can be anything, e.g. "starv_2018-11-06_Ctrl starved 1". Within the 
sample directories, cell positions and other data should be in channel-specific 
directories named as"<channel>_xyz", e.g. GFP_Statistics or GFP+Pros_whatevs. 
Avoid using underscore "_" in naming of directories and files, as it is used
as delimiter between the various information contained in the paths. Doing so
may cause the analysis to fail.

The channel directories have to contain "Position.csv" with column labels 
"Position X", "Position Y", "Position Z", and "ID". The cell ID should be the 
same between files containing other used data, such as "Area.csv", to properly 
associate the data.

The script first creates a vector based on one channel ("vectChannel", typically 
DAPI), in order to approximate the midgut along its length. Positions on other 
channels can then be projected onto the vector, and cell numbers can be quanti-
fied along the midgut. The vector is divided into user-defined number of bins 
that are used for comparative analyses.

On some experiments the size proportions of different regions may alter, e.g.
when comparing starved and fully-fed midguts, more accurate results can be 
obtained by dividing the image/data into multiple analyses. A typical way to do 
this is to run separate analyses for R1-2, R3, and R4-5. Alternatively, a user-
defined coordinate (MP = measurement point) at a distinguishable point can be 
used to anchor the individual samples for comparison, e.g. points at R2/3-border 
are lined, with each sample having variable numbers of bins on either side. The
variation however likely leads to a compounding error as distance from the MP 
grows. When MP is not used, the samples are lined at bin 0, and compared bin-
by-bin. The MP-input is done similarly to channel data, i.e. as a separate 
directory that contains position.csv for a single coordinate, the MP.

For more extensive instructions, see user manual. 
"""
from settings import settings as Sett

def main(LAM_logger): 
    import system, analysis, process
    from system import store
    systemPaths = system.start()
    # If sample processing set to True, create vectors, collect and project 
    # data etc. Otherwise continue to plotting and group-wise operations.
    if Sett.process_samples:
        process.Create_Samples(systemPaths)
        # If only creating vectors, return from main()
        if not any([Sett.process_counts,Sett.process_dists,Sett.Create_Plots,
            Sett.statistics]):  
            return
    if Sett.process_counts:
        process.Project(systemPaths)
    # After all samples have been collected/created, find their respective MP 
    # bins and normalize (anchor) cell count data. If MP's are not used, the 
    # samples are anchored at bin == 0.
    process.Get_Counts(systemPaths)
    # Storing of descriptive data of analysis, i.e. channels/samples/groups
    systemPaths.save_AnalysisInfo(store.samples, store.samplegroups, 
                                  store.channels)
    # After samples have been counted and normalized
    SampleGroups = analysis.Samplegroups(store.samplegroups, store.channels,
                                        store.totalLength, store.center, 
                                        systemPaths)
    # Computing total cell numbers from each sample's each bin
    if Sett.process_counts:
        SampleGroups.Get_Totals()
        # TODO add cluster counting (maybe save channel paths to csv when finding clusters) ???
#        if store.clusterPaths: # Collect clustering data from existing files
#            SampleGroups.Read_Clusters()
    # Finding of nearest cells and distances
    if Sett.Find_Distances and Sett.process_dists:
        SampleGroups.Get_DistanceMean()
    # Finding clustered cells
    if Sett.Find_Clusters and Sett.process_dists:
        SampleGroups.Get_Clusters()  
    # Calculation of MWW-statistics for cell counts and other data
    if Sett.statistics:
        SampleGroups.Get_Statistics()
    # Creation of plots from various data (excluding statistical plots)
    if Sett.Create_Plots:
        SampleGroups.create_plots()
    
def MAIN_catch_exit():
    """Run main() while catching system exit and keyboard interrupt for log."""
    # Get premade logger
    import logger as lg
    LAM_logger = lg.get_logger(__name__)
    try:
        print("START ANALYSIS")
        main(LAM_logger) # run analysis
    # Catch and log possible exits from the analysis
    except KeyboardInterrupt:
        lg.logprint(LAM_logger, 'STOPPED: keyboard interrupt', 'e')
        print("STOPPED: Keyboard interrupt by user")
    except SystemExit:
        lg.logprint(LAM_logger, 'SYSTEM EXIT\n', 'ex')
        print("System Exit")
        lg.log_Shutdown()
    finally:
        print('\nCOMPLETED\n')
        lg.logprint(LAM_logger, 'Completed', 'i')
        lg.Close() 
        

if __name__ == '__main__':
    if Sett.GUI: # Create GUI if using it
        import tkinter as tk
        import interface
        root = tk.Tk()
        gui = interface.base_GUI(root)
        root.mainloop()
    else: # Otherwise start the analysis
        import logger as lg
        LAM_logger = lg.setup_logger(__name__)
        lg.print_settings(LAM_logger) # print settings of analysis to log
        MAIN_catch_exit()