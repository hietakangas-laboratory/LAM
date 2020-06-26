# -*- coding: utf-8 -*-
r"""
Run file for Longitudinal Analysis of Midgut.

Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen
@version: 0.2.2

Distributed under GNU General Public License v3.0
-------------------------------------------------------------------------------

DEPENDENCIES:
------------
matplotlib (3.1.3), numpy (1.18.1), pandas (1.0.1), pathlib2
(2.3.5), scipy (1.4.1), seaborn (0.10.0), shapely (1.7.0),
scikit-image (0.16.2), statsmodels (0.11.0)

INSTALLATION:
------------
The master folder contains environment.yml that can be used to create Anaconda
environment for LAM-use. For a Python environment, requirements.txt is
included.

- Anaconda env:
    conda env create -n <yourenvname> -f <path\to\environment.yml>
    conda activate <yourenvname>

- Python env:
    1.	python -m venv <yourenvname>
        •	Linux:
            source <yourenvname>/bin/activate
        •	Windows:
            <yourenvname>\Scripts\activate.bat
    2.	pip install -r <path-to-requirements.txt>
        • !!!  On Windows you need to install Shapely separately (see below).
        You can either remove shapely from the requirements.txt or add ‘#’ in
        front of the line to pass it, in order to install all other necessary
        dependencies.

- Anaconda3 base environment:
    1. install Anaconda3 distribution (https://www.anaconda.com/distribution/)
    2. Add dependencies
         Open Anaconda Prompt and write command:
           conda install --file <LAM-master\requirements.txt>
         (You may need to add conda-forge to conda channels:
           conda config –add channels conda-forge           )
 

USAGE:
-----
Script for longitudinal analysis of Drosophila midgut images. To run the
script, change the work directory in either Sett.py or the GUI to the directory
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
associate the data

The script first creates a vector based on one channel ("vectChannel",
typically DAPI), in order to approximate the midgut along its length. Positions
on other channels can then be projected onto the vector, and cell numbers can
be quantified along the midgut. The vector is divided into user-defined number
of bins that are used for comparative analyses

On some experiments the size proportions of different regions may alter, e.g.
when comparing starved and fully-fed midguts, more accurate results can be
obtained by dividing the image/data into multiple analyses. A typical way to do
this is to run separate analyses for R1-2, R3, and R4-5. Alternatively, a user-
defined coordinate (MP = measurement point) at a distinguishable point can be
used to anchor the individual samples for comparison, e.g. points at R2/3-
border are lined, with each sample having variable numbers of bins on either
side. The variation however likely leads to a compounding error as distance
from the MP grows. When MP is not used, the samples are lined at bin 0, and
compared bin-by-bin. The MP-input is done similarly to channel data, i.e. as a
separate directory that contains position.csv for a single coordinate, the MP.

For more extensive instructions, see user manual.
"""
# LAM module
from settings import settings as Sett
import ParseCmds as pc

# Standard libs
import sys
import pathlib as pl


def main(gui_root=None):
    """Perform LAM-analysis based on settings.py."""
    import system
    import analysis
    import process
    import BorderDetection as bd
    from settings import store
    system_paths = system.start()
    # If sample processing set to True, create vectors, collect and project
    # data etc. Otherwise continue to plotting and group-wise operations.
    if Sett.process_samples:
        system.test_vector_ext(system_paths.samplesdir)
        process.Create_Samples(system_paths)
        # If only creating vectors, return from main()
        if not any([Sett.process_counts, Sett.process_dists,
                    Sett.Create_Plots, Sett.statistics]):
            return
    if Sett.process_counts and Sett.project:
        if not Sett.process_samples:
            process.vector_test(system_paths.samplesdir)
        process.Project(system_paths)
    # If performing 'Count' without projection, only calculate counts:
    elif Sett.process_counts and not Sett.project:
        process.find_existing(system_paths)
        if Sett.measure_width:
            analysis.Get_Widths(system_paths.samplesdir, system_paths.datadir)
    # After all samples have been collected/created, find their respective MP
    # bins and normalize (anchor) cell count data.
    process.Get_Counts(system_paths)
    # Storing of descriptive data of analysis, i.e. channels/samples/groups
    system_paths.save_AnalysisInfo(store.samples, store.samplegroups,
                                   store.channels)
    # After samples have been counted and normalized
    sample_groups = analysis.Samplegroups(system_paths)
    # Finding of nearest cells and distances
    if Sett.Find_Distances and Sett.process_dists:
        sample_groups.Get_DistanceMean()
    # Finding clustered cells
    if Sett.Find_Clusters and Sett.process_dists:
        sample_groups.Get_Clusters()
    # Computing total values from each sample's each bin
    if (Sett.statistics and Sett.stat_total) or Sett.process_counts:
        sample_groups.Get_Totals()
    # Find border regions
    if Sett.border_detection:
        bd.detect_borders(system_paths, sample_groups._samplePaths,
                          sample_groups._grpPalette, store.center,
                          Sett.border_vars, Sett.scoring_vars,
                          Sett.peak_thresh, Sett.border_channel)
    # Get and select border data if needed:
    if Sett.Create_Plots and Sett.add_peaks:
        bd.peak_selection(system_paths.datadir, gui_root)
    # Calculation of MWW-statistics for cell counts and other data
    if Sett.statistics:
        analysis.test_control()
        sample_groups.Get_Statistics()
    # Creation of plots from various data (excluding statistical plots)
    if Sett.Create_Plots:
        sample_groups.create_plots()


def main_catch_exit(LAM_logger=None, gui_root=None):
    """Run main() while catching exc eptions for logging."""
    import logger as lg
    if LAM_logger is None:  # If no logger given, get one
        LAM_logger = lg.setup_logger(__name__, new=True)
        lg.print_settings()  # print settings of analysis to log
        LAM_logger = lg.get_logger(__name__)
    try:
        print("START ANALYSIS")
        main(gui_root=gui_root)  # run analysis
        lg.logprint(LAM_logger, 'Completed', 'i')
        lg.Close()
        print('\nCOMPLETED\n')
    # Catch and log possible exits from the analysis
    except KeyboardInterrupt:
        lg.logprint(LAM_logger, 'STOPPED: keyboard interrupt', 'e')
        print("STOPPED: Keyboard interrupt by user.\n")
    except SystemExit:
        lg.logprint(LAM_logger, 'EXIT\n', 'ex')
        print("STOPPED\n")
        lg.log_Shutdown()
    except AssertionError:
        msg = 'STOPPED: No vectors found for samples.'
        print(msg + '\n')
        lg.logprint(LAM_logger, msg, 'c')
        lg.log_Shutdown()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        parser = pc.make_parser()
        pc.change_settings(parser)
    if Sett.GUI:  # Create GUI if using it
        import tkinter as tk
        import interface
        ROOT = tk.Tk()
        GUI = interface.base_GUI(ROOT)
        ROOT.mainloop()
    else:  # Otherwise make workdir into usable path and start the analysis
        Sett.workdir = pl.Path(Sett.workdir)
        main_catch_exit()
