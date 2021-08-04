# -*- coding: utf-8 -*-
# Standard libs
r"""
Run file for Longitudinal Analysis of Midgut (LAM v0.4.3).

Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen

Distributed under GNU General Public License v3.0
-------------------------------------------------------------------------------
INSTALLATION:
------------
LAM is developed in Python >=3.7 environment. The LAM-master–folder has setup.cfg
that contains required information for installation with setuptools.

It is recommended to create your own virtual environment for LAM in order to avoid
any dependency clashes. You can do this in command line by giving the following
command:
     python -m venv <yourenvname>
     e.g. python -m venv lam_env

If you do not have Python in your system environment variables, you will also need
to give the full path to your python.exe,
    e.g. c:\Program Files\Python38\python setup.py install

Then activate the virtual environment with:
     Linux: 	source <yourenvname>/bin/activate
     Windows: 	<yourenvname>\Scripts\activate

You can then install LAM and dependencies on your environment with:
    pip install lam

    OR ALTERNATIVELY by using the setup.cfg with the following command while
    located in in LAM-master folder:

    python setup.py install

Windows-users may need to install Shapely>=1.7.0 from a pre-compiled wheel in order
to properly link GEOS and cython.
Wheel can be found here: https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely


USAGE:
-----
Script for longitudinal analysis of Drosophila midgut images. To run the
script, change the work directory in either Sett.py or the GUI to the directory
containing the directories for each individual sample.
The sample directories should be named as:
    <samplegroup>_<samplename>
    e.g. starv_sample1
Within the sample directories, cell positions and other data should be in
channel-specific directories named as:
    <samplegroup>_<samplename>_<channel>_xyz
    e.g. starv_sample1_GFP_Stats
Avoid using additional underscores "_" in naming of directories and files, as
it is used as delimiter between the various information contained in the paths.
Doing so may cause the analysis to fail.

The channel directories have to contain "Position.csv" with column labels
"Position X", "Position Y", "Position Z", and "ID". The cell ID should be the
same between files containing other used data, such as "Area.csv", to properly
associate the data.

You must first create vectors for the samples based on one channel ("vectChannel",
typically DAPI) in order to approximate the midgut along its length. The
vector-creation is controlled by the process-setting. After, positions on other
channels can then be projected onto the vector, and cell numbers can be
quantified along the midgut. The vector is divided into user-defined number of
bins that are used for comparative analyses.

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

For more extensive description and instructions, see docs\user manual.
"""
import sys
import pathlib as pl

# LAM module
from src.settings import Store, Settings as Sett
import src.parse_cmds as pc
import src.system as system
import src.analysis as analysis
import src.process as process
import src.border_detect as bd
import src.logger as lg

LAM_logger = None


def main(gui_root=None):
    """Perform LAM-analysis based on settings.py."""
    system_paths = system.start()

    # If sample processing set to True, create vectors, collect and project
    # data etc. Otherwise continue to plotting and group-wise operations.
    if Sett.process_samples:
        system.test_vector_ext(system_paths.samplesdir)
        process.create_samples(system_paths)
        # If only creating vectors, return from main()
        if not any([Sett.process_counts, Sett.process_dists, Sett.Create_Plots, Sett.statistics]):
            return
    if Sett.process_counts and Sett.project:
        if not Sett.process_samples:
            process.vector_test(system_paths.samplesdir)
        process.project(system_paths)

    # If performing 'Count' without projection, only calculate counts:
    elif Sett.process_counts and not Sett.project:
        process.find_existing(system_paths)
        if Sett.measure_width:
            analysis.get_widths(system_paths.samplesdir, system_paths.datadir)

    # After all samples have been collected/created, find their respective MP
    # bins and normalize (anchor) cell count data.
    process.get_counts(system_paths)

    # Storing of descriptive data of analysis, i.e. channels/samples/groups
    system_paths.save_analysis_info(Store.samples, Store.samplegroups, Store.channels)

    # Create object to hold samplegroup info
    sample_groups = analysis.Samplegroups(system_paths)

    # Finding of nearest cells and distances
    if Sett.find_distances and Sett.process_dists:
        sample_groups.get_distance_mean()

    # Finding clustered cells
    if Sett.find_clusters and Sett.process_dists:
        sample_groups.get_clusters()

    # Computing total values from each sample's each bin
    if (Sett.statistics and Sett.stat_total) or Sett.process_counts:
        sample_groups.get_totals()

    # Find border regions
    if Sett.border_detection:
        conf = bd.test_channel(sample_groups.sample_paths, Sett.border_channel)
        if conf:
            bd.detect_borders(system_paths, sample_groups.sample_paths, sample_groups.grp_palette, Store.center,
                              Sett.border_vars, Sett.scoring_vars, Sett.peak_thresh, Sett.border_channel)

    # Get and select border data if needed:
    if Sett.Create_Plots and Sett.add_peaks:
        bd.peak_selection(system_paths.datadir, gui_root)

    # Calculation of MWW-statistics for cell counts and other data
    if Sett.statistics:
        conf = analysis.test_control()
        if conf:
            sample_groups.get_statistics()

    # Creation of plots from various data (excluding statistical plots)
    if Sett.Create_Plots:
        sample_groups.create_plots()


def main_catch_exit(LAM_logger=None, gui_root=None):
    """Run main() while catching exceptions for logging."""
    if LAM_logger is None:  # If no logger given, get one
        LAM_logger = lg.setup_logger(__name__, new=True)
        lg.print_settings()  # print settings of analysis to log
        lg.create_loggers()

    try:
        print("START ANALYSIS")
        main(gui_root=gui_root)  # run analysis
        lg.logprint(LAM_logger, 'Completed', 'i')
        lg.close_loggers()
        print('\nCOMPLETED\n')

    # Catch and log possible exits from the analysis:
    except KeyboardInterrupt:
        lg.logprint(LAM_logger, 'STOPPED: keyboard interrupt', 'e')
        print("STOPPED: Keyboard interrupt by user.\n")
        lg.close_loggers()

    except SystemExit:
        lg.logprint(LAM_logger, 'EXIT\n\n', 'ex')
        print("STOPPED\n")
        lg.log_shutdown()

    except process.VectorError as e:
        print(e.message + '\n')
        print(f'Missing: {", ".join(e.samples)}')
        lg.logprint(LAM_logger, e.message, 'ex')
        lg.log_shutdown()


def run():
    # If arguments given from commandline, parse them
    if len(sys.argv) > 1:
        parser = pc.make_parser()
        pc.change_settings(parser)

    # Get or set work directory.
    system.check_workdir()

    # Create GUI if needed
    if Sett.GUI:
        import tkinter as tk
        import src.interface as interface
        root = tk.Tk()
        interface.BaseGUI(root)
        # Add LAM-icon
        interface.set_icon(root)

        root.mainloop()

    # Otherwise make workdir into usable path and start the analysis
    else:
        Sett.workdir = pl.Path(Sett.workdir)
        main_catch_exit()


if __name__ == '__main__':
    run()
