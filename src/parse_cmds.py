# -*- coding: utf-8 -*-
"""
LAM-module for parsing commandline arguments.

Created on Fri Jun  5 15:26:28 2020

@author: Arto I. Viitanen
"""


import argparse
import pathlib as pl

# LAM module
from src.settings import settings as Sett

def make_parser():
    """Make parser-object for LAM run."""
    hmsg = ("""Perform LAM analysis from command line. Args described as toggle
            alter the default value in settings.py to the opposite Boolean. All
            settings that are not altered through parsed arguments will use
            default values from settings.py.
            
            Example:
                python src\\run.py -p C:\\DSS -o cds -MGD -F -f GFP -f DAPI
            """)

    parser = argparse.ArgumentParser(description=hmsg)

    # MAIN
    parser.add_argument("-p", "--path", help="Analysis directory path",
                        type=str)
    ops = "r (process), c (count), d (distance), l (plots), s (stats)"
    htext = f"primary option string: {ops}"
    parser.add_argument("-o", "--options", help=htext, type=str)
    parser.add_argument("-b", "--bins", help="Sample bin number", type=int)
    parser.add_argument("-v", "--channel", help="Vector channel name", type=str)
    parser.add_argument("-g", "--control_group", help="Name of control group", type=str)
    parser.add_argument("-H", "--header", help="Data header row number", type=int)
    parser.add_argument("-M", "--measurement_point", help="toggle useMP", action="store_true")
    parser.add_argument("-m", "--mp_name", help="Name of MP", type=str)
    parser.add_argument("-G", "--GUI", help="toggle GUI", action="store_true")


    # Distance args
    parser.add_argument("-F", "--feature_distances", help="f-to-f distances", action="store_true")
    parser.add_argument("-f", "--distance_channels", help="f-to-f distance channels", type=str, action='append')


    # Cluster args
    parser.add_argument("-C", "--clusters", help="Feature clustering", action="store_true")
    parser.add_argument("-c", "--cluster_channels", help="Clustering channels", type=str, action='append')
    parser.add_argument("-d", "--cluster_distance", help="Clustering max distance", type=int)


    # Other operations
    parser.add_argument("-B", "--borders", help="Toggle border detection", action="store_true")
    parser.add_argument("-W", "--widths", help="Toggle width calculation", action="store_true")
    parser.add_argument("-r", "--no_projection", help="Projection to false", action="store_true")
    parser.add_argument("-D", "--force_dialog", help="Force no user input", action="store_true")

    parser = parser.parse_args()
    return parser


def change_settings(parser):
    """Use parsed arguments to change settings."""
    changed_settings = {}  # For storing changed settings for printing

    # ADJUSTMENT OF SETTINGS BASED ON PARSED ARGUMENTS:
    # Work directory, i.e. analysis directory
    if parser.path:
        Sett.workdir = pl.Path(parser.path)
    print(f'Work directory: {Sett.workdir}')

    # Primary options
    if parser.options:
        primaries = primary_options(parser.options)
        print(f"Primary settings: {', '.join(primaries)}")

    # Number of bins:
    if parser.bins is not None:
        Sett.projBins = parser.bins
        changed_settings.update({'Bins': Sett.projBins})

    # Vector creation channel
    if parser.channel is not None:
        Sett.vectChannel = parser.channel
        changed_settings.update({'Vector channel': Sett.vectChannel})

    # Control group for statistical testing
    if parser.control_group is not None:
        Sett.cntrlGroup = parser.control_group
        changed_settings.update({'Control group': Sett.cntrlGroup})

    # Header row for data collection
    if parser.header is not None:
        Sett.header_row = parser.header
        changed_settings.update({'Header row': Sett.header_row})

    # Whether to calculate feature-tofeature distances
    if parser.feature_distances is not None:
        Sett.find_distances = parser.feature_distances
        changed_settings.update({'Distance': Sett.find_distances})

    # Channels that are used for f-to-f distances
    if parser.distance_channels is not None:
        Sett.distance_channels = parser.distance_channels
        changed_settings.update({'Distance channels': Sett.distance_channels})

    # Whether to determine clusters
    if parser.clusters is not None:
        Sett.find_clusters = parser.clusters
        changed_settings.update({'Clusters': Sett.find_clusters})

    # Defining of channels that clustering is performed on
    if parser.cluster_channels is not None:
        Sett.Cluster_Channels = parser.Cluster_Channels
        changed_settings.update({'Cluster channels': Sett.Cluster_Channels})

    # Maximal distance of cells that are considered a cluster
    if parser.cluster_distance is not None:
        Sett.cl_max_dist = parser.cluster_distance
        changed_settings.update({'Cluster max distance': Sett.cl_max_dist})

    # Whether to do border detection
    if parser.borders  is True:
        Sett.border_detection = not Sett.border_detection
        changed_settings.update({'Detect borders': Sett.border_detection})

    # Whether to estimate sample width
    if parser.widths  is True:
        Sett.measure_width = not Sett.measure_width
        changed_settings.update({'Width estimation': Sett.measure_width})

    # Whether to project cell data on to the vector
    if parser.no_projection is True:
        Sett.project = False
        changed_settings.update({'Projection': Sett.project})

    # Use of MP
    if parser.measurement_point is True:
        Sett.useMP = not Sett.useMP
        changed_settings.update({'MP': Sett.useMP})

    # Define MP name
    if parser.mp_name is not None:
        Sett.MPname = parser.mp_name
        changed_settings.update({'MP name': Sett.MPname})

    # Whether to use GUI
    if parser.GUI is True:
        Sett.GUI = not Sett.GUI
        changed_settings.update({'GUI': Sett.GUI})

    # Whather to pass all dialog
    if parser.force_dialog is True:
        Sett.force_dialog = parser.force_dialog
        changed_settings.update({'Force pass dialog': Sett.force_dialog})

    # Print changed settings if any
    if changed_settings:
        print("\nCHANGED SETTINGS:")
        for (key, value) in changed_settings.items():
            print(f"{key}: {value}")
        print("\n")


def primary_options(string):
    """Define primary settings from argument string."""
    string = string.lower()  # change to lower for comparison
    setting_list = []  # Holds used primary settings

    # Change everything to False
    Sett.process_samples = False
    Sett.process_counts = False
    Sett.process_dists = False
    Sett.Create_Plots = False
    Sett.statistics = False

    # Adjust settings
    if 'r' in string:  # Vector creation
        Sett.process_samples = True
        setting_list.append("Process")
    if 'c' in string:  # Counting
        Sett.process_counts = True
        setting_list.append("Count")
    if 'd' in string:  # Distances
        Sett.process_dists = True
        setting_list.append("Distance")
    if 'l' in string:  # Plotting
        Sett.Create_Plots = True
        setting_list.append("Plots")
    if 's' in string:  # Statistics
        Sett.statistics = True
        setting_list.append("Stats")
    return setting_list
