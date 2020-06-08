# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:26:28 2020

@author: Arto I. Viitanen
"""


import argparse
import pathlib as pl
from settings import settings as Sett

def make_parser():
    parser = argparse.ArgumentParser(description='Perform LAM analysis.')
    # MAIN
    parser.add_argument("-p", "--path", help="analysis directory path",
                        type=str)
    ops = "r (process), c (count), d (distance), l (plots), s (stats)"
    htext = f"primary option string: {ops}"
    parser.add_argument("-o", "--options", help=htext, type=str)
    parser.add_argument("-b", "--bins", help="sample bin number", type=int)
    parser.add_argument("-v", "--channel", help="vector channel name",
                        type=str)
    parser.add_argument("-g", "--control_group", help="name of control group",
                        type=str)
    parser.add_argument("-H", "--header", help="header row number", type=int)
    parser.add_argument("-M", "--measurement_point", help="toggle useMP",
                        action="store_true")
    parser.add_argument("-m", "--mp_name", help="name of MP", type=str)
    parser.add_argument("-G", "--GUI", help="toggle GUI", action="store_true")

    # Distance args
    parser.add_argument("-F", "--feature_distances", help="f-to-f distances",
                        action="store_true")
    parser.add_argument("-f", "--distance_channels",
                        help="f-to-f distance channels", type=str,
                        action='append')

    # Cluster args
    parser.add_argument("-C", "--clusters", help="feature clustering",
                        action="store_true")
    parser.add_argument("-c", "--cluster_channels",
                        help="clustering channels", type=str, action='append')
    parser.add_argument("-d", "--cluster_distance",
                        help="clustering max distance", type=int)

    # Other operations
    parser.add_argument("-B", "--borders",
                        help="toggle border detection",
                        action="store_true")
    parser.add_argument("-W", "--widths", help="toggle width calculation",
                        action="store_true")
    parser.add_argument("-r", "--no_projection", help="projection to false",
                        action="store_true")
    parser = parser.parse_args()
    return parser


def change_settings(parser):
    if parser.path:
        Sett.workdir = pl.Path(parser.path)
    print(f'Work directory: {Sett.workdir}')
    if parser.options:
        primary_options(parser.options)
    if parser.bins:
        Sett.projBins = parser.bins
    if parser.channel:
        Sett.vectChannel = parser.channel
    if parser.control_group:
        Sett.cntrlGroup = parser.control_group
    if parser.header:
        Sett.header_row = parser.header

    if parser.feature_distances:
        Sett.Find_Distances = parser.feature_distances
    if parser.distance_channels:
        Sett.Distance_Channels = parser.distance_channels

    if parser.clusters:
        Sett.Find_Clusters = parser.clusters
    if parser.cluster_channels:
        Sett.Cluster_Channels = parser.cluster_channels
    if parser.cluster_distance:
        Sett.Cl_maxDist = parser.cluster_distance

    if parser.borders:
        Sett.border_detection = not Sett.border_detection
    if parser.widths:
        Sett.measure_width = not Sett.measure_width
    if parser.no_projection:
        Sett.project = False
    if parser.measurement_point:
        Sett.useMP = not Sett.useMP
    if parser.mp_name:
        Sett.MPname = parser.mp_name
    if parser.GUI:
        Sett.GUI = not Sett.GUI


def primary_options(string):
    string = string.lower()
    Sett.process_samples = False
    Sett.process_counts = False
    Sett.process_dists = False
    Sett.Create_Plots = False
    Sett.statistics = False
    if 'r' in string:
        Sett.process_samples = True
    if 'c' in string:
        Sett.process_counts = True
    if 'd' in string:
        Sett.process_dists = True
    if 'l' in string:
        Sett.Create_Plots = True
    if 's' in string:
        Sett.statistics = True
