# -*- coding: utf-8 -*-
"""
LAM-module for user-defined settings.

Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen

"""
import pathlib as pl


class settings:
    """A class for holding all user-end settings for the analysis."""

    # ####################### PRIMARY SETTINGS #######################
    GUI = True  # Use graphical user interface (True / False)

    # DEFINE PATH TO ANALYSIS FOLDER:
    # (Use input r'PATH' where PATH is your path)
    workdir = pl.Path(r'C:')
    # Whether to gather raw data and create vectors. If False, expects to find
    # pre-created datafiles in the Analysis Data directory, i.e. a previous
    # full run has been made, and there has been no edits to the data files.
    process_samples = False  # CLEARS DATA FILES-DIRECTORY
    # Whether to project, count and normalize data. If set to False, expect all
    # data to be in place. Can be used to e.g. create additional plots faster.
    process_counts = False
    # Whether to compute average distances and clusters.
    process_dists = False
    # Set True/False to set all plotting functionalities ON/OFF
    Create_Plots = True     # ON / OFF switch for plots
    # Whether to calculate statistics
    statistics = True
    ##################################################################

    # -#-#-#-#-#-#-#-# VECTOR CREATION & PROJECTION #-#-#-#-#-#-#-#-#- #
    # The channel based on which the vector is created
    vectChannel = "DAPI"
    # Whether to do projection when using count. If False, expects projection
    # data to be found in channel files of './Analysis Data/Samples/'.
    project = True
    # Number of bins used for projection unto vector.
    projBins = 40

    # Make vector by creating binary image and then skeletonizing. If False,
    # vector is created by finding middle point between smallest and largest
    # Y-axis position in bin.
    SkeletonVector = True
    SkeletonResize = 0.4    # Binary image resize. Keep at steps of 0.1.
    # Find distance (in input coords) in skeleton vector creation
    find_dist = 75
    BDiter = 5          # Binary dilation iterations (set to 0 if not needed)
    SigmaGauss = 7    # Sigma for gaussian smoothing (set to 0 if not needed)
    simplifyTol = 20    # Tolerance for vector simplification.
    # Number of bins used for vector creation when using the median vector
    # creation. Increasing bin number too much may lead to stair-like vector;
    # increasing 'simplifyTol' can correct the steps.
    medianBins = 100
    # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#- #

    # ---MEASUREMENT POINTS--- #
    # Whether to use measurement point coordinates for normalization. If False,
    # the samples will be handled as perfectly aligned from beginning to end.
    useMP = True
    # The name of the file used for normalizing between samples, i.e. anchoring
    MPname = "MP"

    # ---DATA GATHERING--- #
    header_row = 2  # On which row does the data have its header (start = 0)

    # Additional data to be collected from channels. Key (the first string
    # before ':') must be the data column label and the following string for
    # searching the csv-file containing the wanted data. If multiple files are
    # to be collected (e.g. intensities), the code expects the data column to
    # have an ID number after the search string, separated by an underscore "_"
    # E.g. "Intensity_Mean" => "Intensity_Mean_Ch=1".
    # The last value is the unit of the values used for plotting labels,
    # e.g. um^2 for area. um^2 = "$\u03BCm^2$"  ;   um^3 = "$\u03BCm^3$"
    AddData = {"Area": ["Area.csv", "Area, $\u03BCm^2$"],
               "Volume": ["Volume.csv", "Volume, $\u03BCm^3$"],
               "Intensity Mean": ["Intensity_Mean", "Intensity"]
               }
    # If set to true, replaces the above mentioned (AddData) ID numbers with an
    # alternative moniker as defined in channelID
    replaceID = True
    channelID = {"Ch=1": "Pros",
                 "Ch=2": "GFP",
                 "Ch=3": "SuH",
                 "Ch=4": "DAPI"
                 }
    ###################################################################

    # ------ANALYSIS OPTIONS------ #
    # ---DISTANCE MEANS--- #
    # Find nearest cell of each cell. Distance estimation is performed for all
    # channels in Distance_Channels list. If use target is True, the nearest
    # cell is found on the channel defined by target_chan, otherwise they are
    # found within the channel undergoing analysis.
    Find_Distances = True
    Distance_Channels = ['DAPI', 'DAPIEC']
    use_target = False
    target_chan = "Pros"
    # The maximum distance the nearest cell will be looked at. Increase is
    # computationally expensive, depending on the size of the dataset and the
    # density of cells.
    maxDist = 40    # Radius around a cell
    # Whether to look only at cells of certain size. Default is to include
    # cells smaller than Vol_inclusion. If cells of greater volume are wanted,
    # designate incl_type to be 'greater'. Otherwise, it can be left empty.
    Vol_inclusion = 0    # Set to zero if not wanted.
    incl_type = ""

    # ---CLUSTERS--- #
    # Whether to compute clusters
    Find_Clusters = False
    Cluster_Channels = ["GFP"]  # , "Pros"]
    Cl_maxDist = 10         # Radius around a cell
    Cl_Vol_inclusion = 0    # Set to zero if not wanted.
    Cl_incl_type = ""       # Same as above in Find_Distances
    Cl_min = 3
    Cl_max = 50

    # ---STATISTICS OPTIONS--- #
    stat_versus = True
    stat_total = True
    windowed = True
    trail = 1
    lead = 1
    ylim = 25               # negative log2 y-limit
    alpha = 0.05            # for rejection of H_0, applies to statistics files
    # Plots
    stars = False  # Make P-value stars (*:<0.05 **:<0.01 ***:<0.001)
    fill = True  # fill significant bins with marker color
    negLog2 = True  # Forces stars to be False
    observations = True  # Plot individual observations
    # The name of the control group that the statistics are run against.
    cntrlGroup = "starv"

    # ---PLOTTING OPTIONS--- #
    Create_Channel_Plots = True
    Create_AddData_Plots = True     # Plots also nearest distance & clusters
    Create_Channel_PairPlots = False
    Create_Heatmaps = True
    Create_Distribution_Plots = True
    Create_Statistics_Plots = True  # requires statistics to be True
    Create_Cluster_Plots = False

    # Variable vs. variable plots:
    Create_ChanVSAdd_Plots = False  # Pairs of channel and additional data
    Create_AddVSAdd_Plots = False  # Pairs of additional data
    # Create plots of all possible pair combinations of the following:
    vs_channels = ['DAPI', 'GFP']
    vs_adds = ['Intensity Mean']  # Use the pre-defined keys

    # Whether to drop outliers from plots ONLY
    Drop_Outliers = True
    dropSTD = 2.5  # The standard deviation limit for drop
    # NOTE ON DROPPING OUTLIERS:
    #   Some variables may not be normally distributed (e.g. feature counts),
    #   and using standard deviation for dropping on such data might remove
    #   valid features from the plot.

    # Gives values some coordinate-shift in channel pairplots. Useful in pre-
    # senting the data, as it is discrete; most of the data would be hidden
    # under others. Does not affect the underlying data.
    plot_jitter = True

    # ---Figure save-format ---#
    # Supported formats for the plot files: eps, jpeg, jpg, pdf, pgf, png, ps,
    # raw, rgba, svg, svgz, tif, tiff.
    saveformat = "pdf"

    seaborn_style = 'ticks'   # Different styles of plots, e.g. background
    # Available styles #
    # write to the console the following lines
    # 1.    import matplotlib.style as style
    # 2.    style.available

    seaborn_context = 'talk'  # Defines plot object sizes, i.e. text
    # size order small to large: 'paper', 'talk', 'poster'

    # Define colors used for sample groups
    # (xkcd colors: 'https://xkcd.com/color/rgb/')
    palette_colors = ['orange yellow', 'aqua marine', 'tomato', 'dark lime',
                      'tan brown', 'red violet', 'dusty green', 'sandy brown']

    non_stdout = False  # Redirect stdout to a window when using executable
    # Determine width of gut based on vector channel projection distances
    measure_width = True


class store:
    """
    Store important variables for the analysis.

    VARIABLES FOR LAM INTERNAL USE!
    """

    samplegroups = []  # All samplegroup in analysis
    channels = []  # All channels in analysis
    samples = []  # All samples in analysis
    totalLength = 0  # The length of DataFrame after all samples are anchored
    center = 0  # The index of the anchoring point within the DataFrame
    clusterPaths = []
