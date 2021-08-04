# -*- coding: utf-8 -*-
"""
LAM-module for user-defined settings.

Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen

"""

class Settings:
    """A class for holding all user-end settings for the analysis."""

    # ####################### PRIMARY SETTINGS #######################
    GUI = True  # Use graphical user interface (True / False)
    force_dialog = False  # Force no user input (True for no dialog)

    # Detect border regions
    border_detection = True
    # Determine width of gut based on vector channel projection distances
    measure_width = True
    plot_width = True

    # DEFINE PATH TO ANALYSIS FOLDER:
    # (Use input r'PATH' where PATH is your path)
    workdir = r''

    # Whether to gather raw data and create vectors. If False, expects to find
    # pre-created datafiles in the Analysis Data directory, i.e. a previous
    # full run has been made, and there has been no edits to the data files.
    process_samples = False  # CLEARS DATA FILES-DIRECTORY
    # Whether to project, count and normalize data. If set to False, expect all
    # data to be in place. Can be used to e.g. create additional plots faster.
    process_counts = True
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
    projBins = 62

    # Make vector by creating binary image and then skeletonizing. If False,
    # vector is created by finding middle point between smallest and largest
    # Y-axis position in bin.
    SkeletonVector = True
    SkeletonResize = 0.1    # Binary image resize. Keep at steps of 0.1.
    # Find distance (in input coords) in skeleton vector creation
    find_dist = 180
    BDiter = 3         # Binary dilation iterations (set to 0 if not needed)
    SigmaGauss = 5    # Sigma for gaussian smoothing (set to 0 if not needed)
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

    header_row = 0  # On which row does the data have its header (starts from 0)

    # Additional data to be collected from channels. Key (the first string
    # before ':') must be the data column label and the following string for
    # searching the csv-file containing the wanted data. If multiple files are
    # to be collected (e.g. intensities), the code expects the data column to
    # have an ID number after the search string, separated by an underscore "_"
    # E.g. "Intensity_Mean" => "Intensity_Mean_Ch=1".
    # The last value is the unit of the values used for plotting labels,
    # e.g. um^2 for area. um^2 = "$\u03BCm^2$"  ;   um^3 = "$\u03BCm^3$"
    AddData = {"Area": ["Position.csv", "Area, $\u03BCm^2$"],
               "Volume": ["Position.csv", "Volume, $\u03BCm^3$"],
               "Intensity Mean": ["Position.csv", "Intensity"]
               }
    # If set to true, replaces the above mentioned (AddData) ID numbers with an
    # alternative moniker as defined in channelID
    replaceID = False
    channelID = {"Ch=1": "Pros",
                 "Ch=2": "GFP",
                 "Ch=3": "SuH",
                 "Ch=4": "DAPI"
                 }
    ###################################################################

    # ------ANALYSIS OPTIONS------ #


    # ---FEATURE-TO-FEATURE DISTANCE--- #

    # Define column name for filtering data (for inclusion and cl_inclusion)
    incl_col = "Volume"

    # Find nearest cell of each cell. Distance estimation is performed for all
    # channels in distance_channels list. If use target is True, the nearest
    # cell is found on the channel defined by target_chan, otherwise they are
    # found within the channel undergoing analysis.
    find_distances = True
    distance_channels = ['DAPI']
    use_target = False
    target_chan = "Pros"
    # FILTERING DOES NOT AFFECT DATA ON TARGET CHANNEL (use_target/target_chan)

    # The maximum distance the nearest cell will be looked at.
    max_dist = 99    # Radius around a cell
    # Whether to look only at cells with certain characteristics. Default is
    # to include cells smaller than 'inclusion'. If cells of greater volume are
    # wanted, designate incl_type to be 'greater'. Otherwise, it can be left
    # empty. Set 'inclusion' to zero if not wanted.
    inclusion = 0
    incl_type = ""


    # ---CLUSTERS--- #

    # Whether to compute clusters
    find_clusters = False
    cluster_channels = ["GFP"]  # , "Pros"]
    cl_max_dist = 10         # Radius around a cell
    cl_inclusion = 0    # Set to zero if not wanted.
    cl_incl_type = ""       # Same as above in feature-to-feature distances
    cl_min = 3
    cl_max = 50


    # ---BORDER DETECTION--- #

    # Name of channel from which scoring variables are collected from
    border_channel = vectChannel  # Default is vector creation channel
    peak_thresh = 0.5  # Border score threshold for peak detection
    # Plot individual samples (requires Create_Border_Plots==True)
    plot_samples = False
    # Data columns to use for detection (sample width is always collected)
    # Adding 'Count' will get cell counts
    border_vars = ['Area', f'Nearest_Dist_{border_channel}']
    # Scoring weights for variables. Adding extension '_std' or '_diff' will
    # perform respective calculations to the underlying variables.
    scoring_vars = {'width': -1.5,
                    'width_diff': 1.5,
                    'Area_diff': -1,
                    f'Nearest_Dist_{border_channel}_diff': 1.5}


    # ---STATISTICS OPTIONS--- #

    stat_versus = True
    stat_total = True
    windowed = True
    trail = 1
    lead = 1
    # for rejection of H_0, applies to statistics files
    alpha = 0.05

    # Plots
    stars = False  # Make P-value stars (*:<0.05 **:<0.01 ***:<0.001)
    fill = True  # fill significant bins with marker color
    negLog2 = True  # Forces stars to be False
    ylim = 35  # negative log2 y-limit

    # The name of the control group that the statistics are run against.
    cntrlGroup = "ctrl"

    observations = True  # Plot individual observations. DEPRECATED!


    # ---PLOTTING OPTIONS--- #

    Create_Channel_Plots = True
    Create_AddData_Plots = True     # Plots also nearest distance & clusters
    Create_Channel_PairPlots = False
    Create_Heatmaps = True
    Create_Distribution_Plots = True
    Create_Statistics_Plots = True  # requires statistics to be True
    Create_Cluster_Plots = False
    Create_Border_Plots = True  # Requires calculated width data

    # Variable vs. variable plots:
    Create_ChanVSAdd_Plots = False  # Pairs of channel and additional data
    Create_AddVSAdd_Plots = False  # Pairs of additional data
    # Create plots of all possible pair combinations of the following:
    vs_channels = ['DAPI', 'GFP']
    vs_adds = ['Intensity Mean', 'Area']  # Use the pre-defined keys

    # Add border detection peaks to other plots
    add_peaks = True
    select_peaks = True  # If true, LAM asks which detected peaks are plotted
    # Use "All-Border_Scores"-plot to determine valid peaks

    # Whether to drop outliers from plots ONLY
    Drop_Outliers = True
    dropSTD = 3  # The standard deviation limit for drop
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

    # Define colors used for sample groups
    # (xkcd colors: 'https://xkcd.com/color/rgb/')
    palette_colors = ['soft blue', 'goldenrod', 'moss green', 'faded red', 'turquoise', 'dusky pink', 'aqua marine',
                      'tan brown', 'red violet', 'blood', 'brown grey',  'dark lime', 'sandy brown', 'pine',
                      'purply pink']

    non_stdout = False  # Redirect stdout to a window when using executable


class Store:
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
    border_peaks = []
