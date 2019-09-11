# -*- coding: utf-8 -*-
import numpy as np, pathlib as pl
#
class settings:
    """ A class for holding all user-end settings for the analysis."""
    ######################## PRIMARY SETTINGS ########################
    # DEFINE PATH TO ANALYSIS FOLDER:
    # (Use input r'PATH' where PATH is your path)
    workdir = pl.Path(r'\\ad.helsinki.fi\home\a\artoviit\Desktop\test')
    # Whether to gather data and create vectors. If False, expects to find pre-created
    # datafiles in the Analysis Data directory.
    process_samples = False
    # The channel based on which the vector is created
    vectChannel = "DAPI"
    # Make vector by creating binary image and then skeletonizing. If False, vector 
    # is created by finding middle point between smallest and largest Y-axis position in bin.
    SkeletonVector = True
    SkeletonResize = 0.5    # Binary image resize for reprocessing, e.g. smoothing. Keep at steps of ten.
    BDiter = 0             # Number of iterations for binary dilation (set to 0 if not needed)
    SigmaGauss = 0.5       # Sigma for gaussian smoothing (set to 0 if not needed)
    simplifyTol = 20        # Tolerance for vector simplification
    # Number of bins used for vector creation when using the median vector creation. You should probably keep this 
    # smaller than the projBins, depending on cell density (<50%).
    medianBins = 35
    # Number of bins used for projection unto vector. Typical value for whole midgut is 100,
    # for R1-2 or R4-5 is 45, and R3 is 10
    projBins = np.linspace(0, 1, 100)
    # The name of the file used for normalizing between samples, i.e. R3 measurement point
    MPname = "MP"
    # Whether to use measurement point coordinates for normalization. If False,
    # the samples will be handled as perfectly aligned from beginning to end.
    useMP = True
    # Include secondary measurement point. Used to see e.g. proportional change.
    useSecMP = False
    # Name of secondary measurement point
    secMP = 'R45'
    # Whether to perform group-wise stat analysis.
    statistics = False
    alpha = 0.05
    # The name of the control group that the statistics are run against.
    cntrlGroup = "starv"    # CASE-SENSITIVE!
    
    # Additional data to be collected from channels. Key must be the data column 
    # label and the following string for searching the csv. If multiple files are
    # to be collected (e.g. intensities), the code expects an ID number after
    # the search string, separated by "_". E.g. "Intensity_Mean" => "Intensity_Mean_Ch=1"
    AddData = {"Area": "Area.csv",
               "Volume": "Volume.csv",
               "Intensity Mean": "Intensity_Mean"
               }
    # If set to true, replaces the above mentioned (AddData) ID numbers with an
    # alternative moniker as defined in channelID
    replaceID = True
    channelID = {"Ch=1": "Pros",
                 "Ch=2": "GFP",
                 "Ch=3": "SuH",
                 "Ch=4": "DAPI",}
    ###################################################################
    
    #### DATA COLLECTION AND PLOT SETTINGS ####
    # Whether to make plots for individual samples (does NOT include heatmaps)
    sampleplots = False
    # Whether to create heatmaps of cell counts
    heatmaps = False
    # Whether to find and plot distance means for Spots-objects, and to which channels.
    distmean = True     # Time-consuming depending on Distmean settings!
    # Find cell clusters
    clusters = False     # Time-consuming depending on Distmean settings!
    # Whether to plot intensity means for each channel. Requires "xyz_Intensity_Mean_Ch=X_xyz"
    # file in sample folder, where X is channel number. Must have column label "Intensity".
    intmean = False
    intFileNamer = "Intensity_Mean_Ch=" # search files with this + value or string from intChannels
    intChannels = [1,2,3,4]
    # Whether to plot distribution histograms of volumes, and to which channels.
    # Must have "xyz_Volume.csv" with column "Volume" in sample folder.
    volhist = False
    # Whether to plot distribution histograms of areas, and to which channels.
    # Must have "xyz_Area.csv" with column "Area" in sample folder.
    areahist = False
    # Whether to create boxplots of cell counts
    boxplots = False
    # Whether to create plots from the statistics. Requires statistics = True.
    statplots = False
    # Whether to create a localization map of the clusters.
    clustermap = False
    
    ## Figure save-format ##
    # Supported formats for the plot files: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    saveformat = "pdf"
    # The extension for file saving, so "period+X"
    figExt = ".pdf"
    
    ## Settings DISTMEAN and CLUSTER ##
    # Channels to which cluster analysis or average distance between cells are calculated.
    DistMeanChans = ["DAPI"]
    ClusterChans = ["GFP"]
    # Whether to find nearest points in another channel than self (DISTMIN)
    useTarget = False
    targetChan = "Both"
    # Distmean plot y-axis limits
    distYlim = 15   # Should be greater than maxDist
    distYmin = 0
    # Whether to filter by cell area and the minimum area of cell to include in DistMean and cluster
    filterArea = True
    areaIncl = 1
    # Maximum distance to find other cells. Greatly affects performance.
    maxDist = 25.0
    
    ## Settings VOLHIST ##
    # To which channels are volume averages calculated and histograms plotted
    volChans = ["DAPI"]     # Also determines all channels from which volumes are collected
    # Plot areahists with density instead of counts
    voldensity = True
    # The number of bins used in the volume distribution histograms
    volhistBins = 100
    # The X-limit for the volume histogram
    volhistLim = 1200
    # the smallest cell nucleus size included in the volume plots
    volInclSize = 1
    
    ## Settings AREAHIST ##
    # To which channels are area averages calculated and histograms plotted
    areaChans = ["DAPI"]    # Also determines all channels from which areas are collected
    # Plot areahists with density instead of counts
    density = True
    # The number of bins used in the area distribution histograms
    histBins = 100
    # The X-limit for the area histogram
    histLim = 1500
    # the smallest cell nucleus size included in the area plots
    areaInclSize = 1
    
    ## Settings BOXPLOT ##
    # Add secondary axis. Requires either volhist or areahist to be True.
    BPsec = True
    # Use area for secondary axis. False = volume.
    areas = True
    # the smallest cell nucleus size included in the volume plots
    boxInclSize = 1
    # Y-limit for the secondary Y-axis (Area) in box plots
    areaYlim = 500
    
    ## Settings STATPLOT ##
    # Whether to plot star significance to statplots
    stars = True
    
    ## AESTHETICS ##
    # set linewidth for the center line.
    lw = 0.5
    # Set font scale
    fontscale=1
    # Scatter plot marker size.
    scatterSize = 3
    # Size of the cell position plot canvas
    posSize=(15,2.5)
    # Settings for box plot fliers (markers outside boxes)
    markprops = dict(markersize=1)    
    # Color, heatmap color, value max (heatmap), Y-limit (other plots). Can add more.
    plotValues = {
            "DAPI": ['darkorange', 'YlOrRd', 75, 250],
            "Pros": ['r', 'OrRd', 16, 50],
            "Delta": ['b', 'Blues', 35, 70],
            "Delta+Prospero": ['y', 'YlOrBr', 12, 20],
            "GFP": ['g', 'Greens', 35, 150],
            "GFP+Pros": ['y', 'YlOrBr', 10, 10],
            "Pros+GFP": ['y', 'YlOrBr', 10, 40],
            "GFP+Delta": ['cyan', 'PuBu', 20, 60],
            "Suh+GFP": ['cyan', 'PuBu', 20, 60],
            "SuH": ['b', 'Blues', 35, 75],
            "randomsample": ['cyan', 'PuBu', 20, 60]
            }
    # The values for other channels not defined above
    plotValuesOther = ['darkorange', 'YlOrRd', 20, 20]
    # Samplegroup colors for the statistics plots.
    grpColors = { # Sample groups and respective colors
                "starv": 'tab:red',
                "Holidic": 'tab:blue',
                "starvgln": 'tab:orange',
                "Fed": 'tab:green',
                "ctrl": 'tab:red',
                "trp": 'tab:blue'
                 }
