# -*- coding: utf-8 -*-
import numpy as np, pathlib as pl
class settings:
    """ A class for holding all user-end settings for the analysis."""
    ######################## PRIMARY SETTINGS ########################
    GUI = True  # Use graphical user interface (True / False)
    
    # DEFINE PATH TO ANALYSIS FOLDER:
    # (Use input r'PATH' where PATH is your path)
    workdir = pl.Path(r'P:\h919\hietakangas\Arto\21.01.2019 Statistics')
    # Whether to gather raw data and create vectors. If False, expects to find 
    # pre-created datafiles in the Analysis Data directory, i.e. a previous 
    # full run has been made, and there has been no edits to the data files.
    process_samples = True
    # Whether to count and normalize data. If set to False, expect all data to 
    # be in place. Can be used to e.g. create additional plots faster.
    process_counts = True
    # Whether to compute average distances and clusters.
    process_dists = True
    # Set True/False to set all plotting functionalities ON/OFF
    Create_Plots = True     # ON / OFF switch for plots
    ##################################################################
    
    #-#-#-#-#-#-#-#-# VECTOR CREATION & PROJECTION #-#-#-#-#-#-#-#-#-#
    # The channel based on which the vector is created
    vectChannel = "DAPI"
    # Number of bins used for projection unto vector (the second value).
    projBins = np.linspace(0, 1, 80)
    
    # Make vector by creating binary image and then skeletonizing. If False, 
    # vector is created by finding middle point between smallest and largest 
    # Y-axis position in bin.
    SkeletonVector = False
    SkeletonResize = 0.2    # Binary image resize. Keep at steps of ten.
    find_dist = 30      # Find distance of pixels in skeleton vector creation
    BDiter = 0           # Binary dilation iterations (set to 0 if not needed)
    SigmaGauss = 1       # Sigma for gaussian smoothing (set to 0 if not needed)
    simplifyTol = 50     # Tolerance for vector simplification.
    # Number of bins used for vector creation when using the median vector 
    # creation. You should probably keep this smaller than the projBins, 
    # depending on cell density (e.g. <50%).
    medianBins = 25
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

    ## MEASUREMENT POINTS ##
    # Whether to use measurement point coordinates for normalization. If False,
    # the samples will be handled as perfectly aligned from beginning to end.
    useMP = True
    # The name of the file used for normalizing between samples, i.e. anchoring
    MPname = "MP"
    # Include secondary measurement point. Used to see e.g. proportional change.
    useSecMP = False    # SECMP NOT PROPERLY IMPLEMENTED
    # Name of secondary measurement point
    secMP = 'R45'
    
    
    ### DATA GATHERING ###
    header_row = 2  # On which row does the data have its header (starts from 0)
    # Additional data to be collected from channels. Key (the first string before 
    # ':') must be the data column label and the following string for searching 
    # the csv-file containing the wanted data. If multiple files are to be 
    # collected (e.g. intensities), the code expects the data column to have an 
    # ID number after the search string, separated by an underscore "_". 
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
                 "Ch=4": "DAPI",}
    ###################################################################
    
    ### ANALYSIS OPTIONS ###
    ## DISTANCE MEANS ##
    # Find nearest cell of each cell. Distance estimation is performed for all 
    # channels in Distance_Channels list. If use target is True, the nearest
    # cell is found on the channel defined by target_chan, otherwise they are
    # found within the channel undergoing analysis.
    Find_Distances = False
    Distance_Channels = ["GFP"]
    use_target = True
    target_chan = "Pros"
    # The maximum distance the nearest cell will be looked at. Increase is
    # computationally expensive, depending on the size of the dataset and the 
    # density of cells.
    maxDist = 30    # Radius around the cell
    # Whether to look only at cells of certain size. Default is to include cells 
    # smaller than Vol_inclusion. If cells of greater volume are wanted, 
    # designate incl_type to be 'greater'. Otherwise, the string can be left empty.
    Vol_inclusion = 0    # Set to zero if not wanted.
    incl_type = ""
    
    ## CLUSTERS ##
    # Whether to compute clusters
    Find_Clusters = False
    Cluster_Channels = ["GFP"]
    Cl_maxDist = 20         # Radius around the cell
    Cl_Vol_inclusion = 0    # Set to zero if not wanted.
    Cl_incl_type = ""       # Same as above in Find_Distances
    Cl_min = 3
    Cl_max = 50
    
    ### STATISTICS OPTIONS ###
    # Whether to perform group-wise stat analysis.
    statistics = True
    stat_versus = True
    stat_total = True
    windowed = True
    trail = 1
    lead = 1
    ylim = 25               # negative log2 y-limit
    alpha = 0.05            # for rejection of H_0, applies to statistics files
    # Plots
    stars = True # Make P-value stars to the plot (*:<0.05 **:<0.01 ***:<0.001)
    fill = True
    negLog2 = True # Forces stars to be False
    # The name of the control group that the statistics are run against.
    cntrlGroup = "starv"
    
    ### PLOTTING OPTIONS ###  
    Create_Channel_Plots = True
    Create_AddData_Plots = True     # Plots also nearest distance & clusters
    Create_Channel_PairPlots = True
    Create_Heatmaps = True
    Create_Statistics_Plots = True  # requires statistics to be True
    # Variable vs. variable plots:
    Create_ChanVSAdd_Plots = False
    Create_AddVSAdd_Plots = False
    vs_channels = ['DAPI', 'Pros', 'SuH', 'GFP']
    vs_adds = ['Intensity Mean']
    
    # Whether to drop outliers from plots ONLY, and the standard deviation limit
    # for considering what is an outlier.
    Drop_Outliers = True
    dropSTD = 3
    
    # Gives values some coordinate-shift in channel pairplots. Useful in presenting
    # the data, as it is discrete; most of the data would be hidden under others.
    # Doesn't affect the underlying data.
    plot_jitter = True
    
    ## Figure save-format ##
    # Supported formats for the plot files: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    saveformat = "pdf"
    
    seaborn_style = 'ticks'   # Different styles of plots, e.g. background
    # Available styles #
    # write to the console the following lines
    # 1.    import matplotlib.style as style
    # 2.    style.available
    
    seaborn_context = 'talk'  # Defines plot object sizes, i.e. text
    # size order small to large: 'paper', 'talk', 'poster'
    
    # Define colors used for sample groups (xkcd colors: 'https://xkcd.com/color/rgb/')
    palette_colors = ['orange yellow', 'aqua marine', 'tomato', 'dark lime', 'tan brown']