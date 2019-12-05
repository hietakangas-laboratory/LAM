# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen

"""
import logging, time
loggers = []

def setup_logger(name=None, new=True):
    """Sets up variables for the logger when run starts. 
        Param. name: the calling module."""
    # Create variables for the creation of logfile
    global logFile, ctime, log_created
    ctime = time.strftime("%d%b%y_%H%M%S") # Start time for the run
    from settings import settings as Sett
    logFile = Sett.workdir.joinpath("log_{}.txt".format(ctime)) # filepath
    if new == True:
        logger = get_logger(name) # Call for logger-object creation    
        log_created = True # Create variable to indicate log has been created
        return logger
    
def get_logger(name):
    """Get module-specific logger.
        Param. name: the calling module."""
    logger = logging.getLogger(name)# Create logger      
    logger.addHandler(_get_handler())# Call for handler that passes messages
    logger.propagate = False # No propagation as modules call individually
    logger.setLevel(logging.DEBUG) # Set logs of all level to be shown
    if name not in loggers:
        loggers.append(name)
    return logger
    
def _get_handler():
    """Create message handler in conjunction with get_logger()"""
    # Create format for log messages
    Formatter = logging.Formatter(
            "%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    # create handler and assign logfile's path
    file_handler = logging.FileHandler(logFile)
    file_handler.setFormatter(Formatter) # Set format of log messages
    file_handler.setLevel(logging.DEBUG) # Set logs of all level to be shown
    return file_handler

def Close():
    for lgr in loggers:
        logger = logging.getLogger(lgr)
        for handler in logger.handlers:
            handler.close()
        logger.handlers = []
    
def Update():
    setup_logger(new=False)
    for lgr in loggers:
        logger = logging.getLogger(lgr)
        logger.addHandler(_get_handler())

def log_Shutdown():
    logging.shutdown()

def logprint(self, msg="Missing", logtype='e'):
    """Prints information of different level to the log file.
    Params: 
        msg: message to log
        logtype: type of log
            i = info; w = warning; d = debug; c = critical; e = error; 
            ex = exception
    """
    if logtype == 'i':
        self.info(msg)
    elif logtype == 'w':
        self.warning(msg)
    elif logtype == 'd':
        self.debug(msg)
    elif logtype == 'c':
        self.critical(msg)
    elif logtype == 'ex':
        self.exception(msg)
    else:
        self.error(msg)       

def print_settings(self):
    """Write settings into the log file."""
    from settings import settings as Sett
    with open(logFile, 'w') as file: # Write into the logfile
        file.write("Log time: {}\n".format(ctime))
        file.write("Analysis directory: {}\n\n".format(str(Sett.workdir)))
        pnames = ['Process', 'Count', 'Plots', 'Distances', 'Stats']
        psets = [Sett.process_samples, Sett.process_counts, 
                 Sett.Create_Plots, Sett.process_dists, 
                 Sett.statistics]
        primarymsg = ', '.join([pnames[i] for i, s in enumerate(psets) if 
                             s == True])
        file.write("Primary settings: {}\n".format(primarymsg))
        if any([Sett.process_samples, Sett.process_counts, Sett.process_dists]):
            if Sett.useMP:
                    MPmsg = "Using MP with label {}.\n".format(Sett.MPname)
            else:
                MPmsg = "Not using MP.\n" 
        # If creating vectors, get and print related settings
        if Sett.process_samples:
            file.write("--- Process Settings ---\n")
            file.write("Vector channel: {}\n".format(Sett.vectChannel))
            vectordict = {'Simplify tolerance': Sett.simplifyTol}
            if Sett.SkeletonVector:
                file.write("Creation type: Skeleton\n")
                vectordict.update({'Resize': Sett.SkeletonResize, 
                                   'Find distance': Sett.find_dist, 
                                   'Dilation iterations': Sett.BDiter, 
                                   'Smoothing': Sett.SigmaGauss})
            else:
                file.write("Creation type: Median\n")
                vectordict.update({'Median bins': Sett.medianBins})
            keys = sorted(list(vectordict.keys()))
            file.write(', '.join(["{}: {}".format(key, vectordict.get(key))\
                                  for key in keys]))
            file.write("\n")
        if Sett.process_counts: # Count settings
            file.write("--- Count Settings ---\n")         
            file.write(MPmsg)
            file.write("Number of bins: {}\n".format(len(Sett.projBins)))
            file.write("-Additional data-\n")
            addD = Sett.AddData
            addtypes = ', '.join(["{}".format(key)for key in sorted(list(
                                                            addD.keys()))])
            file.write("Types: {}\n".format(addtypes))  
        if Sett.process_dists: # Distance settings
            file.write("--- Distance Settings ---\n")
            if Sett.Find_Distances:
                file.write("-Nearest Distance-\n")
                distD = {'Channels': Sett.Distance_Channels,
                        'Maximum distance': Sett.maxDist}
                if Sett.use_target:
                    distD.update({'Target channel': Sett.target_chan})
                if Sett.Vol_inclusion > 0:
                    if not Sett.incl_type:
                        inclmsg = 'Smaller than {}'.format(
                                                        Sett.Vol_inclusion)
                    else:
                        inclmsg = 'Greater than {}'.format(
                                                        Sett.Vol_inclusion)
                    distD.update({'Cell inclusion': inclmsg})
                keys = sorted(list(distD.keys()))
                file.write(', '.join(["{}: {}".format(key, distD.get(key))\
                                  for key in keys]))
                file.write("\n")
            if Sett.Find_Clusters: # Cluster settings
                file.write("-Clusters-\n")
                clustD = {'Channels': Sett.Cluster_Channels,
                        'Maximum distance': Sett.Cl_maxDist,
                        'Minimum cluster': Sett.Cl_min,
                        'Maximum cluster': Sett.Cl_max}
                if Sett.Vol_inclusion > 0:
                    if not Sett.Cl_incl_type:
                        inclmsg = 'Smaller than {}'.format(
                                                    Sett.Cl_Vol_inclusion)
                    else:
                        inclmsg = 'Greater than {}'.format(
                                                    Sett.Cl_Vol_inclusion)
                    clustD.update({'Cell inclusion': inclmsg})
                keys = sorted(list(clustD.keys()))
                file.write(', '.join(["{}: {}".format(key, clustD.get(key))\
                                  for key in keys]))
                file.write("\n")       
        if Sett.statistics: # Statistics settings
            file.write("--- Statistics Settings ---\n")
            file.write("Control group: {}\n".format(Sett.cntrlGroup))
            file.write("Types: versus={}; total={}\n".format(
                                        Sett.stat_versus, Sett.stat_total))
            if Sett.windowed:
                file.write("windowed: trail={}; lead={}\n".format(
                                        Sett.trail, Sett.lead))    
        if Sett.Create_Plots: # Plotting settings
            file.write("--- Plot Settings ---\n")
            plotnames = ['Channels', 'Additional', 'Pair', 'Heatmap', 
                         'Distribution', 'Statistics', 'ChanVSAdd', 
                         'AddVSAdd']
            plots = [Sett.Create_Channel_Plots,Sett.Create_AddData_Plots,
                    Sett.Create_Channel_PairPlots,Sett.Create_Heatmaps,
                Sett.Create_Distribution_Plots,Sett.Create_Statistics_Plots,
                    Sett.Create_ChanVSAdd_Plots,Sett.Create_AddVSAdd_Plots]
            plotmsg = ', '.join([plotnames[i] for i, s in enumerate(plots)\
                                 if  s == True])
            file.write("Plot types: {}\n".format(plotmsg))
            file.write("Drop outliers: {}\n".format(Sett.Drop_Outliers))
        # Create header for the messages sent during the analysis
        file.write("="*75 + "\n")
        msg= ' '*9+"-Time-"+' '*12+"-Module-"+' '*6+"-Level-"+' '*9+"-Message-"
        file.write(msg)
        file.write("\n" + "-" * 75 + "\n")