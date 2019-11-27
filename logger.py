# -*- coding: utf-8 -*-
global LAM_logger
import logging, time

def setup_logger(name):
    global logFile, ctime, log_created
    ctime = time.strftime("%d%b%y_%H%M%S")
    from settings import settings as Sett
    logFile = Sett.workdir.joinpath("log_{}.txt".format(ctime))
    logger = get_logger(name)
    log_created = True
    return logger
    
def get_logger(name):
    logger = logging.getLogger(name)           
    logger.addHandler(get_handler())        
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger
    
def get_handler():   
    Formatter = logging.Formatter(
            "%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    file_handler = logging.FileHandler(logFile)
    file_handler.setFormatter(Formatter)
    file_handler.setLevel(logging.DEBUG)
    return file_handler

def log_print(self, msg="Message missing!", logtype='e'):
    """Prints information on the log file.
    Params: 
        msg = message to log
        logtype = type of log
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
        self.critical(msg)
    else:
        self.error(msg)
        

def print_settings(self):
    """Write settings into the log file."""
    from settings import settings as Sett
    with open(logFile, 'w') as file:
        file.write("Log time: {}\n".format(ctime))
        file.write("Analysis directory: {}\n\n".format(str(Sett.workdir)))
        pnames = ['Process', 'Count', 'Plots', 'Distances', 'Stats']
        psets = [Sett.process_samples, Sett.process_counts, 
                 Sett.Create_Plots, Sett.process_dists, 
                 Sett.statistics]
        primarymsg = ', '.join([pnames[i] for i, s in enumerate(psets) if 
                             s == True])
        file.write("Primary settings: {}\n".format(primarymsg))
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
        if Sett.process_counts:
            file.write("--- Count Settings ---\n")
            if Sett.useMP:
                MPmsg = "Using MP with label {}.\n".format(Sett.MPname)
            else:
                MPmsg = "Not using MP.\n"            
            file.write(MPmsg)
            file.write("Number of bins: {}\n".format(len(Sett.projBins)))
            file.write("-Additional data-\n")
            addD = Sett.AddData
            addtypes = ', '.join(["{}".format(key)for key in sorted(list(
                                                            addD.keys()))])
            file.write("Types: {}\n".format(addtypes))  
        if Sett.process_dists:
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
            if Sett.Find_Clusters:
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
        if Sett.statistics:
            file.write("--- Statistics Settings ---\n")
            file.write("Control group: {}\n".format(Sett.cntrlGroup))
            file.write("Types: versus={}; total={}\n".format(
                                        Sett.stat_versus, Sett.stat_total))
            if Sett.windowed:
                file.write("windowed: trail={}; lead={}\n".format(
                                        Sett.stat_versus, Sett.stat_total))    
        if Sett.Create_Plots:
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
        file.write("#" * 50 + "\n")
        file.write(' - '*3+"Time"+' - '*5+"Module"+' - '*2+"Level"+' - '*3+"Message")
        file.write("\n" + "-" * 50 + "\n")