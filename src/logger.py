# -*- coding: utf-8 -*-
"""
LAM-module for establishing logging.

Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen

"""

# Standard libraries
import logging
import time
import sys

# A list of logger names used by the modules. Used for clearing of handlers.
loggers = ['src.run', 'src.process', 'src.analysis', 'src.interface', 'src.plot', 'src.system', 'src.plotfuncs',
           'src.border_detect']
# Needed variables
logFile = ""
ctime = time.strftime("%d%b%y_%H%M%S")
log_created = False


def setup_logger(name=None, new=True):
    """
    Set up variables for the logger when run starts.

    Args:
    ----
        name - the calling module.
        new - create new logger
    """
    # Create variables for the creation of logfile
    global logFile, ctime, log_created
    ctime = time.strftime("%d%b%y_%H%M%S")  # Start time for the run
    from src.settings import settings as Sett

    # filepath:
    logFile = str(Sett.workdir.joinpath("log_{}.txt".format(ctime)))

    if new is True:  # If setting up new logger
        logger = get_logger(name)  # Call for logger-object creation
        log_created = True  # Create variable to indicate log has been created
        return logger
    return


def get_logger(name):
    """Get module-specific logger."""
    logger = logging.getLogger(name)  # Create logger
    if not logger.handlers:
        logger.addHandler(_get_handler())  # Get handler that passes messages
    logger.propagate = False  # No propagation as modules call individually
    logger.setLevel(logging.DEBUG)  # Set logs of all level to be shown
    if name not in loggers:
        loggers.append(name)
    return logger


def _get_handler():
    """Create message handler in conjunction with get_logger()."""
    # Create format for log messages
    formatting = "%(asctime)-25s %(name)-20s %(levelname)-10s %(message)s"
    # formatting = "{:s<25s} {:s<20s} {:s<10s} {:s<}".format(*strings)
    #formatting = "%(asctime)s %(name)s %(levelname)s %(message)s"
    Formatter = logging.Formatter(formatting)

    # create handler and assign logfile's path
    file_handler = logging.FileHandler(logFile)
    file_handler.setFormatter(Formatter)  # Set format of log messages
    file_handler.setLevel(logging.DEBUG)  # Set logs of all level to be shown
    return file_handler


def create_loggers():
    for module in loggers:
        mod = sys.modules.get(module)
        if mod.LAM_logger is not None:
            continue
        setattr(mod, 'LAM_logger', get_logger(module))


def Close():
    """Close all created loggers."""
    for lgr in loggers:
        logger = logging.getLogger(lgr)
        for handler in logger.handlers:
            handler.close()
        logger.handlers = []
        mod = sys.modules.get(lgr)
        setattr(mod, 'LAM_logger', None)


def Update():
    """Update current loggers."""
    setup_logger(new=False)
    for lgr in loggers:
        logger = logging.getLogger(lgr)
        logger.addHandler(_get_handler())


def log_Shutdown():
    """Shut down all logging elements."""
    Close()
    logging.shutdown()


def logprint(self, msg="Missing", logtype='e'):
    """
    Print information of different levels to the log file.

    Args:
    ----
        msg - message to log
        logtype - type of log
            i = info, w = warning, d = debug, c = critical, e = error,
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


def print_settings():
    """Write settings into the log file."""
    from src.settings import settings as Sett
    with open(logFile, 'w') as file:  # Write into the logfile
        file.write("Log time: {}\n".format(ctime))
        file.write("Analysis directory: {}\n\n".format(str(Sett.workdir)))
        pnames = ['Process', 'Count', 'Plots', 'Distances', 'Stats']
        psets = [Sett.process_samples, Sett.process_counts, Sett.Create_Plots, Sett.process_dists, Sett.statistics]
        primarymsg = ', '.join([pnames[i] for i, s in enumerate(psets) if s is True])
        file.write("Primary settings: {}\n".format(primarymsg))
        if any([Sett.process_samples, Sett.process_counts, Sett.process_dists]):
            if Sett.useMP:
                MPmsg = "Using MP with label {}.\n".format(Sett.MPname)
            else:
                MPmsg = "Not using MP.\n"

        if Sett.border_detection or Sett.measure_width:
            sn = ['Widths', 'Borders']
            ss = [Sett.measure_width, Sett.border_detection]
            msg = ', '.join([sn[i] for i, s in enumerate(ss) if s is True])
            if Sett.border_detection:
                msg = msg + f', Border channel = {Sett.border_channel}'
            file.write("Secondary: {}\n".format(msg))

        # If creating vectors, get and print related settings
        if Sett.process_samples:
            file.write("--- Process Settings ---\n")
            file.write("Vector channel: {}\n".format(Sett.vectChannel))
            vectordict = {'Simplify tolerance': Sett.simplifyTol}
            if Sett.SkeletonVector:
                file.write("Creation type: Skeleton\n")
                vectordict.update({'Resize': Sett.SkeletonResize, 'Find distance': Sett.find_dist,
                                   'Dilation iterations': Sett.BDiter, 'Smoothing': Sett.SigmaGauss})
            else:
                file.write("Creation type: Median\n")
                vectordict.update({'Median bins': Sett.medianBins})
            keys = sorted(list(vectordict.keys()))
            file.write(', '.join(["{}: {}".format(key, vectordict.get(key)) for key in keys]))
            file.write("\n")

        if Sett.process_counts:  # Count settings
            file.write("--- Count Settings ---\n")
            file.write(MPmsg)
            file.write("Number of bins: {}\n".format(Sett.projBins))
            file.write("-Additional data-\n")
            addD = Sett.AddData
            addtypes = ', '.join(["{}".format(key) for key in sorted(list(addD.keys()))])
            file.write("Types: {}\n".format(addtypes))

        if Sett.process_dists:  # Distance settings
            file.write("--- Distance Settings ---\n")
            if Sett.find_distances:
                file.write("-Nearest Distance-\n")
                distD = {'Channels': Sett.distance_channels, 'Maximum distance': Sett.max_dist}
                if Sett.use_target:
                    distD.update({'Target channel': Sett.target_chan})
                if Sett.inclusion > 0:
                    if not Sett.incl_type:
                        inclmsg = f'Smaller than {Sett.inclusion}'
                    else:
                        inclmsg = f'Greater than {Sett.inclusion}'
                    distD.update({'Cell inclusion': inclmsg})
                keys = sorted(list(distD.keys()))
                file.write(', '.join(["{}: {}".format(key, distD.get(key)) for key in keys]))
                file.write("\n")

            if Sett.find_clusters:  # Cluster settings
                file.write("-Clusters-\n")
                clustD = {'Channels': Sett.Cluster_Channels, 'Maximum distance': Sett.cl_max_dist,
                          'Minimum cluster': Sett.cl_min, 'Maximum cluster': Sett.cl_max}
                if Sett.inclusion > 0:
                    if not Sett.cl_incl_type:
                        inclmsg = f'Smaller than {Sett.cl_inclusion}'
                    else:
                        inclmsg = f'Greater than {Sett.cl_inclusion}'
                    clustD.update({'Cell inclusion': inclmsg})
                keys = sorted(list(clustD.keys()))
                file.write(', '.join([f"{key}: {clustD.get(key)}" for key in keys]))
                file.write("\n")

        if Sett.statistics:  # Statistics settings
            file.write("--- Statistics Settings ---\n")
            file.write(f"Control group: {Sett.cntrlGroup}\n")
            txt = f"Type: versus={Sett.stat_versus}; total={Sett.stat_total}\n"
            file.write(txt)
            if Sett.windowed:
                file.write(f"windowed: trail={Sett.trail}; lead={Sett.lead}\n")

        if Sett.Create_Plots:  # Plotting settings
            file.write("--- Plot Settings ---\n")
            plotnames = ['Channels', 'Additional', 'Pair', 'Heatmap', 'Distribution', 'Borders', 'Sample borders',
                         'Width', 'Statistics', 'ChanVSAdd', 'AddVSAdd']
            plots = [Sett.Create_Channel_Plots, Sett.Create_AddData_Plots, Sett.Create_Channel_PairPlots,
                     Sett.Create_Heatmaps, Sett.Create_Distribution_Plots, Sett.Create_Border_Plots, Sett.plot_samples,
                     Sett.plot_width, Sett.Create_Statistics_Plots, Sett.Create_ChanVSAdd_Plots,
                     Sett.Create_AddVSAdd_Plots]
            plotmsg = ', '.join([plotnames[i] for i, s in enumerate(plots) if s is True])
            file.write("Plot types: {}\n".format(plotmsg))
            file.write("Drop outliers: {}\n".format(Sett.Drop_Outliers))

        # Create header for the messages sent during the analysis
        file.write("="*75 + "\n")
        "%(asctime)-25s %(name)-20s %(levelname)-10s %(message)s"
        msg = "{:<25} {:<19}{:^10}\t\t{}".format('-Time-', '-Module-', '-Level-', '-Message-')
        file.write(msg)
        file.write("\n" + "-" * 75 + "\n")
