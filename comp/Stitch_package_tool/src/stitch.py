import csv
import os
import shutil
import logging
from ij import IJ


def create_directories(aurox_dir_path):
    """
        Creates all directories needed for the other functions and stores their
        paths in a list for access by other functions. Returns this list.
    """
    
    current_dirs = []

    new_dir_names = ["FUSED_TIFFS", "PROCESSED_TIFFS", "C_OME_FILES",
                     "ORIG_FUSED_TIFFS"]

    abs_dirpath = os.path.abspath(aurox_dir_path)
    new_dirs_path = os.path.dirname(abs_dirpath)

    for n in new_dir_names:
        n_path = os.path.join(new_dirs_path, n)
        if not os.path.isdir(n_path):
            logging.info("Creating directory" + n)
            os.mkdir(n_path)
        current_dirs.append(n_path)

    return current_dirs


def move_ome_files_out(aurox_dir_path, current_dirs):
    """
        Moves companion ome files to the C_OME_FILES directory to avoid
        overlap calculation issues by the Grid/Collection plugin due to
        incorrect reading of metadata.
    """

    omeloc_dict = {}

    c_ome_path, found_ome = ome_checker(aurox_dir_path)

    if found_ome:
        prefix = os.path.basename(c_ome_path)
        ome_folder_file_path = current_dirs[2]
        move_dir_ome = os.path.join(ome_folder_file_path, prefix)
        shutil.move(c_ome_path, move_dir_ome)
        omeloc_dict[c_ome_path] = move_dir_ome
    else:
        logging.info("No companion.ome file in " + aurox_dir_path)

    return omeloc_dict, c_ome_path


def move_ome_file_in(omeloc_dict):
    """
        Returns companion.ome files to their original locations using the path
        information stored in the omeloc_dict
    """

    logging.info("Returning companion ome files to original location")

    try:
        for key in omeloc_dict:
            shutil.move(omeloc_dict[key], key)
    except Exception as e:
        logging.exception(str(e))


def ome_checker(aurox_dir_path):
    """
        Checks for companion.ome file in the tiff series image directory.
        Returns the path of the companion.ome file.
    """
    if not len(os.listdir(aurox_dir_path)) == 0:
        for file in os.listdir(aurox_dir_path):
            if file.endswith("companion.ome"):
                c_ome_p = os.path.join(aurox_dir_path, file)
                c_ome_path = c_ome_p.replace("\\", "//")
                found_ome = True
                return c_ome_path, found_ome
            else:
                c_ome_path = ""
                found_ome = False
                return c_ome_path, found_ome
    else:
        c_ome_path = ""
        found_ome = False
        return c_ome_path, found_ome


def calculate_overlap(aurox_dir_path):
    """
        Calculates overlap between images using imageJ stitching plugin. This
        gets saved as TileConfiguration.registered.txt in the tiff series
        directory.
    """

    abs_dirpath = os.path.abspath(aurox_dir_path)
    # Calculate overlap

    IJ.run("Grid/Collection stitching",
           "type=[Positions from file] order=[Defined by TileConfiguration] "
           "directory=[%s] layout_file=TileConfiguration.txt "
           "fusion_method=[Do not fuse images (only write TileConfiguration)] "
           "regression_threshold=0.30 max/avg_displacement_threshold=2.50 "
           "absolute_displacement_threshold=3.50 "
           "compute_overlap "
           "computation_parameters=[Save computation time (but use more RAM)] "
           "image_output=[Fuse and display]" % abs_dirpath)


def linear_blending(aurox_dir_path, current_dirs, c_ome_path, yes_orig):
    """
        Linear blend images. It saves two images:
        A fused tiff from TileConfiguration.fixed.txt with Z axis set to 0,
        saved into FUSED_TIFFS and a separate fused tiff from
        TileConfiguration.txt with original positions from positions.csv file,
        saved into ORIG_FUSED_TIFFS
    """

    fusloc_dict = {}

    abs_dirpath = os.path.abspath(aurox_dir_path)
    prefix = os.path.basename(aurox_dir_path)

    sav_fu = "%s/%s.tiff" % (current_dirs[0], prefix)
    sav_orig = "%s/%s.orig.tiff" % (current_dirs[3], prefix)

    # Creates both images if 'Create image from original positions?' was
    # checked.
    if yes_orig:
        # Linear Blending
        IJ.run("Grid/Collection stitching",
               "type=[Positions from file] "
               "order=[Defined by TileConfiguration]"
               " directory=[%s] layout_file=TileConfiguration.fixed.txt "
               "fusion_method=[Linear Blending] "
               "regression_threshold=0.30 max/avg_displacement_threshold=2.50 "
               "absolute_displacement_threshold=3.50 "
               "computation_parameters=[Save computation time (but use more RAM)] "
               "image_output=[Fuse and display]" % abs_dirpath)

        IJ.saveAs("Tiff", sav_fu)
        IJ.run("Close All")

        # Linear Blending
        IJ.run("Grid/Collection stitching",
               "type=[Positions from file] "
               "order=[Defined by TileConfiguration] "
               "directory=[%s] layout_file=TileConfiguration.txt "
               "fusion_method=[Linear Blending] "
               "regression_threshold=0.30 max/avg_displacement_threshold=2.50 "
               "absolute_displacement_threshold=3.50 "
               "computation_parameters=[Save computation time (but use more RAM)] "
               "image_output=[Fuse and display]" % abs_dirpath)

        IJ.saveAs("Tiff", sav_orig)
        IJ.run("Close All")

    # Creates only a single image with the calculated overlap and new positions
    else:
        # Linear Blending
        IJ.run("Grid/Collection stitching",
               "type=[Positions from file] "
               "order=[Defined by TileConfiguration] "
               "directory=[%s] layout_file=TileConfiguration.fixed.txt "
               "fusion_method=[Linear Blending] "
               "regression_threshold=0.30 max/avg_displacement_threshold=2.50 "
               "absolute_displacement_threshold=3.50 "
               "computation_parameters=[Save computation time (but use more RAM)] "
               "image_output=[Fuse and display]" % abs_dirpath)

        IJ.saveAs("Tiff", sav_fu)
        IJ.run("Close All")

    fusloc_dict[sav_fu] = [c_ome_path, current_dirs[1]]

    return fusloc_dict


def ome_only_blending(aurox_dir_path, current_dirs, c_ome_path):
    """
        Linear blends images. It saves one image:
        A fused image using the position data from the companion.ome metadata
    """

    fusloc_dict = {}

    prefix = os.path.basename(aurox_dir_path)

    sav_fu = "%s/%s.tiff" % (current_dirs[0], prefix)

    # Linear Blending
    IJ.run("Grid/Collection stitching",
           "type=[Positions from file] order=[Defined by image metadata] "
           "browse=[%s] "
           "multi_series_file=[%s] "
           "fusion_method=[Linear Blending] "
           "regression_threshold=0.30 max/avg_displacement_threshold=2.50 "
           "absolute_displacement_threshold=3.50 "
           "compute_overlap "
           "increase_overlap=0 "
           "computation_parameters=[Save computation time (but use more RAM)] "
           "image_output=[Fuse and display]" % (c_ome_path, c_ome_path))

    IJ.saveAs("Tiff", sav_fu)
    IJ.run("Close All")

    fusloc_dict[sav_fu] = [c_ome_path, current_dirs[1]]

    return fusloc_dict


def list_tiff_files(aurox_dir_path):
    """
        Lists all *.tiff files in tiff series directory
    """
    tiff_file_names = []

    for file in os.listdir(aurox_dir_path):
        if file.endswith(".tiff"):
            tiff_file_names.append(file)

    return tiff_file_names


def read_positions(aurox_dir_path, dir_name):
    """
        Reads the positions content in the positions.csv file.
        Read dir/{dir}.positions.csv. Returns a list of the content.
    """
    positions_name = '%s/%s.positions.csv' % (aurox_dir_path, dir_name)
    with open(positions_name, 'rb') as f:
        reader = csv.reader(f)
        positions_content = list(reader)

    return positions_content


def create_tile_configuration(tiff_files_names, tiff_img_positions, dir,
                              mult_const):
    """
        Creates file TileConfiguration.txt from the positions.csv file.
    """

    tile_conf_path = '%s/TileConfiguration.txt' % dir
    with open(tile_conf_path, 'w') as f:
        write_header(f)

        tiff_files_names.sort()
        tiff_files_names.sort(key=len)
        l = list(zip(tiff_files_names, tiff_img_positions))
        for (name, pos) in l:
            h = abs(int(pos[1])) * float(mult_const)
            w = abs(int(pos[2])) * float(mult_const)
            f.write('%s     ;    ;    (    %s,    %s,    0.0    )\n' % (
                name, h, w))


def set_tile_configuration_z_axis_to_zero(dir):
    """
        Using the calculated overlap file (TileConfiguration.registered.txt)
        for linear blending causes z-axis overlap issues.
        This function creates a further tile configuration
        (TileConfiguration.fixed.txt) that has the calculated overlap positions
        for x and y but the z axis is reset to 0.
    """

    configuration_name = '%s/TileConfiguration.registered.txt' % dir
    conf_fixed_name = '%s/TileConfiguration.fixed.txt' % dir

    with open(configuration_name, 'rb') as f:
        with open(conf_fixed_name, 'w') as fw:
            write_header(fw)

            line_number = 1
            line = f.readline()
            while line:
                if line_number > 4:
                    #
                    (name, _, positions) = line.split(';')
                    name = name.strip()
                    position_list = positions.strip()[1:-1].split(',')

                    fw.write(
                        '%s     ;    ;    (    %s,    %s,    0.0    )\n' % (
                            name, position_list[0], position_list[1]))
                    #

                line = f.readline()
                line_number += 1


def write_header(fw):
    """
        Writes header of TileConfiguration.txt files.
    """
    fw.write(
"""# Define the number of dimensions we are working on
dim = 3

# Define the image coordinates\n""")


def dir_checker(aurox_dir_path):
    """
        Checks to make sure there is a positions.csv file as well as more than
        one .tiff  file available in the directory. If so it sees this as an
        image directory and returns True.
    """
    value1 = 0
    value2 = 0

    for file in os.listdir(aurox_dir_path):
        if file.endswith('.tiff'):
            value1 += 1
        if file.endswith('positions.csv'):
            value2 += 1
    if value1 > 1 and value2 > 0:
        status = True
    else:
        status = False

    return status


def process_aurox_image(aurox_dir_path, dir_name, yes_orig, mult_const):

    """
        Checks the directory is an image directory using dir_checker.
        Then proceeds with sticthing. Returns fusloc_dict and omeloc_dict to
        main(), repectively for final macro processing and returning
        companion.ome files to original locations.
    """

    logging.info('Processing %s' % dir_name)
    check_status = dir_checker(aurox_dir_path)

    if check_status:

        current_dirs = create_directories(aurox_dir_path)

        logging.info('Temporarily relocating companion ome files')
        omeloc_dict, c_ome_path = move_ome_files_out(aurox_dir_path,
                                                     current_dirs)

        logging.info('Reading configurations')
        tiff_files_names = list_tiff_files(aurox_dir_path)
        tiff_img_positions = read_positions(aurox_dir_path, dir_name)

        logging.info('Creating TileConfiguration.txt')
        create_tile_configuration(tiff_files_names, tiff_img_positions,
                                  aurox_dir_path, mult_const)

        logging.info('Calculating overlap')
        calculate_overlap(aurox_dir_path)

        logging.info('Setting TileConfiguration.txt Z axis to 0')
        set_tile_configuration_z_axis_to_zero(aurox_dir_path)

        logging.info('Creating fused/stitched image/s')
        fusloc_dict = linear_blending(aurox_dir_path, current_dirs, c_ome_path,
                                      yes_orig)

        return fusloc_dict, omeloc_dict

    else:
        logging.info('%s directory does not contain an aurox image' % dir_name)
        fusloc_dict = {}
        omeloc_dict = {}
        return fusloc_dict, omeloc_dict


def process_aurox_ome_only(aurox_dir_path, dir_name):
    """
    Checks for a companion.ome file and uses this for linear blending.
    """

    logging.info('Processing %s' % dir_name)
    c_ome_path, found_ome = ome_checker(aurox_dir_path)

    if found_ome:

        current_dirs = create_directories(aurox_dir_path)

        fusloc_dict = ome_only_blending(aurox_dir_path, current_dirs,
                                        c_ome_path)

        logging.info('Creating fused/stitched image/s')

        return fusloc_dict

    else:
        logging.info('%s directory does not contain a companion.ome file'
                     % dir_name)
        fusloc_dict = {}
        return fusloc_dict


def ij_macro_processor(fus_tiffs_dict, py_file_loc):
    """
    Uses a specified ImageJ macro to process the fused tiffs located in the
    FUSED_TIFFS directory and then saves the processed output in
    PROCESSED_TIFFS directory.
    """

    stitch_path = os.path.dirname(os.path.abspath(py_file_loc))

    for key in fus_tiffs_dict:

        if fus_tiffs_dict[key][0] == '':
            logging.info('Companion file not available for' + key)
            logging.info('Running imageJ macro on ' + key)
            IJ.runMacroFile(stitch_path + r"\stitch_macro.ijm")
            IJ.saveAs("tiff",
                      fus_tiffs_dict[key][1] + '/' + os.path.basename(key))
            IJ.run("Close All")
        else:
            logging.info('Running imageJ macro on ' + key)
            IJ.runMacroFile(stitch_path + r"\stitch_macro.ijm",
                            key + "<>" + fus_tiffs_dict[key][0])
            IJ.saveAs("tiff", fus_tiffs_dict[key][1] + '/'
                      + os.path.basename(key))
            IJ.run("Close All")


def ome_transfer(root_dir_path):
    """
    If the programme was previously hard quit, companion.ome files may remain
    in the C_OME_FILES directory. This function returns them before continuing
    the run.
    """

    for root, dir_names, files in os.walk(root_dir_path):
        for dir in dir_names:
            if dir == 'C_OME_FILES':
                ome_dir = os.path.join(root, dir)
                if not len(os.listdir(ome_dir)) == 0:
                    for file in os.listdir(ome_dir):
                        if file.endswith('.companion.ome'):
                            ome_basename = file.split('.companion')
                            ome_src = os.path.join(ome_dir, file)
                            ome_p = os.path.dirname(os.path.abspath(
                                ome_dir)) + '/%s/%s.companion.ome' % (
                                    ome_basename[0], ome_basename[0])
                            shutil.move(ome_src, ome_p)
                else:
                    logging.info('No leftover companion.ome files')



def main(root_dir_path):
    fus_tiffs_dict = {}
    comp_ome_dict = {}
    mult_const = multiplier

    py_file_loc = os.path.realpath(__file__)

    # initialize the log settings
    logging.basicConfig(filename='%s/stitch.log' % root_dir_path,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO, datefmt='%d-%m-%Y %H:%M:%S')

    logging.info('Searching for tiff series in: %s' % root_dir_path)

    # Uses the passed parameters from the GUI to do boolean checks.
    if y_orig == '0':
        yes_orig = False
    else:
        yes_orig = True

    if y_macro == '0':
        yes_macro = False
    else:
        yes_macro = True
    
    if ome_only == '0':
        ome_process = False
    else:
        ome_process = True

    logging.info('Create fused from original positions (Fused.orig) = '
                 + str(yes_orig))
    logging.info('Run imageJ macro = ' + str(yes_macro))
    logging.info('Run on companion.ome file only = ' + str(ome_process))
    logging.info('Multiplier value set as: ' + str(mult_const))

    # First returns any companion.ome files from C_OME_FILES directory
    try:
        ome_transfer(root_dir_path)
    except Exception as e:
        logging.exception(str(e))

    try:
        logging.info('Attempting to stitch')

        # Runs stitcher on companion.ome files only
        if ome_process:
            for root, dir_names, files in os.walk(root_dir_path):
                for dir in dir_names:
                    aurox_dir_path = os.path.join(root, dir)
                    dir_name = os.path.basename(aurox_dir_path)
                    fusloc_dict = process_aurox_ome_only(
                        aurox_dir_path, dir_name)
                    fus_tiffs_dict.update(fusloc_dict)

        # Runs stitcher using selected parameters.
        else:
            for root, dir_names, files in os.walk(root_dir_path):
                for dir in dir_names:
                    aurox_dir_path = os.path.join(root, dir)
                    dir_name = os.path.basename(aurox_dir_path)
                    fusloc_dict, omeloc_dict = process_aurox_image(aurox_dir_path, dir_name, yes_orig, mult_const)
                    fus_tiffs_dict.update(fusloc_dict)
                    comp_ome_dict.update(omeloc_dict)

            if not len(comp_ome_dict) == 0:
                move_ome_file_in(comp_ome_dict)

        logging.info('Stitching ended')

    except Exception as e:
        logging.exception(str(e))

    # Runs macro on the FUSED_TIFFS tiffs if this option was selected.
    if yes_macro:
        try:
            logging.info('Attempting to run Stitch '
                         'ImageJ macro on fused images.')

            ij_macro_processor(fus_tiffs_dict, py_file_loc)

            logging.info('Macro processing ended')

        except Exception as e:
            logging.exception(str(e))
    else:
        logging.info('Macro processing SKIPPED.')

    logging.info('RUN FINISHED')

# The following are the parameters passed from the command line prompt to
# this script after the user has interacted with the stitch_RUN GUI:

#@String root_dir_path
#@String y_orig
#@String y_macro
#@String multiplier
#@String ome_only


main(root_dir_path)
