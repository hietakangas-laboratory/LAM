import csv
import os

from ij import IJ 

"""
README

Copy all aurox images inside a single folder (e.g. stitching). Inside each aurox image folder there should only be the 
*.positions.csv file and the *ome.tiff files (*companion.ome file can stay also -Arto).

USAGE
Open command line and use the following command:
<1> --ij2 --headless --console --run <2> "root_dir_path='<3>'"

with replacing <x> with following FULL PATHS
<1> Path to fiji exe-file
<2> Path to this python file
<3> Path to the directory containing sub-directories with image files

e.g.
C:\hyapp\fiji-win64-1.51u\Fiji.app\ImageJ-win64.exe --ij2 --headless --console --run C:\Users\artoviit\Downloads\stitcher-grandma-pro-2.0.py "root_dir_path='D:\Arto Viitanen\Microscopy\20.11.2018'"


SEE BELOW FOR COMMAND ON MAC

#Old usage example 
#For Mac
#./Fiji.app/Contents/MacOS/ImageJ-macosx --ij2 --headless --console --run ./stitcher-grandma-pro.py 'root_dir_path="./stitching"'
#
#For Windows (notice that " and ' are swaped)
#./Fiji.app/ImageJ-win64.exe --ij2 --headless --console --run ./stitcher-grandma-pro.py "root_dir_path='./stitching'"

"""

def calculate_overlap(dir_path):
    """
        TODO parameterize directory
    """
    
    abs_dirpath = os.path.abspath(dir_path)
    print(abs_dirpath)
    # Calculate overlap
    IJ.run("Grid/Collection stitching", "type=[Positions from file] order=[Defined by TileConfiguration] \
           directory=[%s] layout_file=TileConfiguration.txt fusion_method=[Do not fuse images (only write \
           TileConfiguration)] regression_threshold=0.30 max/avg_displacement_threshold=2.50 \
    absolute_displacement_threshold=3.50 compute_overlap computation_parameters=[Save memory (but be slower)] \
    image_output=[Fuse and display]" % abs_dirpath)


def linear_blending(dir_path):
    """
        Linear blend images. It saves two images: 
        Fused.tiff from TileConfiguration.fixed.txt with Z axis set to 0
        Fused.orig.tiff from TileConfiguration.txt with original overlap from aurox
    """
    abs_dirpath = os.path.abspath(dir_path)
    print(abs_dirpath)
    # Linear Blending
    IJ.run("Grid/Collection stitching", "type=[Positions from file] order=[Defined by TileConfiguration] directory=[%s] layout_file=TileConfiguration.fixed.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save memory (but be slower)] image_output=[Fuse and display]" % abs_dirpath)

    IJ.saveAs("Tiff", "%s/Fused.tiff" % abs_dirpath)

    # Linear Blending
    IJ.run("Grid/Collection stitching", "type=[Positions from file] order=[Defined by TileConfiguration] directory=[%s] layout_file=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save memory (but be slower)] image_output=[Fuse and display]" % abs_dirpath)

    IJ.saveAs("Tiff", "%s/Fused.orig.tiff" % abs_dirpath)

def list_tiff_files(dir_path):
    """
        List all *.tiff files in directory dir
    """
    tiff_file_names = []

    for file in os.listdir(dir_path):
        if file.endswith(".tiff"):
            tiff_file_names.append(file)
    
    return tiff_file_names


def read_positions(dir_path, dir_name):
    """
        Read dir/{dir}.positions.csv
    """
    positions_name = '%s/%s/%s.positions.csv' % (dir_path, dir_name, dir_name)
    with open(positions_name, 'rb') as f:
        reader = csv.reader(f)
        positions_content = list(reader)
    
    return positions_content
    
def create_tile_configuration(tiff_files_names, tiff_img_positions, dir):
    """
        Create file TileConfiguration.txt
    """

    tile_conf_path = '%s/TileConfiguration.txt' % dir
    with open(tile_conf_path, 'w') as f:
        write_header(f)
        
        tiff_files_names.sort()
        tiff_files_names.sort(key=len)
        l = list(zip(tiff_files_names, tiff_img_positions))
        for (name, pos) in l:
            mult_const = 3.10
            h = abs(int(pos[1])) * mult_const
            w = abs(int(pos[2])) * mult_const
            f.write('%s     ;    ;    (    %s,    %s,    0.0    )\n' % (name, h, w) ) 


def set_tile_configuration_z_axis_to_zero(dir):
    """
        Set
        TODO clean this function
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
                    
                    fw.write('%s     ;    ;    (    %s,    %s,    0.0    )\n' % (name, position_list[0], position_list[1]) ) 
                    #

                line = f.readline()
                line_number += 1

def write_header(fw):
    """
        Write header of TileConfiguration.txt
    """
    fw.write(
"""# Define the number of dimensions we are working on
dim = 3

# Define the image coordinates\n""") 

def process_aurox_image(dir_path, dir_name):
    
    print('\n\nProcessing %s' % dir_name)
    aurox_dir_path = '%s/%s' % (dir_path, dir_name)

    print('Reading configurations')
    tiff_files_names = list_tiff_files(aurox_dir_path)
    tiff_img_positions = read_positions(dir_path, dir_name)

    print('Creating TileConfiguration.txt')
    create_tile_configuration(tiff_files_names, tiff_img_positions, aurox_dir_path)

    print('Calculating overlap')
    calculate_overlap(aurox_dir_path)

    print('Setting TileConfiguration.txt Z axis to 0')
    set_tile_configuration_z_axis_to_zero(aurox_dir_path)

    print('Blending images')
    linear_blending(aurox_dir_path)


def main(root_dir_path):
    print('Searching aurox images in: %s' % root_dir_path)
    for dir_name in os.listdir(root_dir_path):
        if not dir_name.startswith('.'):
            process_aurox_image(root_dir_path, dir_name)
    
#@String root_dir_path
main(root_dir_path)   
    
