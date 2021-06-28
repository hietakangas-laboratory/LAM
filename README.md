![](/img/lam.ico)

# Linear Analysis of Midgut
### ---------------LAM---------------

Linear Analysis of Midgut (LAM) is a tool for reducing the dimensionality of microscopy image–obtained data, and for
subsequent quantification of variables and object counts while preserving spatial context. LAM’s intended use is to
analyze whole Drosophila melanogaster midguts or their sub-regions for phenotypical variation due to differing
nutrition, altered genetics, etc. Key functionality is to provide statistical and comparative analysis of variables
along the whole length of the midgut for multiple sample groups. Additionally, LAM has algorithms for the estimation of
feature-to-feature nearest distances and for the detection of cell clusters, both of which also retain the regional
context. LAM also approximates sample widths and can perform multivariate border-region detection on sample groups. The
analysis is performed after image processing and object detection. Consequently, LAM requires coordinate data of the
features as input.

### Installation
LAM is used in a Python (>=3.7, <3.9) environment and can be found on PyPI. It can be installed with command
**pip install lam**. Note that by installing this way, you will be restricted to the settings on the GUI and/or the
available command line arguments, and will not be able to alter settings.py.

You can alternatively install LAM from command line using the 'setup.py' by giving command:
'python setup.py install' while located inside the LAM-master -directory. Windows-users are recommended to install
Shapely>=1.7.0 from a pre-compiled wheel found [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely) in order to
properly link GEOS and cython. The wheel can be installed with **pip install path/to/wheel**.

The distribution also includes docs/requirements.txt and docs/LAMenv.yml that can be used to install dependencies using
pip or conda (Anaconda), respectively. Recommendation is to install LAM into its own virtual environment.

### Usage
LAM is used by executing 'src/run.py' or with console command 'lam-run', both of which by default open up the graphical
user interface. Settings can be handled through src/settings.py, but LAM also includes argument parsing for most
important settings ('python src/run.py -h' OR 'lam-run -h'). Refer to 'docs/UserManual' for additional information.
Several modules related to forming LAM-compatible folder structures can be found
[here](https://github.com/hietakangas-laboratory/LAM-helper-modules).

A video tutorial series on LAM can be found on YouTube
[here](https://www.youtube.com/playlist?list=PLjv-8Gzxh3AynUtI3HaahU2oddMbDpgtx).

Hietakangas lab also provides a stitching script that uses ImageJ to properly stitch images for object detection and
following LAM analysis. The script can be found [here](https://github.com/hietakangas-laboratory/Stitch).

For object segmentation and/or acquirement of label information, we also provide a wrapper package for
[StarDist](https://github.com/stardist/stardist) called [predictSD](https://github.com/hietakangas-laboratory/predictSD)
that includes several 3D deep learning models that have been trained on images from Aurox spinning disc confocal. The
package can extract label information in a format that is directly usable by LAM.

### Test data
The 'data/'-directory includes a small test dataset of two sample groups with four samples each. Note that the
sample number is not enough for a proper analysis; in ideal circumstances, it is recommended that each sample group
should have >=10 samples. Refer to user-manual for additional information.

### License
This project is licensed under the GPL-3.0 License  - see the LICENSE.md file for details

### Authors
Arto I. Viitanen - [Hietakangas laboratory](https://www.helsinki.fi/en/researchgroups/nutrient-sensing)

### Acknowledgments
Ville Hietakangas - [Hietakangas laboratory](https://www.helsinki.fi/en/researchgroups/nutrient-sensing/)

Jaakko Mattila - [Mattila laboratory](https://www.helsinki.fi/en/researchgroups/metabolism-and-signaling/)
