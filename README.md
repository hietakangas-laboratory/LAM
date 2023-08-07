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
LAM can be used in a Python (>=3.7, <=3.11) environment and can be found on PyPI. Recommendation is to install LAM into
its own virtual environment. An easy and functional way to prepare the LAM-environment is by using the "recipe"-file,
LAMenv.yml, to create a conda environment ([Download Anaconda](https://www.anaconda.com/download/)). Simply, in
'Anaconda Prompt' give the following commands using functional path (e.g. <samp>"D:\user\LAM-master\LAMenv.yml"</samp>):
```console
conda env create -n lamenv -f="Path/To/LAM-master/LAMenv.yml"
conda activate lamenv
# [OPTIONAL]: pip install lam
```

As a PyPI-package, LAM can be installed with the command `pip install lam`. Installing LAM enables launching the
graphical user interface (GUI) via prompt command `lam-run`. However, installing this way disables direct editing of
<samp>"src/settings.py"</samp> and consequently restricts LAM to the settings available on the GUI and/or command line.
However, a separate LAM-master **can be edited** and then be executed in the environment with command
`python "Path/To/Alternate/src/run.py"`.

LAM can alternatively be installed from command line using the 'setup.py' by giving command: `python setup.py install`
while located inside the LAM-master -directory. ~~Windows-users are recommended to install Shapely>=1.7.0 from a
pre-compiled wheel found [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely) in order to properly link GEOS and
cython. The wheel can be installed with **pip install path/to/wheel**.~~

- UPDATE, 2023: conda installations of Shapely **are functional** for Windows.

---

### Usage
LAM is used by executing <samp>"src/run.py"</samp> or with console command `lam-run`, both of which by default open up
the GUI. Settings can be handled through <samp>src/settings.py</samp>, but LAM also includes argument parsing for most
important settings (`python src/run.py -h` or `lam-run -h`). Refer to <samp>'docs/UserManual'</samp> for additional
information.

#### Run examples

Note that many of LAM's command line arguments are toggles that switch the settings from their default behaviour as
defined in <samp>"src/settings.py"</samp>. This allows for better customization when for example designing batch files
(see <samp>"docs/run_split_count.bat"</samp>).

```console
# IN LAM ENVIRONMENT:
# Launch GUI with default settings
lam-run

# Project and count the dataset at given path without GUI and bypassing user prompts (Linux path).
lam-run -p ~/datasets/lam-data -o c -GD

# Perform Count, Plots and Stats using 50 bins, and on input files with column names on the third row.  
lam-run -p "D:\user\LAM-master\data" -o cls -b 50 -H 2

# Launch GUI of a non-installed version of LAM and specify path to dataset
python "D:\user\LAM-master\src\run.py" -p "D:\user\LAM-master\data"
```

#### Related material
A video tutorial series on LAM can be found on
YouTube [here](https://www.youtube.com/playlist?list=PLjv-8Gzxh3AynUtI3HaahU2oddMbDpgtx). Several modules related to
forming LAM-compatible folder structures can be found
[here](https://github.com/hietakangas-laboratory/LAM-helper-modules). Hietakangas lab also provides a stitching script that uses ImageJ to properly stitch tile scan images for object
detection and following LAM analysis. The script can be found [here](https://github.com/hietakangas-laboratory/Stitch).

For object segmentation and/or acquirement of label information, we also provide a wrapper package for
[StarDist](https://github.com/stardist/stardist) called [predictSD](https://github.com/hietakangas-laboratory/predictSD)
that includes several 3D deep learning models that have been trained on images from Aurox spinning disc confocal. The
package can extract label information in a format that is directly usable by LAM.

---
### Test data
The 'data/'-directory includes a small test dataset of two sample groups with four samples each. Note that the
sample number is not enough for a proper analysis; in ideal circumstances, it is recommended that each sample group
should have >=10 samples. Refer to user-manual for additional information.

---
### Publication
* Viitanen, A., Gullmets, J., Morikka, J., Katajisto, P., Mattila, J., & Hietakangas, V. (2021). An image analysis method
for regionally defined cellular phenotyping of the Drosophila midgut. Cell Reports Methods, Sep 27th.
https://doi.org/10.1016/j.crmeth.2021.100059

---
### Additional Resources
* [LAM helper modules](https://github.com/hietakangas-laboratory/LAM-helper-modules) - organize data for LAM input
* [LAM tutorial videos](https://www.youtube.com/playlist?list=PLjv-8Gzxh3AynUtI3HaahU2oddMbDpgtx)
* [predictSD](https://github.com/hietakangas-laboratory/predictSD) - a wrapper and some models for running
  [StarDist](https://github.com/stardist/stardist) segmentation with LAM-compatible output
* [Stitch](https://github.com/hietakangas-laboratory/Stitch) - Tile scan image stitching

### License
This project is licensed under the GPL-3.0 License  - see the LICENSE.md file for details

### Authors
* Arto I. Viitanen - [Hietakangas laboratory](https://www.helsinki.fi/en/researchgroups/nutrient-sensing)

### Acknowledgments
* Ville Hietakangas - [Hietakangas laboratory](https://www.helsinki.fi/en/researchgroups/nutrient-sensing/)
* Jaakko Mattila - [Mattila laboratory](https://www.helsinki.fi/en/researchgroups/metabolism-and-signaling/)
