# Stitch
### ---------------Stitch---------------

'Stitch' is a tool for stitching together tiff images within a directory, utilising 
the ImageJ Grid/Collection plugin, and performing this stitching for multiple directories
(each containing a tiff series) in a batch process. The original purpose was to batch stitch 
the output of microscopy using an aurox clarity microscope; with a typical output being many 
directories containing .tiff files to be stitched, a positions.csv file (with tile positions) 
and a companion.ome metadata file. However, this programme can also be used for batch 
processing ONLY using a companion.ome metadata file associated with a tiff series.

### Installation
Stitch is used in Python 3=> environment. 

### Usage
Stitch is used by executing 'stitch_RUN.py', which by default opens up a graphical
user interface. Refer to Stitch_user_manual.pdf for full usage instructions.


### Authors

    Jack Morikka - [Mattila laboratory](https://www.helsinki.fi/en/researchgroups/metabolism-and-signaling)
    Jorge Fuentes

### Acknowledgments and References

    Josef Gullmets - [Hietakangas laboratory](https://www.helsinki.fi/en/researchgroups/nutrient-sensing/)

    Ville Hietakangas - [Hietakangas laboratory](https://www.helsinki.fi/en/researchgroups/nutrient-sensing/)
    
    Jaakko Mattila - [Mattila laboratory](https://www.helsinki.fi/en/researchgroups/metabolism-and-signaling/)

    'Stitch' uses the ImageJ Grid/Collection plugin which is based on a bioinformatics publication:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2682522/
    doi: 10.1093/bioinformatics/btp184
    
    Image of Disney character 'Stitch'. Under 'personal use only' license this image can 
    be used and distributed for non-commercial use.
    https://www.pngitem.com/middle/mxxwb_tumblr-stitch-disney-liloestitch-liloandstitch-cute-disney-characters/
