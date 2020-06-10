# Linear Analysis of Midgut
### ---------------LAM---------------

Linear Analysis of Midgut (LAM) is a tool for reducing the dimensionality
of microscopy image–obtained data, and for subsequent quantification of
variables and feature counts while preserving regional context. LAM’s intended
use is to analyze whole Drosophila melanogaster midguts or their sub-regions for
phenotypical variation due to differing nutrition, altered genetics, etc. Key
functionality is to provide statistical and comparative analysis of variables
along the whole length of the midgut for multiple sample groups. Additionally,
LAM has algorithms for the estimation of feature-to-feature nearest distances
and for the detection of cell clusters, both of which also retain the regional
context. The analysis is performed after image processing and feature detection.
Consequently, LAM requires coordinate data of the features as input.

### Installation
LAM is used in Python 3.7=> environment. The distribution includes
requirements.txt and LAMenv.yml that both contain names and version numbers
of LAM dependencies.

### Usage
LAM is used by executing 'src/run.py', which by default opens up a graphical
user interface. Settings are handled through src/settings.py, but LAM
includes argument parsing for most important settings ('python run.py -h').
Refer to 'docs\UserManual.pdf' for additional information.

### Test data
The 'data/'-directory includes a small test data set of three sample groups with
three samples each. Note that the sample number is not enough for a proper
analysis; in ideal circumstances, it is recommended that each sample group
should have >=10 samples. Refer to 'docs/UserManual.pdf' for additional
information.

### Authors

    Arto I. Viitanen - [Hietakangas laboratory](https://www.helsinki.fi/en/researchgroups/nutrient-sensing)

### License

This project is licensed under the MIT License - see the LICENSE.md file for details

### Acknowledgments
    Ville Hietakangas - [Hietakangas laboratory](https://www.helsinki.fi/en/researchgroups/nutrient-sensing/)
    Jaakko Mattila - [Mattila laboratory](https://www.helsinki.fi/en/researchgroups/metabolism-and-signaling/)

