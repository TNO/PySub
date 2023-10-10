# PySub
PySub is a modelling framework containing tools to predict subsidence caused by mining activities and can make subsidence prognoses, show the results and determine some statistical characteristics. The output consists of overview text files, a variety of figures and stored models. It builds on top of other open source packages: numpy, xarray, pandas, numba, shapely, shapefile, osgeo, pyproj, descartes, matplotlib, cartopy, tqdm, adjustText and scipy.

In this README the following is covered:
- Installation
- Tutorials
- Example scripts
- Documentation
- Templates

## Installation
When this package has been cloned to your local machine you can open Anaconda prompt and `cd` to the location where the package is installed.

To install the environment used for in this package:
>conda env create -n YOURENVIRONMENT -f PySub.yml

Then, to make sure the scripts can be found:
>pip install -e .

The encironemnt comes with a woeking Spyder built, but feel free to install your own preferred IDE.

## Tutorials
After you have installed PySub you can get started. To assist you in your start there are 4 tutorials available in the Tutorials folder. These are Jupyter notebooks that assist you in running simple example models and explain the steps, function arguments and results.

## Example scripts
More cases are displayed in these example scripts. Examples are given for:
- **Calculate subsidence.py**: A simple comparison between the three available compaction models.
- **Calculate subsidence - BucketEnsemble.py**: Probabilistic analyses using a bucket method (see documentation).
- **Calculate subsidence - Suite.py**: Running and comparing multiple models (Suites)
- **Calculate subsidence - from pressure grid.py**: How to implement data from grids.
- **Calculate subsidence - Salt moving rigid basement.py**: Run a model where salt subsidence behaves according to the moving rigid basement method (source)

## Documentation 
The documentation on this framework is split up in three parts: The technical manual, the user manual and a case study. In the technical manual the 

# Plaatje
# KEM16 project
# Licentie
