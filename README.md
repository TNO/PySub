# PySub
PySub is a python package with a modelling framework containing tools to predict subsidence caused by mining activities and can make subsidence prognoses, show the results and determine some statistical characteristics. This framework was developed by TNO-AGE by request of the KEM (Knowledge Programme on Effects of Mining)-programme. The output consists of overview text files, a variety of figures and stored models. It builds on top of other open source packages: numpy, xarray, pandas, numba, shapely, shapefile, osgeo, pyproj, descartes, matplotlib, cartopy, tqdm, adjustText and scipy.

In this README the following is covered:
- Installation
- Tutorials
- Example scripts
- Documentation
- Templates

Below, a hypothetical subsidence bowl is displayed in a contour plot, together with two cross sections where individual contributions of gas fields are highlighted:

![Model](https://github.com/TNO/PySub/blob/main/Subsidence_bowl_topview.png?raw=True)

![Model](https://github.com/TNO/PySub/blob/main/Subsidence_bowl_AB.png)

![Model](https://github.com/TNO/PySub/blob/main/Subsidence_bowl_CD.png)

## Installation
When this package has been cloned to your local machine you can open Anaconda prompt and `cd` to the location where the package is installed.

To install the environment used in this package:
>conda env create -n YOURENVIRONMENT -f PySub.yml

Then, to make sure the scripts can be found:
>pip install -e .

The environment comes with a working Spyder built, but feel free to install your own preferred IDE.

## Tutorials
After you have installed PySub you can get started. To assist you in your start there are 4 tutorials available in the Tutorials folder. These are Jupyter notebooks that assist you in running simple example models and explain the steps, function arguments and results.

## Example scripts
More cases are displayed in example scripts in the folder Example_scripts. Examples are given for:
- **Calculate subsidence.py**: A simple comparison between the three available compaction models.
- **Calculate subsidence - BucketEnsemble.py**: Probabilistic analyses using a bucket method (see documentation).
- **Calculate subsidence - Suite.py**: Running and comparing multiple models (Suites)
- **Calculate subsidence - from pressure grid.py**: How to implement data from grids.
- **Calculate subsidence - Salt moving rigid basement.py**: Run a model where salt subsidence behaves according to the moving rigid basement method ([source](https://www.nlog.nl/sites/default/files/tno_rapport_waddenzee_final_v17092012_public%20version%20-gelakt.pdf))

## Documentation 
The documentation (available in the folder Documentation) on this framework is split up in three parts: The technical manual, the user manual and a case study. In the technical manual the methods to determine the subsidence are explained. In the user manual the input files are elaborated on and instructions for entering the variables are given. When appropriate, a range of valid values is given. The case study elaborates on some of the use cases and shows some results.
