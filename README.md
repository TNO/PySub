# PySub
PySub is a python package with a modelling framework containing tools to predict subsidence caused by mining activities and can make subsidence prognoses, show the results and relevant statistical characteristics. PySub is developed within in the KEM-16 project as an element of a wider toolbox with the aim to model and show how contributions from multiple mining activities be discriminated. No new development of PySub is anticipated.
The output consists of overview text files, a variety of figures and stored models. It builds on top of other open source packages: numpy, xarray, pandas, numba, shapely, shapefile, osgeo, pyproj, descartes, matplotlib, cartopy, tqdm, adjustText and scipy.
In this README the following is covered:
- [Installation](Installation)
- [Tutorials](Tutorials)
- [Example scripts](Example-scripts)
- [Documentation](Documentation)
- [Templates](Templates)

Below, a hypothetical subsidence bowl is displayed in a contour plot, together with two cross sections where individual contributions of gas fields are highlighted:

![Model](https://github.com/TNO/PySub/blob/main/Subsidence_bowl_topview.png?raw=True)

![Model](https://github.com/TNO/PySub/blob/main/Subsidence_bowl_AB.png)

![Model](https://github.com/TNO/PySub/blob/main/Subsidence_bowl_CD.png)

## Installation
First clone the package to your local machine. You must have conda as package installer, Anaconda/Miniconda is optional. If you have Anaconda/Miniconda open Anaconda/Miniconda prompt, else use command prompt. Type `cd` to the location where the package is installed. For instance:
>cd C:\Users\user\Documents\PySub

This must be the folder where the setup.py file is in.

To install the environment used in this package:
>conda env create -n YOURENVIRONMENT -f PySub.yml

You can change YOURENVIRONMENT to a name of your choosing, we recommend using the environment name PySub.

Then, activate the environment by typing:
>conda activate YOURENVIRONMENT

And make sure the scripts can be found with:
>pip install -e .

Keep in mind that the space and "." after "-e" are also required!

The environment comes with a working Spyder and Jupyter Notebook built. Which you can open with
> spyder -p .

(keep in mind the space and "." after the "-p") or
>jupyter notebook

## Tutorials
To help you getting started with PySub there are 4 tutorials available in the Tutorials folder. These are Jupyter notebooks that assist you in running simple example models and explain the steps, function arguments and results.

## Example scripts
Example scripts are available to show case and help with the different functionalities of PySub and can be found in the folder Example_scripts. Examples are given

Calculate subsidence.py: A simple comparison between the three available compaction models.
Calculate subsidence - BucketEnsemble.py: Probabilistic analyses using a bucket method (see documentation).
Calculate subsidence - Suite.py: Running and comparing multiple models (Suites)
Calculate subsidence - from pressure grid.py: How to implement data from grids.
Calculate subsidence - Salt moving rigid basement.py: Run a model where salt subsidence behaves according to the moving rigid basement method ([source](https://www.nlog.nl/sites/default/files/tno_rapport_waddenzee_final_v17092012_public%20version%20-gelakt.pdf))

## Documentation
The documentation (available in the folder Documentation) on PySub is split in three parts: a technical manual, an user manual and a case study. In the technical manual the methodology and implementation of the relevant models are explained. In the user manual the input files are extensively described and instructions for entering the variables are given. When appropriate, a range of valid values is given. The case study elaborates on some of the use cases with the aim to show cases the (visualization) capabilities of PySub.

## Templates
In the templates folder the input excel files are given for certain use cases (as discussed in the user manual).
