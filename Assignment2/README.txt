The code is available at: https://github.com/karthiksbhat/CS7641-MachineLearning
The dataset is available at:
FIFA 19 - https://www.kaggle.com/karangadiya/fifa19

%% This is for pre-processing the FIFA data, from Assignment 1 %%
In order to run the code, please have python, matplotlib, sklearn, numpy, and pandas installed
Please make sure to change the location of dataset files in the different python scripts (It is currently hardcoded to my computer's file structure -- the repository was created later for the sake of the submission)
These are found on:
1. line 19-20 in fifa-analysis.py
2. line 21-22 in fashion-mnist.py
3. line 6, line 49-50 in fifa-data-preprocessing.py

For FIFA dataset,
1. Run fifa-data-preprocessing.py first (`python fifa-data-preprocessing.py` on terminal from within the the files' location)
2. This code will clean the dataset (remove empty values etc), remove extraneous attributes etc, and write into two files for testing and training.

%% This is for the Assignment 2 %%
To run the NN random optimization code:
1. Install 'mlrose' python package (pip install mlrose)
2. Install jupyter -- the code is submitted as a Jupyter notebook.
3. As before, make sure that the datasets are in the right folder, or that the locations are changed to match your laptop's file structure.
4. Run each block of code as necessary.
5. There exists a demarcating line ("# ----------...") that separates working code from earlier trial and messed up code.

To run the Problems code: (This code has been taken from https://github.com/cmaron/CS-7641-assignments; the readme too will be very similar.)
1.  pip install -r requirements.txt
2. Install jython!
3. Run `jython continuouspeaks.py` or `jython flipflop.py` or `jython tsp.py` depending on the problem you're running for; this generates output CSVs of different values.
4. Once this is done (takes a while), run `python plotting.py` to get graphs generated to the output folder.