The code and datasets are available at: https://github.com/karthiksbhat/CS7641-MachineLearning

In order to run the code, please have python, matplotlib, sklearn, numpy, and pandas installed
Please make sure to change the location of dataset files in the different python scripts (It is currently hardcoded to my computer's file structure -- the repository was created late for the sake of the submission)
These are found on:
1. line 19-20 in fifa-analysis.py
2. line 21-22 in fashion-mnist.py
3. line 6, line 49-50 in fifa-data-preprocessing.py

For FIFA dataset,
1. Run fifa-data-preprocessing.py first (`python fifa-data-preprocessing.py` on terminal from within the the files' location)
2. This code will clean the code, remove extraneous attributes etc, and write into two files for testing and training.

Following running the preprocessing file...
For both datasets, (`python fifa-analysis.py` or `python fashion-mnist.py` on terminal from within the the files' location)
In fifa-analysis.py and fashion-mnist.py:
1. The code is written essentially in series, demarkated by comments.
2. Please comment out any ML algorithm that you don't wish to run before running the files.
3. The code creates visualizations and prints out confusion matrices and classification scores. This is the only output from the files.