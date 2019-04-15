The code for this assignment has been taken from:
https://github.com/juanjose49/omscs-cs7641-machine-learning-assignment-4
The readme on that link (https://github.com/juanjose49/omscs-cs7641-machine-learning-assignment-4/blob/master/README.md) explains how to set up the code for execution.

I've made a few small changes for modifying the environment. This was done in these lines:
https://github.com/juanjose49/omscs-cs7641-machine-learning-assignment-4/blob/master/src/main/java/assignment4/HardGridWorldLauncher.java#L35
There, 1's represent walls, and 0's represent open spaces. I modified those for my "hardWorld2" environment.

I used plotter.py (uploaded here: https://github.com/karthiksbhat/CS7641-MachineLearning/tree/master/Assignment4)
to plot the results from the above code. These results are written onto the console in eclipse, as explained in the above documentation. They were copied over to arrays in the above code and platted using matplotlib.