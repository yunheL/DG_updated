# MNIST_TF

# Overview
This MNIST\_TF implementation attemps to partition the training gradient and only use part of the gradient for training.

# Usage
Run ps.sh and experiment.sh to start experiment. They can either be on the same instance or different ones.  

Modify the important parameters listed below before starting the experiment.

The experiment script will create a log file for the evluation output.  

## Important Parameters
**ps\_hosts:** The IP addresses of ps instances.  
**worker\_hosts:** The IP addresses of worker instances.  
**num\_partition:** How many partitions is the gradient divided into. For is example, if --num\_partition=4, then each partition is 25% of the gradient.  
**num\_batch:** Number of partitions to kee during training.

# Contact
Please contact Yunhe <liu348@wisc.edu> and Xiangjin <xwu@cs.wisc.edu> if you have concerns or questions. Thank you!
