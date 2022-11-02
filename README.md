# Digit Recognition for MNIST Data set

## CONFIGURATION 

Python version : 3.11  
library : scikit-datasets numpy matplotlib

The required libraries can be download here :
```
pip install -r requirement.txt  
```

##  Description 
The MNIST data set is a collection of 70.000 black and white handwritten digits from 0 to 9.
The digits have been size-normalized and centered in a fixed-size image.

![pic_digits_mnist](mnist-digits-small.webp)

I took a subset of the data set that I split into a train and test set in the following manner :  
training set - 10.000 samples  
test set - 1.000 samples

Once the splitting part done, I applied the algorithm K-Nearest Neighbors

1. with fixed k and n in order to test my code.
2. by tuning the value of the hyperparameter k to be the most efficient.
3. by increasing the volume of my training set.

## Results

1. In the first execution of the code, I ran KNN with 1000 samples in my training set and k = 10. I get an accuracy of 0.846  

2. For the second execution I made a grid for the hyper parameter k and kept 1000 samples in my training set. Here are the results :  

We can see that k = 1 is the best match since it has the highest accuracy. This results can be explained by the fact that some numbers when they are handwritten can look alike, like 6 and 8. Thus increasing the k would add variance in a probleme where there isn't any overfit.  

3. Concerning the last execution of my code, I wanted to plot a well-known effect : increasing the number of sample in the training set would only increase the accuracy.  

## Execution
Run the Code here :
```
python DigitRecogition.py
```
