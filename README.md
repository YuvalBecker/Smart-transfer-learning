# Statistics-pretrained
Method for adjusting pretrained networks for faster convergence and performance.
Done by estimating  the distribution of all pretrained layers using the trained dataset , and compare it to the distribution over the same pretrained layers using the desire new dataset

For example : Given VGG network trained over imagenet,  D - pretrained imagenet dataset , layer1(D)  - output layer1 distribution given imagenet dataset , D2 - new dataset , layer1(D2) - output layer1 distriubtion given new dataset.

Assumming we have small amount of data for trainning we want to use pretrained layers (weights) efficient as possible. Therfore in this approach , There is a differentiation process in order to identify layers which correspond similarly over both datasets, Those layers wont participate in the optimization process Because their operations seem to align with the new dataset. 
By "similarly" - compare between the 2 distributions of the 2 datasets.
This approach reduces overfitting by reducing the amount of variables and increase performance.

At the moment each layer activation is under the assumption of log normal distribution (given relu), Therfor I aggragate all layer activations in the network given all dataset
and compare between the 2 distributions by transforming to "normal" and use t test with different variances.

** assuming log normal - is very naiv approach (And incorrect) .

![alt text](https://github.com/YuvalBecker/Statistics-pretrained/blob/main/output_layer_histograms.JPG)


In the figure - layers output distributions, compare between 2 datsets. we can see for intuition , that the first layers seems to have global feature extraction , therfor its output
distributions are more similar than the deeper layers. 

### First results:
Trained 200 samples from CIFAR10 using vgg (pretrained from imagenet) 
1. Trained without modifying layers gradients - 54% accuracy over test data (10000 samples) 
2. Trained with modifying specific layers chosen by t -test - 68 % accuracy over test data (10000 samples)

** The trainning process was done without shuffle for comparison purposes. 

** stopping criteria : over fitting.
