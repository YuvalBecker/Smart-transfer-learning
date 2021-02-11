# Statistics-pretrained
Method for adjusting pretrained networks for faster convergence and performance.
Done by estimating  the distribution of all pretrained layers using the trained dataset , and compare it to the distribution over the same pretrained layers using the desire new dataset

For example : Given VGG network trained over imagenet,  D - pretrained imagenet dataset , layer1(D)  - output layer1 distribution given imagenet dataset , D2 - new dataset , layer1(D2) - output layer1 distriubtion given new dataset.

Assumming we have small amount of data for trainning we want to use pretrained layers (weights) efficient as possible. Therfore in this approach , There is a differentiation process in order to identify layers which correspond similarly over both datasets, Those layers wont participate in the optimization process Because their operations seem to align with the new dataset. 
By "similarly" - compare between the 2 distributions of the 2 datasets.
This approach reduces overfitting by reducing the amount of variables and increase performance.

At the moment each layer activation is under the assumption of log normal distribution (given relu), Therfor I aggragate all layer activations in the network given all dataset
and compare between the 2 distributions by transforming to "normal" and use  kl divergence.
Or by comparing between the distributions of gram matrixes , in order to reduce the localization dependancy.

call the constructor : `rg = CustomRequireGrad(network, dataloader, dataloader2)`

To run distribution calculations : `rg.run(stats_value = xx)`

Run inside the training loop : 
```
            loss = criterion(#Your inputs) 
            loss.backward()
            rg.update_grads(network)
            optimizer.step()
``` 
            
In order to change the specific weights grads
 

![alt text](https://github.com/YuvalBecker/Statistics-pretrained/blob/main/output_layer_histograms.JPG)


In the figure - layers output distributions, compare between 2 datsets. we can see for intuition , that the first layers seems to have global feature extraction , therfor its output
distributions are more similar than the deeper layers. 

** clarification: Distribution measurement is performed for every kernel in each layer - meanning , some kernel weights will 
be modified while other may not in the same layer!

Example of chosen weight kernelS in a specific layer:

![alt text](https://github.com/YuvalBecker/Statistics-pretrained/blob/main/data/save_activations/features.9_new.jpg)

In the figure FMNIST dataset over VGG imagenet pretrained

![alt text](https://github.com/YuvalBecker/Statistics-pretrained/blob/main/data/save_activations/features.9_pre.jpg)

In the figure The same layer with imagenet data as input


The activation maps are the ones the algorithm chose as meanningful between the 2 datasets. 

### First results:
Trained 200 samples from CIFAR10 using vgg (pretrained from imagenet) 
1. Trained without modifying layers gradients - 54% accuracy over test data (10000 samples) 
2. Trained with modifying specific layers chosen by t -test - 68 % accuracy over test data (10000 samples)

** The trainning process was done without shuffle for comparison purposes. 

** stopping criteria : over fitting.
 
 
