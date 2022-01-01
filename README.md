# Smart transfer learning
The last decade has proved Deep Learning models to be very successful in solving tasks from various fields including Computer Vision, Natural Language Processing, Signal Processing, Biology, and others. Many DL models consists of thousands and even billions of learnable parameters making the learning process very expensive both computationally and in the amount of training samples. In recent years, a lot of effort was put into the generation and training of large, complex “generalize” models based on vast amount of data and a lot of computational resources.
Therefore, transfer learning has become a common method.
Despite the immense popularity of transfer learning, there has been little work studying its precise effects.

Method that performs smart transfer learning by calculating the statistics of two datasets over a pre-trained network. One is the dataset our pre-trained model was trained on(dataset_pre) and the second is the new dataset we want to train the network(dataset_new). 
In order to identify which layer/kernel performs generalization for the new task we conduct a similarity test between the two distribution for every layer/kernel. A layer/kernel with similarity higher than a threshold (hyper parameter) is identify as a generalized layer.


We will describe our general identification process in steps:
	Data collection:
Some definitions first: 
D_pre-pretrained dataset,D_new-new dataset,Net_pre-pretrained network
	Running over all new dataset and aggregate the outputs: 
aggregatio_new (i,layer,kernel)=Net_pre (D_(new(i)) (k))  
i-sample index
	Running over all pre-trained dataset and aggregate the outputs: 
 aggregatio_pre (i,layer,kernel)=Net_pre (D_(pre(i)) (k))
i-sample index
We forward pass all pre-trained dataset and collect network’s activations for every kernel in every layer. We repeat the process for the new dataset.

![asdasd](https://github.com/YuvalBecker/Statistics-pretrained/blob/main/statistics.png)


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
 
 
