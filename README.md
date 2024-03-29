# Smart transfer learning
The last decade has proved Deep Learning models to be very successful in solving tasks from various fields including Computer Vision, Natural Language Processing, Signal Processing, Biology, and others. Many DL models consists of thousands and even billions of learnable parameters making the learning process very expensive both computationally and in the amount of training samples. In recent years, a lot of effort was put into the generation and training of large, complex “generalize” models based on vast amount of data and a lot of computational resources.
Therefore, transfer learning has become a common method.
Despite the immense popularity of transfer learning, there has been little work studying its precise effects.

We develop a method that performs smart transfer learning by calculating the statistics of two datasets over a pre-trained network. One is the dataset our pre-trained model was trained on(dataset_pre) and the second is the new dataset we want to train the network(dataset_new). 
In order to identify which layer/kernel performs generalization for the new task we conduct a similarity test between the two distribution for every layer/kernel. A layer/kernel with similarity higher than a threshold (hyper parameter) is identify as a generalized layer.

We will describe our general identification process in steps:

Data collection:

Some definitions first: 
D_pre-pretrained dataset,  D_new-new dataset,  Net_pre-pretrained network

Running over all new dataset and aggregate the outputs: 

aggregation_new(i,layer,kernel) = Net_pre (D_(new(i)) (k))  , i-sample index

Running over all pre-trained dataset and aggregate the outputs: 

aggregation_pre (i,layer,kernel)=Net_pre (D_(pre(i)) (k)), i-sample index

We forward pass all pre-trained dataset and collect network’s activations for every kernel in every layer. We repeat the process for the new dataset.

![](https://github.com/YuvalBecker/Statistics-pretrained/blob/main/artifacts/statistics.png)
Figure 1, illustration of activation maps aggregation process, we collect for every input sample the activations of all kernels outputs in all layers


Distribution similarity test: 

We calculate for each layer and kernel it’s distribution, we repeat the process for both pre-trained dataset activations and for the new dataset activations.

aggregatio_pre (i,layer,kernel) ~ p_correct(layer,kernel)  

aggregatio_new (i,layer,kernel) ~ p_unknown(layer,kernel) 

Given the calculated two distributions for every kernel we perform a statistic test to measure the similarity between the distributions.

Statistic Test(p_correct(layer,kernel) , p_unknown(layer,kernel))



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
  

![alt text](https://github.com/YuvalBecker/Statistics-pretrained/blob/main/artifacts/stats_kernels.png)
Figure 2, Example of the produced 2 distributions for each kernel in specific layer. We used a DENSENET121 pretrained on FMNIST and compare to KMNIST dataset. Every plot represents the two distribution over specific kernel. 

### First results:
![alt text](https://github.com/YuvalBecker/Statistics-pretrained/blob/main/artifacts/densenet_results.png)

Figure3, Results over Densenet121 architecture, every color represents a specific datasets pair, dotted line represents the results using our algorithm. 
 

 
