# Beta-VAE implemented in Pytorch

In this repo, I have implemented two VAE:s inspired by the Beta-VAE [[1]](#1). One has a Fully Connected Encoder/decoder architecture and the other CNN. The networks have been trained on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. I have chosen the Fashion-MNIST because it's a relativly simple dataset that I should be able to recreate with a CNN on my won laptop (no GPU) as an exercise.

The CNN-model can recreate more details than the fully connected one, even though it only uses 0.05 as many parameters, clearly illustrating the advantage of using CNN:s when working with images.

The Beta-parameter in the title is an added weight to the Kullback Leibler divergence term of the loss-function. The KL-term of the loss increases the more our latent space representation of the data diverges from a Standard multivariate normal distribution. In this project, and for this dataset, I have observed that a lower Beta-term has added more flexibility, leading to more separation in the dataset and a better recreation of the images. However, it is worth noting that I am also using a KL-penalty term, based on the size of the dataset to increase stability during training, so the KL-term is being scaled down always during training.

I have experimented with both: MSE-loss (used in the paper) and binary cross-entropy. Since the images are in black and white, binary cross-entropy has shown more promising results in separating the different classes.


## Fully connected encoder/decoder network
A model made out of fully connected networks has no problem learning a general representation of each label. However, I does not recreate details well. In general, it recreates each image as a standard representation of the pice of clothing rather than exact recreations. Bellow is a example generated with the Beta=0.1, where the right side are the real images, and the left the reconstructions.

![Alt text](/img/fc_result.png?raw=true "FC-VAE reconstruction")


We don't train this kind of model only for the reconstruction, there are better-suited autoencoders for that. The strength here is the ability to now sample from the latent space and create new pieces of clothing. 

The sampling is not as sharp as the reconstruction, but we can see some real clothes.

![Alt text](/img/fc_samples.png?raw=true "FC-VAE Samples")



## CNN encoder/Decoder network
![Alt text](/img/cnn_result.png?raw=true "CNN-VAE reconstruction")


![Alt text](/img/cnn_samples.png?raw=true "CNN-VAE samples")


### TODO
- [x] Optimize hyper-parameters for FC networks.
- [ ] Optimize hyper-parameters for CNN networks.
- [ ] Documentation.

## References
<a id="1">[1]</a> 
[beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) 2017.
