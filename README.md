# Beta-VAE implemented in Pytorch

In this repo, I have implemented two VAE:s inspired by the Beta-VAE [[1]](#1). One has a Fully Connected Encoder/decoder architecture and the other CNN. The networks have been trained on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

The Beta-parameter in the title is an added weight to the Kullback Leibler divergence term of the loss-function. The KL-term of the loss increases the more our latent space representation of the data diverges from a Standard multivariate normal distribution. In this project, and for this dataset, I have observed that a lower Beta-term has added more flexibility, leading to more separation in the dataset and a better recreation of the images. However, it is worth noting that I am also using a KL-penalty term, based on the size of the dataset to increase stability during training, so the KL-term is being scaled down always during training.

I have experimented with both: MSE-loss (used in the paper) and binary cross-entropy. Since the images are in black and white, binary cross-entropy has shown more promising results in separating the different classes.

### TODO
- [x] Optimize hyper-parameters for FC networks.
- [ ] Optimize hyper-parameters for CNN networks.
- [ ] Documentation.

## Fully connected encoder/decoder network
A model made out of fully connected networks has no problem learning a general representation of each label. However, I does not recreate details well. In general, it recreates each image as a standard representation of the pice of clothing rather than exact recreations. Bellow is a example generated with the Beta=0.1, where the right side are the real images, and the left the reconstructions.

![Alt text](/img/fc_reconstruction.png?raw=true "FC-VAE reconstruction")



## CNN encoder/Decoder network



## References
<a id="1">[1]</a> 
[beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) 2017.
