# β-VAE implemented in Pytorch

In this repo, I have implemented two VAE:s inspired by the β-VAE [[1]](#1). One has a Fully Connected Encoder/decoder architecture and the other CNN. The networks have been trained on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

I have experimented with what kind of loss to use: MSE (used in the paper) and binary cross-entropy. Since the images are in black and white, binary cross-entropy has so far shown more promising results in separating the different classes with the fc-network, however, no conclusions can be presented at this stage.

### TODO
- [ ] Optimize hyper-parameters for both networks.
- [ ] Quantify comparisons between binary cross entropy loss vs mse for this specific dataset.
- [ ] Documentation.

## Fully connected encoder/decoder network
A model made out of fully connected networks has no problem learning a general representation of each label. However, so far it doesn't generalize very well, and rather than recreating the exact instance of a shoe, for example, it recreates a general representation. See the show in the lower-left corner, for example, it recreates a shoe but not in the right direction. 

![Alt text](/img/fc_reconstruction.png?raw=true "FC-VAE reconstruction")



## CNN encoder/Decoder network



## References
<a id="1">[1]</a> 
[beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) 2017.
