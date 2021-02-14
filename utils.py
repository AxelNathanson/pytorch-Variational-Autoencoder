import torch
from torchvision import datasets, transforms

torch.set_default_dtype(torch.float)


def load_dataset(**kwargs):
    """
    Function to load the Fashion-MNIST dataset.


    Optional Arguments:
        split {float} -- Training-set split
    Returns:
        [tuple] -- training-set, validation-set, testing-set
    """

    split = kwargs.pop('split', 0.8)

    normalize = transforms.Normalize(mean=(0.5,),
                                     std=(0.5,),
                                     inplace=True)

    transform = transforms.Compose([transforms.ToTensor()])

    train_val_set = datasets.FashionMNIST(root="\FMNIST",
                                          train=True,
                                          transform=transform,
                                          download=True)

    test_set = datasets.FashionMNIST(root="\FMNIST",
                                     train=False,
                                     transform=transform,
                                     download=True)

    num_train = int(len(train_val_set)*split)
    train_set, val_set = torch.utils.data.random_split(train_val_set, [num_train, len(train_val_set)-num_train])

    return train_set, val_set, test_set


def load_small_dataset(**kwargs):
    """
    Function to load a smaller part of the Fashion-MNIST dataset.


    Optional Arguments:
        split {float} -- Training-set split
    Returns:
        [tuple] -- training-set, validation-set, testing-set
    """

    num_samples = kwargs.pop('num_samples', 3000)

    normalize = transforms.Normalize(mean=(0.5,),
                                     std=(0.5,),
                                     inplace=True)

    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.FashionMNIST(root="\FMNIST",
                                      train=True,
                                      transform=transform,
                                      download=True)

    train_split, _ = torch.utils.data.random_split(train_set, [num_samples, len(train_set)-num_samples])

    return train_split
