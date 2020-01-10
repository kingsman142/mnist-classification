import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from model import MNISTClassifier
from utils import *

def load_dataset(path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)), # normalize images
    ])
    # load dataset
    mnist = datasets.MNIST(path, train = False, transform = transform, target_transform = None, download = True)

    num_points = 1000 #len(mnist) # 500
    loader = torch.utils.data.DataLoader(mnist, batch_size = num_points, shuffle = False, num_workers = 1)
    batch = next(iter(loader))

    imgs, labels = batch
    return imgs, labels

def plot_points(data, labels, chart_title):
    x = data[:, 0] # i.e. shape is (N, 2) because it's reduced to 2D, so the x points need to be the first column
    y = data[:, 1] # y points are the second column
    sactter = plt.scatter(x = x, y = y, c = labels)
    plt.title("{}".format(chart_title))
    plt.show()

# use GPU if available, otherwise use the CPU
# also, set up model
print("GPU available: {}".format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTClassifier(pretrained = True).to(device)
model.load_state_dict(torch.load(os.path.join("models", "mnist_model")))
model.to(device)

# load the data
data_path = os.path.join("data")
imgs, labels = load_dataset(data_path)
mnist_pretrained = model.model[0:10](torch.Tensor(imgs).to(device)).cpu().view(imgs.shape[0], -1).detach().numpy() # use layers 0:10 from the model for feature extraction, pass images through the model, come back, reshape to (N, feats), and convert from tensor to numpy array
mnist_original = imgs.reshape((imgs.shape[0], 784))
print("Dataset size: {}".format(len(imgs)))

# reduce dimensionality with TSNE
print("Processing TNSE for original dataset (without using pretrained model)...")
mnist_tsne_original = TSNE(n_components = 2).fit_transform(mnist_original) # reduce the original data to 2D
print("Processing TNSE for pretrained dataset (features extracted from pretrained model)...")
mnist_tsne_pretrained = TSNE(n_components = 2).fit_transform(mnist_pretrained) # reduce the output features from a model to 2D

# visualize the two plots
plot_points(mnist_tsne_original, labels, "Vanilla MNIST TSNE") # plot the TSNE points from the original dataset
plot_points(mnist_tsne_pretrained, labels, "Pretrained-model MNIST TSNE") # plot the TSNE points from the pretrained model
