import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import MNISTClassifier
from utils import *

# use GPU if available, otherwise use the CPU
print("GPU available: {}".format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up model
model = MNISTClassifier(pretrained = True).to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)), # normalize images
])

# set up train and test datasets
mnist_train = datasets.MNIST("data", train = True, transform = transform, target_transform = None, download = True)
mnist_test = datasets.MNIST("data", train = False, transform = transform, target_transform = None, download = True)

print("Train size: {}".format(len(mnist_train)))
print("Test size: {}".format(len(mnist_test)))

# set up train and test dataset loaders
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size = args.batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size = 16, shuffle = True)

# set up optimizer and loss function
if args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, betas = (0.9, 0.999))
else:
    optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9)
loss_func = nn.CrossEntropyLoss(reduction = 'sum').to(device)

print("\nTraining on MNIST:")
print("\tNum epochs: {}".format(args.num_epochs))
print("\tLearning rate: {}".format(args.learning_rate))
print("\tBatch size: {}".format(args.batch_size))

# train the model
for epoch in range(args.num_epochs):
    for batch_id, sample in enumerate(train_loader):
        # get the data (images and labels)
        imgs, labels = sample
        imgs = imgs.to(device)
        true_labels = labels.to(device)

        # make predictions and calculate loss
        pred_scores = model(imgs).to(device)
        loss = loss_func(pred_scores, true_labels)

        # weight update
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # log info for the user
        if batch_id % 10 == 0:
            print("(train) => Epoch {}/{} - Batch {}/{} - Loss: {}".format(epoch, args.num_epochs, batch_id, len(train_loader), loss.item()))

# test the model
correct = 0
for batch_id, sample in enumerate(test_loader):
    # get the data (images and labels)
    imgs, labels = sample
    imgs = imgs.to(device)
    true_labels = labels.to(device)

    # make predictions for each class
    pred_scores = model(imgs).to(device)

    # take the highest class prediction and count how many labels matched the ground-truth labels
    pred_label = torch.argmax(pred_scores, 1) # i.e. if predictions were [0.1, 0.0, 0.6, ..., 0.0], it'd return index/class 2
    correct += torch.sum(pred_label == true_labels).item() # count how many were correct

    # log info for the user
    if batch_id % 100 == 0:
        print("(test) => Batch {}/{}".format(batch_id, len(test_loader)))
test_accuracy = correct / len(mnist_test)
print("Test accuracy: {}%".format(round(test_accuracy * 100.0, 2)))
