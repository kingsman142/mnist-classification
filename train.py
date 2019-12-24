import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import MNISTClassifier
from utils import *

print("GPU available: {}".format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up model
model = MNISTClassifier(pretrained = True).to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# set up train and test datasets
mnist_train = datasets.MNIST("data", train = True, transform = transform, target_transform = None, download = True)
mnist_test = datasets.MNIST("data", train = False, transform = transform, target_transform = None, download = True)

print("Train size: {}".format(len(mnist_train)))
print("Test size: {}".format(len(mnist_test)))

img, label = mnist_train[0]

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
        imgs, labels = sample
        imgs = imgs.to(device)
        true_labels = labels.to(device)

        pred_scores = model(imgs).to(device)
        loss = loss_func(pred_scores, true_labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 10 == 0:
            print("(train) => Epoch {}/{} - Batch {}/{} - Loss: {}".format(epoch, args.num_epochs, batch_id, len(train_loader), loss.item()))

# test the model
correct = 0
for batch_id, sample in enumerate(test_loader):
    imgs, labels = sample
    imgs = imgs.to(device)
    true_labels = labels.to(device)

    pred_scores = model(imgs).to(device)

    pred_label = torch.argmax(pred_scores, 1)
    correct += torch.sum(pred_label == true_labels).item()

    if batch_id % 100 == 0:
        print("(test) => Batch {}/{}".format(batch_id, len(test_loader)))
test_accuracy = correct / len(mnist_test)
print("Test accuracy: {}%".format(round(test_accuracy * 100.0, 2)))
