# mnist-classification
Recognizing MNIST digits. Simple 10-class classification problem, largely solved in the vision community, but I just wanted to experiment with it.

## Description
In this project, I just try to replicate the high accuracies observed in the mainstream MNIST papers, without necessarily modeling their exact approaches.

## How to run

To download and train the model to replicate this code, it's a very simple two-line approach. It is important to note this code requires at least PyTorch 1.3.1 and Torchvision 0.4.2, as the model requires the torch.nn.Flatten() module. If both of those libraries are at the above versions or most recent, there is no need to run the first command. Otherwise, run both commands:

```
pip3 install -r requirements.txt
python3 train.py --num-epochs 10 --learning-rate 0.001 --batch-size 32
```

## Approach
I use PyTorch to implement everything, and all I needed was 3 files to do so. In `model.py`, the classifier (a CNN) is built. In `utils.py`, command-line arguments are parsed to setup hyperparameters for the model. Finally, `train.py` handles loading the data (MNIST dataset), as well as training and evaluating the model. The model is modified 5 different times, and each modification is described in chronological order of how I chose to implement them, along with the improved accuracy of the model. All models were trained for 10 epochs with the standard MNIST split of 60k training and 10k testing images. No validation set was used in the training of this model, as it can be trained fairly quickly even without a GPU on a standard laptop. All evaluated models have a specific name which are used in the Results section below:

Initial hyperparameters were as follows:
* Adam optimizer (no weight decay, (0.9, 0.999) betas)
* 0.00002 learning rate
* 64 batch size
* 10 epochs
* Cross-entropy loss
* Standardized, but not normalized, images
* Greyscale images (1 input channel instead of 3 like RGB)

**Baseline**
There is no model for this, but since it's a 10-way classification task (10 digits), a non-trained model with random weights should achieve around 10% accuracy.

**Vanilla**

This model is identical to LeNet, introduced by Yann LeCun and gang ( http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf ) back in the 90s. I used all the initial hyperparameters defined above. It achieves an accuracy of 97.25%. The architecture was as follows:

```
nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = (5, 5), stride = 1),
nn.Tanh(),
nn.AvgPool2d(kernel_size = (2, 2), stride = 2),
nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = (5, 5), stride = 1),
nn.Tanh(),
nn.AvgPool2d(kernel_size = (2, 2), stride = 2),
nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = (4, 4), stride = 1),
nn.Flatten(),
nn.Linear(in_features = 120, out_features = 84, bias = True),
nn.Tanh(),
nn.Linear(in_features = 84, out_features = 10, bias = True)
```

**Normalization**
Standardization was already used when reading in images, which basically converts all pixels values from the range [0, 255] to [0, 1]. However, normalization was introduced in this model with mean 0.5 and standard deviation 0.5, further transforming the values from [0, 1] to [-1, 1] and changing the values from a uniform distribution to a gaussian (Normal) distribution. The accuracy improved from 97.25% to 97.98%. The code to implement this was a one-liner:

```
transforms.Normalize((0.5,), (0.5,))
```

**Batch 64**

I reduced the batch size from 128 to 64. I figured this might help the network produce more precise/granular weight updates. It made a decent improvement from 97.98% accuracy to 98.36% accuracy.

**Batch 32**

I further reduced the batch size from 64 to 32 for the same reason. While there was marginal increase in accuracy, there was still about a 0.12% increase, which is decent since we're already at such a high accuracy and we're trying to squeeze out the final 2% to get up to 100%. I suspect reducing the batch size even further will create weight updates that are too granular and will lead to overfitting on the rather large training set of 60k images, so I stop at a batch size of 32. The accuracy is now 98.48%.

**0.0005 Learning rate**

The learning rate is increased from 0.0002 to 0.0005 just to test if a 2.5x increase in learning rate would lead to improvements. Turns out, it led to a decent improvement from 98.48% to 98.73% accuracy.

**Deeper architecture**

At this point, any changes being made lead to such marginal increase in accuracy. As such, I decided to change up the architecture. I switched out average pooling layers for max pooling layers (observed a lot, empirically, in vision research), reduced stride in pooling layers (generates denser descriptors for the image later on), reduced the kernel size in the convolutional layers (leads to finer-grained features), and transitioned from tanh to relu activations (generally lead to faster convergence and no risk of saturation). The accuracy increased from 98.73% to 99.05% and the model is seen as follows:

```
nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3, 3)),
nn.ReLU(),
nn.MaxPool2d(kernel_size = (2, 2)),
nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3)),
nn.ReLU(),
nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3)),
nn.ReLU(),
nn.MaxPool2d(kernel_size = (2, 2)),
nn.Flatten(),
nn.Linear(in_features = 1024, out_features = 100, bias = True),
nn.ReLU(),
nn.Linear(in_features = 100, out_features = 10, bias = True)
```

**0.001 Learning rate**

Finally, I've exhausted nearly all hyperparameters, so I decided to switch out the Adam optimizer for an SGD optimizer with momentum as it generally has better convergence (again, this is taken from empirical observations in research papers). A learning rate of 0.001 and momentum of 0.9 were utilized for this update. The final accuracy achieved was 99.2%, which is about as good as we'll get. The de facto "leaderboard" of the best models on MNIST are located at http://yann.lecun.com/exdb/mnist/ . Most models that are better than the model produced in this repository are either several layers deeper, an ensemble of many neural networks, or require additional preprocessing of the dataset, which I'm not really interested in doing here.

To compare all these models and their accuracies, please view the results in the next section.

## Results

It is important to note all model accuracies are an average after 5 simulations, so the standard deviation of their accuracies should be relatively low. All models are in ascending order of trial and accuracy. Names are taken from the Approach section for brevity.

**Model** | **Accuracy**
Baseline | 10%
Vanilla | 97.25%
Normalization | 97.98%
Batch 64 | 98.36%
Batch 32 | 98.48%
0.0005 Learning rate | 98.73%
Deeper architecture | 99.05%
0.001 Learning rate | **99.2%**

## Failed Ideas

There were a couple of approaches I used to increase generalization of the model, just by using prior knowledge of the vision community. However, they either hurt or didn't improve accuracy. I have them listed below without any further explanation:

* Batch Norm
* Data Augmentation
  * Horizontal flip
  * Random rotate by 45 degrees
  * Random translate by 7 pixels
* Batch size < 16
* Even deeper architecture
* Pretrained AlexNet
