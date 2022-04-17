import torch
import torchvision
import torchvision.transforms as transforms
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        embedding = F.relu(self.fc2(x))
        # embedding = x.clone()
        x = self.fc3(embedding)
        return x, embedding

    # def embedding(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = torch.flatten(x, 1)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     return x

class GrayNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        embedding = F.relu(self.fc2(x))
        # embedding = x.clone()
        x = self.fc3(embedding)
        return x, embedding

net_state_dict = torch.load("/workspace/CHAHAK/dsml/project/data/cifar_grayscale_trained.pth")
net = GrayNet()
net.load_state_dict(net_state_dict)
net.eval()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

grayscale_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Grayscale(num_output_channels=1),
     transforms.Normalize((0.5), (0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='/workspace/CHAHAK/dsml/project/data/cifar-10-batches-py', train=True,
                                        download=True, transform=grayscale_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/workspace/CHAHAK/dsml/project/data/cifar-10-batches-py', train=False,
                                       download=True, transform=grayscale_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

from tqdm import tqdm


embeddings, all_labels = [], []
for i, data in tqdm(enumerate(testloader, 0)):
    inputs, labels = data
    _, emb = net(inputs)
    embeddings.append(emb.detach().numpy())
    all_labels.extend(labels)

import numpy as np

embedding_array = np.vstack(embeddings)
embedding_array.shape

from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import homogeneity_score

kmeans = KMeans(n_clusters=10)
estimator = make_pipeline(StandardScaler(), kmeans).fit(embedding_array)
homogeneity_score(all_labels, estimator[-1].labels_)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
_ = ax.hist(estimator[-1].labels_, bins=100)
