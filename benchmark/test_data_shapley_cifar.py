import sys

sys.path.append("..")

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./cifar10/', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000,
                                          shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./cifar10/', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=200,
                                         shuffle=False, num_workers=2)

dataiter = iter(trainloader)
images, labels = next(dataiter)

x_train = []
y_train = []
for i in range(len(images)):
    x_train.append(torch.reshape(images[i], (-1,)).tolist())
    y_train.append(labels[i].tolist())

dataiter = iter(testloader)
images, labels = next(dataiter)

x_val = []
y_val = []
for i in range(len(images)):
    x_val.append(torch.reshape(images[i], (-1,)).tolist())
    y_val.append(labels[i].tolist())

from opensv import DataShapley

x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)
print(x_train.shape)

shap = DataShapley()
shap.load(x_train, y_train, x_val, y_val)
shap.solve('kernel_shap')
print(shap.get_values())