import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from basic_block import BasicBlock
from resnet import ResNet
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from time import time
import traceback

# hyperparams
num_workers = 12 # tuned by testing (increases CPU efficiency)
batch_size = 32
valid_size = 0.1

# define transform
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# get training and test sets
train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# define classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# split training data
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# define data loaders
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size = batch_size,
    sampler = train_sampler,
    num_workers = num_workers
)
valid_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size = batch_size,
    sampler = valid_sampler,
    num_workers = num_workers
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size = batch_size,
    num_workers = num_workers
)

# instantiate model
model = ResNet(BasicBlock, [2, 2, 2, 2])
if torch.cuda.is_available():
    model.cuda()
    print('Model in GPU')
summary(model, (3, 32, 32))

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# load best model (lowest validation loss)
try:
    model.load_state_dict(torch.load('cifar10_dnn.pt'))
    print('Model weights loaded')
except:
    traceback.print_exc()

# test model
test_loss = 0
total_correct = 0
total = 0

model.eval()
for data, target in test_loader:
    if torch.cuda.is_available():
        data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item() * data.size(0)
    # calculate accuracies
    _, pred = torch.max(output, 1)
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
    total_correct += np.sum(correct)
    total += correct.shape[0]

# calculate overall accuracy
print('Model accuracy on test dataset: {:.2f}%'.format(total_correct / total * 100))






























































































#
