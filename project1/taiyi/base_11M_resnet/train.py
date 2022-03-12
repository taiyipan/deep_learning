import numpy as np
import matplotlib.pyplot as plt
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
num_workers = 16 # tuned by testing (increases CPU efficiency)
batch_size = 32
valid_size = 0.1

# define writer
writer = SummaryWriter()

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

# define training loop
n_epochs = 2
valid_loss_min = np.Inf

train_loss_list = list()
valid_loss_list = list()

start = time()
for epoch in range(1, n_epochs + 1):
    train_loss = 0
    valid_loss = 0
    # train model
    model.train()
    for data, target in train_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        writer.add_scalar('Loss/train', loss, epoch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    # validate model
    model.eval()
    for data, target in valid_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        writer.add_scalar('Loss/valid', loss, epoch)
        valid_loss += loss.item() * data.size(0)

    # compute average loss
    train_loss /= len(train_loader.sampler)
    valid_loss /= len(valid_loader.sampler)

    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)

    # display stats
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

    # save best model
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}. Saving model...'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), 'cifar10_dnn.pt')
        valid_loss_min = valid_loss
end = time()

writer.flush()
writer.close()

print('Time elapsed: {} minutes'.format((end - start) / 60.0))

# visualize learning curve
plt.figure()
plt.plot(np.arange(n_epochs), train_loss_list)
plt.plot(np.arange(n_epochs), valid_loss_list)
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()
















































































#
