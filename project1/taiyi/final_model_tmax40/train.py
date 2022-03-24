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
from time import time
import traceback

# hyperparams
num_workers = 16
batch_size = 128
n_epochs = 200

# define transform
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# get training and test sets
train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)

# define classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# define data loaders
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size = batch_size,
    shuffle = False,
    num_workers = num_workers
)

# calculate block count per residual layer
def block_count(depth: int) -> int:
    assert (depth - 4) % 6 == 0
    return (depth - 4) // 6

def get_num_blocks(depth: int) -> list:
    return [block_count(depth), block_count(depth), block_count(depth)]

def make_model(k = 2, d = 82):
    # instantiate model
    model = ResNet(BasicBlock, get_num_blocks(d), k = k)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print('cuda')
        if torch.cuda.device_count() > 1:
            print('cuda: {}'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
    model.to(device)

    # load best model (lowest validation loss)
    try:
        model.load_state_dict(torch.load('cifar10_dnn.pt'))
        print('Model weights loaded')

    except:
        traceback.print_exc()
    return model
model = make_model()
summary(model, (3, 32, 32))

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 40)

# define training loop
test_loss_min = np.Inf

train_loss_list = list()
test_loss_list = list()
train_acc_list = list()
test_acc_list = list()

start = time()
for epoch in range(1, n_epochs + 1):
    train_loss = 0
    test_loss = 0
    total_correct_train = 0
    total_correct_test = 0
    total_train = 0
    total_test = 0
    # train model
    model.train()
    for data, target in train_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        # calculate accuracies
        _, pred = torch.max(output, 1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
        total_correct_train += np.sum(correct)
        total_train += correct.shape[0]

    # validate model
    model.eval()
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            # calculate accuracies
            _, pred = torch.max(output, 1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
            total_correct_test += np.sum(correct)
            total_test += correct.shape[0]

    # update scheduler
    scheduler.step()

    # compute average loss
    train_loss /= total_train
    test_loss /= total_test

    # compute accuracies
    train_acc = total_correct_train / total_train * 100
    test_acc = total_correct_test / total_test * 100

    # save data
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    # display stats
    print('Epoch: {}/{} \tTrain Loss: {:.6f} \tTest Loss: {:.6f} \tTrain Acc: {:.2f}% \tTest Acc: {:.2f}%'.format(epoch, n_epochs, train_loss, test_loss, train_acc, test_acc))

    # save best model
    if test_loss <= test_loss_min:
        print('Test loss decreased ({:.6f} --> {:.6f}. Saving model...'.format(test_loss_min, test_loss))
        torch.save(model.state_dict(), 'cifar10_dnn.pt')
        test_loss_min = test_loss
end = time()

print('Time elapsed: {} hours'.format((end - start) / 3600.0))

model = make_model()

# test model
test_loss = 0
total_correct = 0
total = 0

model.eval()
for data, target in test_loader:
    if torch.cuda.is_available():
        data, target = data.cuda(), target.cuda()
    with torch.no_grad():
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

# plot and save figures
plt.figure()
plt.plot(np.arange(n_epochs), train_loss_list)
plt.plot(np.arange(n_epochs), test_loss_list)
plt.title('Learning Curve: Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train Loss', 'Test Loss'])
plt.savefig('train_test_loss.png')
plt.close()

plt.figure()
plt.plot(np.arange(n_epochs), train_acc_list)
plt.plot(np.arange(n_epochs), test_acc_list)
plt.title('Learning Curve: Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train Accuracy', 'Test Accuracy'])
plt.savefig('train_test_acc.png')
plt.close()

# write training data to csv file
with open('train_data.csv', 'w') as f:
    f.write('train_loss, test_loss, train_acc, test_acc\n')
    for i in range(n_epochs):
        f.write('{}, {}, {}, {}\n'.format(train_loss_list[i], test_loss_list[i], train_acc_list[i], test_acc_list[i]))


































#
