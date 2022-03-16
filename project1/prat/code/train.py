import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms 
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from models import ResNet, BasicBlock


def train_model(train_loader, epoch, loss_fn, optimizer, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('\nEpoch: %d' % epoch)
    
    model.train()
    model.to(device)
    train_loss_current = 0
    train_current_corrects = 0
    train_current_total = 0


    for batch, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        train_loss_current += loss.item()
        _, predicted_class = y_pred.max(1)
        train_current_total += y.size(0)
        train_current_corrects += (predicted_class == y).sum().item()
    
    # Save Checkpoint
    train_loss = train_loss_current/len(train_loader)
    train_accuracy = 100*float(train_current_corrects) / train_current_total
    
    return train_loss, train_accuracy 

def test_model(test_loader, epoch, loss_fn, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    global best_acc

    model.eval()
    model.to(device)

    test_loss_current = 0
    test_current_corrects = 0
    test_current_total = 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(test_loader):
            X = X.to(device)
            y = y.to(device)
    
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            test_loss_current += loss.item()

            _, predicted_class = y_pred.max(1)
            test_current_total += y.size(0)
            test_current_corrects += (predicted_class == y).sum().item()
    
    # Save Checkpoint
    test_loss = test_loss_current/len(test_loader)
    test_accuracy = 100*float(test_current_corrects) / test_current_total

    if test_accuracy > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': test_accuracy,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/model_40_2.pth')
        best_acc = test_accuracy
    
    return test_loss, test_accuracy



if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    NUMBER_OF_EPOCHS = 200
    SAVE_EVERY_X_EPOCHS = 50
    SAVE_MODEL_LOC = "./save_"
    LOAD_MODEL_LOC = None
    best_acc = 0

    # Code runs deterministically 
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)
    

    parser = argparse.ArgumentParser(description="Deep Learning Project-1")
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from latest checkpoint')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='no. of epochs')
    parser.add_argument('-b','--batch_size',type=int,default=128,help='batch_size')
    args = parser.parse_args()      


    # Data pre-processing
    print("Data Preprocessing ......")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')

    model = ResNet(BasicBlock, [6, 6, 6])
    print('No. of parameters:', sum(p.numel() for p in model.parameters()))

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        model.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_loss_history = []
    train_acc_history = []

    test_loss_history = []
    test_acc_history = []

    for epoch in range(args.epochs):
        train_loss, train_accuracy = train_model(trainloader, epoch, loss_fn, optimizer, model)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_accuracy)

        print("---------------Training-----------------")
        print("training loss:", train_loss)
        print("training accuracy:", train_accuracy)

        print("---------------Testing-----------------")
        
        test_loss, test_accuracy = test_model(testloader, epoch, loss_fn, model)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_accuracy)

        print("test loss:", test_loss)
        print('test accuracy:', test_accuracy)

        print('----------------------------------------------')
        scheduler.step()