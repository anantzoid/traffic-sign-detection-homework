from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import tensorboard_logger



# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--outf', type=str, default='/scratch/ag4508/models/baseline/')
parser.add_argument('--nw', type=int, default=2)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--lr_decay_step', type=int, default=10)
parser.add_argument('--lr_decay_rate', type=float, default=0.9)

args = parser.parse_args()

if not os.path.exists(args.outf):
    os.makedirs(args.outf)
torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(2)

if "/" in args.outf:
    path_split = [args.outf.split("/")[:-1], args.outf.split("/")[-1]]
else:
    path_split = [".", args.outf]
print(path_split[0], 'logs', path_split[1])
log_path = os.path.join(path_split[0], 'logs', path_split[1])
if not os.path.exists(log_path):
    os.makedirs(log_path)
tensorboard_logger.configure(log_path)

### Data Initialization and Loading
from data import initialize_data, data_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=args.nw)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=args.nw)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import *
#model = Net()
model = Net1()
model.apply(weights_init_uniform)
#crit = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
if use_cuda:
    model.cuda()
    #crit.cuda()

def early_stop(val_acc_history, t=10, required_progress=0.01):
    val_acc_history = [_[1] for _ in val_acc_history] 
    if len(val_acc_history) < t:
        return False
    stop = True
    for i in list(reversed(range(len(val_acc_history)-t, len(val_acc_history))))[:-1]:
        if val_acc_history[i] - val_acc_history[i-1] > required_progress:
            stop = False            
    return stop

train = {'step': [], 'loss': []}
test = {'step': [], 'loss': [], 'acc': []}
step = 0
lr = args.lr

def train(epoch):
    model.train()
    validation_acc_history = []
    stop_training = False

    for batch_idx, (data, target) in enumerate(train_loader):
        if data.size(0) != args.batch_size:
            continue
        data, target = Variable(data), Variable(target)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
            val_acc = validation(step)
            validation_acc_history.append([step, val_acc])

            stop_training = early_stop(validation_acc_history)
            if stop_training:
                print("earily stop triggered")
                lr *= args.lr_decay_rate
                print("LR change: ", lr)
                optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
                break            

        step += 1
    train['step'].append(step)
    train['loss'].append(loss.data[0])
    tensorboard_logger.log_value('train_loss', loss.data[0])


def validation(step):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if data.size(0) != args.batch_size:
            continue
        data, target = Variable(data, volatile=True), Variable(target)
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    val_acc = 100. * correct / len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        val_acc))
    train['step'].append(step)
    train['loss'].append(validation_loss)
    train['acc'].append(100. * correct / len(val_loader.dataset))
    tensorboard_logger.log_value('val_loss', validation_loss)
    tensorboard_logger.log_value('val_acc', val_acc)
    return val_acc


for epoch in range(1, args.epochs + 1):
    train(epoch)
    #validation(epoch)
    if epoch % args.lr_decay_step == 0:
        lr *= args.lr_decay_rate
        print("LR change: ", lr)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
         
    '''
    if epoch%30==0:
        lr *= 0.9
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    if epoch == 30:
        lr = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    elif epoch == 70:
        lr = 1e-4
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    '''
    model_file = os.path.join(args.outf, 'model_' + str(epoch) + '.pth')
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')
plt.plot(train['step'], train['loss'])
plt.plot(test['step'], test['loss'])
plt.savefig(os.path.join(args.outf, 'loss.png'))
plt.figure()
plt.plot(test['step'], test['acc'])
plt.savefig(os.path.join(args.outf, 'acc.png'))
