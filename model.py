import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, k_size, out_ch, st, pad, pool=None, pool_k=None, pool_st=None, bias=True):
        super(ConvBlock, self).__init__()
        self.pool = pool

        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=k_size, stride=st, padding=pad, bias=bias)
        self.b1 = nn.BatchNorm2d(out_ch, affine=True)
        self.a1 = nn.ELU(True)
        if self.pool:
            self.p1 = nn.MaxPool2d(pool_k, pool_st)
        self.apply(weights_init) 
    def forward(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        if self.pool:
            x = self.p1(x)
        return x

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.l1 = ConvBlock(3, 3, 32, 1, 1)
        self.l2 = ConvBlock(32, 3, 32, 1, 1, 1, 2, 2)
        self.l3 = ConvBlock(32, 3, 64, 1, 1)
        self.l4 = ConvBlock(64, 3, 64, 1, 1, 1, 2, 2)
        self.l5 = ConvBlock(64, 3, 64, 1, 1, 1, 2, 2)        
        self.l6 = ConvBlock(64, 4, 512, 1, 0)
        self.l7 = nn.Dropout2d(0.5)
        self.l8 = ConvBlock(512, 1, 512, 1, 0, bias=False)
        self.l9 = nn.Dropout2d(0.5)
        # Global avg pooling to be used instead of maxpool
        self.l10 = nn.Conv2d(512, nclasses, 1, 1, 0, bias=False)
        self.l11 = nn.AvgPool2d(1, 1)
        self.l12 = nn.BatchNorm2d(nclasses, affine=True)
        self.apply(weights_init_uniform) 
        

    def forward(self, x):
        x = self.l1(x)
        #print(x.size())
        x = self.l2(x)
        #print(x.size())
        x = self.l3(x)
        #print(x.size())
        x = self.l4(x) 
        #print(x.size())                   
        x = self.l5(x)
        #print("=====>", x.size())            
        x = self.l6(x)
        #print(x.size())
        x = self.l7(x)
        #print(x.size())       
        x = self.l8(x)
        #print(x.size())
        x = self.l9(x)
        #print(x.size())
        x = self.l10(x)
        #print(x.size())
        x = self.l11(x)
        #print(x.size())
        x = self.l12(x)
        #print(x.size())        
        x = x.view(-1, nclasses)
        return F.log_softmax(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.01, 0.01)
    if classname.find('Conv2d') != -1:
        m.weight.data.uniform_(-0.01, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.uniform_(-0.01, 0.01)
        m.bias.data.fill_(0)
