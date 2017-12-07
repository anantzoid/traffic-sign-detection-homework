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
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x) 
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        x = self.l11(x)
        x = self.l12(x)
        x = x.view(-1, nclasses)
        return F.log_softmax(x)


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        layers = []        
        layers.append(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0))
        layers.append(nn.ELU())

        layers.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0))        
        layers.append(nn.BatchNorm2d(32, affine=True))
        layers.append(nn.ELU())

        layers.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0))        
        layers.append(nn.BatchNorm2d(32, affine=True))
        layers.append(nn.ELU())
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Dropout(0.2))

        layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0))        
        layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0))        
        layers.append(nn.BatchNorm2d(64, affine=True))
        layers.append(nn.ELU())
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Dropout(0.2))
        self.convlayers = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Linear(4096, 512))
        layers.append(nn.ELU())
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(512, nclasses))
        self.flatlayers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.convlayers(x)
        x = x.view(x.size(0), -1)        
        x =self.flatlayers(x)
        return F.log_softmax(x)

class Net3(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        layers = []        
        layers.append(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0))                
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0))                
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Dropout(0.2))

        layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0))        
        layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0))        
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Dropout(0.2))
        self.convlayers = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Linear(4096, 512))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(512, nclasses))
        self.flatlayers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.convlayers(x)
        x = x.view(x.size(0), -1)        
        x =self.flatlayers(x)
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



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(8192, 4096)
        self.do = nn.Dropout(0.5)
        self.linear2 = nn.Linear(2304, 512)
        self.linear3 = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        out = F.avg_pool2d(out, 4)        
        out = out.view(out.size(0), -1)        
        #out = self.do(self.linear(out))
        
        out = self.do(self.linear2(out))
        out = self.linear3(out)
        return F.log_softmax(out)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu_act = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.relu_act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu_act(out)
        return out


class Resnet_Custom(nn.Module):
    def __init__(self, num_classes):
        super(Resnet_Custom, self).__init__()
        self.in_ch = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.ReLU()
        # Res Layers
        self.res1 = self.make_res_layer(64, 2, 1)
        self.res2 = self.make_res_layer(128, 2, 2)
        self.res3 = self.make_res_layer(256, 2, 2)

        self.pool = nn.AvgPool2d(4, 4)
        self.do = nn.Dropout(0.5)
        self.l1 = nn.Linear(2304, 512)
        self.l2 = nn.Linear(512, num_classes)

    def make_res_layer(self, out_ch, block_size, stride):
        strides = [stride] + [1]*(block_size-1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.in_ch, out_ch, stride))
            self.in_ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.do(self.l1(x))
        x = self.l2(x)
        return F.log_softmax(x)
