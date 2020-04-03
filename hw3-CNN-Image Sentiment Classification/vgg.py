import torch.nn as nn
import math

'''
    Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
    dilation=1, groups=1, bias=True, padding_mode='zeros') 
        in_channels(int) - Number of channels in the input image (1:binary , 3: RGB)
        out_channels(int) - Number of channels produced by the convolution,
                            i.e., the number of feature filter(neuron) will be conneted.
        
    BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    LeakyReLU(negative_slope=0.01, inplace=False)
    
    MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

    Dropout(p=0.5, inplace=False)
'''

class VGG(nn.Module):
    def __init__(self, ConvNet, n_class):
        super(VGG, self).__init__()
        self.ConvNet = ConvNet
        self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512*1*1, 512), # need to check Linear input size
                nn.ReLU(True),
                nn.Dropout(), 
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, n_class)              
        )
        # Initialize weights(L2 norm)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
    
    def forward(self, x):
        out = self.ConvNet(x)
        out = out.view(out.size(0), -1) # Flatten
        out = self.classifier(out)
        return out
    
def make_layers(in_channels, cfg, batch_norm):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}

def vgg16(in_channels=1 , n_class=7, batch_norm=False):
    return VGG(make_layers(in_channels, cfg['D'], batch_norm), n_class)

def vgg19(in_channels=1 , n_class=7, batch_norm=False):
    return VGG(make_layers(in_channels, cfg['E'], batch_norm), n_class)

            