
from base import model
import torch 
from torch import nn

class gen_base(model):
    def __init__(self):
        super(gen_base, self).__init__()
    def eval(self):
        self.conv.eval()
    def train(self):
        self.conv.train()
    def parameters(self):
        return list(self.conv.parameters())
    def zero_grad(self):
        self.conv.zero_grad()
    def forward(self, input):
        return self.conv(input).permute(0,2,3,1)

    def save(self, name):
        dic = {}
        dic['conv'] = self.conv.state_dict()
        torch.save(dic, name)

    def load(self, name):
        dic = torch.load(name)
        self.conv.load_state_dict(dic['conv'])
    def type(self, dtype):
        self.conv.type(dtype)
class dcgan(gen_base):
    """docstring for fc"""
    def __init__(self, nz=32, ngf=32, nc=3):
        super(dcgan, self).__init__()
        self.nz = nz
        self.conv = torch.nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 4, 4, 1, 0),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1),
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 1,     ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 3, 1, 1),
            nn.Sigmoid()
            # state size. (nc) x 32 x 32
        )

    
    
    
