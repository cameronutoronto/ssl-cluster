
from base import model
import torch 


class cnn_base(model):
    def __init__(self):
        super(cnn_base, self).__init__()
    def eval(self):
        self.fc.eval()
        self.conv.eval()
    def train(self):
        self.fc.train()
        self.conv.train()
    def parameters(self):
        return list(self.conv.parameters())+list(self.fc.parameters())
    def zero_grad(self):
        self.fc.zero_grad()
        self.conv.zero_grad()
    def forward(self, input):
        input = input.permute(0,3,1,2)
        tmp = self.conv(input)
        tmp = tmp.view(-1,self.fc_dim)
        return self.fc(tmp)

class cnn(cnn_base):
    """docstring for fc"""
    def __init__(self, Hn, input_dim=[28,28,1],output_dim=10):
        super(cnn, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[-1], Hn, 3,padding=1),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn, Hn, 3,padding=1),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn, Hn, 3,padding=1),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.ReLU(),
        )
        D = input_dim[0]
        for _ in xrange(3):
            D = D//2

        self.fc_dim = Hn * D*D
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.fc_dim, Hn),
            torch.nn.ReLU(),
            torch.nn.Linear(Hn, output_dim),
        )


    
    
    


class cnn2(cnn_base):
    """docstring for fc"""
    def __init__(self, Hn=1, input_dim=[28,28,1],output_dim=10,dropout=0):
        super(cnn2, self).__init__()
        hid1=48
        hid2=96
        hid3=96
        hidfc=256
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[-1], Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid1, Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid1, Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(Hn*hid1, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(Hn*hid2, Hn*hid3, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid3, Hn*hid3, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid3, Hn*hid3, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.Dropout(dropout),
            #
        )
        D = input_dim[0]
        for _ in xrange(3):
            D = D//2

        self.fc_dim = Hn*hid3 * D*D
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.fc_dim, hidfc),
            torch.nn.ReLU(),
            torch.nn.Dropout(.1),
            torch.nn.Linear(hidfc, output_dim),
            # torch.nn.Tanh(),
        )


class cnn_bn(cnn_base):
    """docstring for fc"""
    def __init__(self, Hn=1, input_dim=[28,28,1],output_dim=10):
        super(cnn_bn, self).__init__()
        hid1=48
        hid2=96
        hid3=144
        hidfc=256
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[-1], Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid1, Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid1, Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.BatchNorm2d(Hn*hid1),
            torch.nn.Conv2d(Hn*hid1, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.BatchNorm2d(Hn*hid2),
            torch.nn.Conv2d(Hn*hid2, Hn*hid3, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid3, Hn*hid3, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid3, Hn*hid3, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.BatchNorm2d(Hn*hid3),
            #
        )
        D = input_dim[0]
        for _ in xrange(3):
            D = D//2

        self.fc_dim = Hn*hid3 * D*D
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.fc_dim, hidfc),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidfc),
            torch.nn.Linear(hidfc, output_dim),
            # torch.nn.Tanh(),
        )


class cnn_bn_db(cnn_base):
    """docstring for fc"""
    def __init__(self, Hn=1, input_dim=[28,28,1],output_dim=10):
        super(cnn_bn_db, self).__init__()
        hid1=32
        hid2=32
        hid3=64
        hidfc=64
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[-1], Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(Hn*hid1, Hn*hid1, 3,padding=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(Hn*hid1, Hn*hid1, 3,padding=1),
            # torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.BatchNorm2d(Hn*hid1),
            torch.nn.Conv2d(Hn*hid1, Hn*hid2, 3,padding=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.BatchNorm2d(Hn*hid2),
            torch.nn.Conv2d(Hn*hid2, Hn*hid3, 3,padding=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(Hn*hid3, Hn*hid3, 3,padding=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(Hn*hid3, Hn*hid3, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.BatchNorm2d(Hn*hid3),
            #
        )
        D = input_dim[0]
        for _ in xrange(3):
            D = D//2

        self.fc_dim = Hn*hid3 * D*D
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.fc_dim, hidfc),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidfc),
            torch.nn.Linear(hidfc, output_dim),
            # torch.nn.Tanh(),
        )


class cnn_globe(cnn_base):
    """architecture of the discriminator of improved gan paper
        only difference is that I'm not using leakyrelu
    """
    def __init__(self, Hn=1, input_dim=[28,28,1],output_dim=10):
        super(cnn_globe, self).__init__()
        hid1=96
        hid2=192
        hidfc=256
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = torch.nn.Sequential(
            torch.nn.Dropout(.2),
            torch.nn.Conv2d(input_dim[-1], Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(Hn*hid1),
            torch.nn.Conv2d(Hn*hid1, Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(Hn*hid1),
            torch.nn.Conv2d(Hn*hid1, Hn*hid1,3,stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(Hn*hid1),
            torch.nn.Dropout(.5),
            torch.nn.Conv2d(Hn*hid1, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(Hn*hid2),
            torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(Hn*hid2),
            torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(Hn*hid2),
            torch.nn.Dropout(.5),
            torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(Hn*hid2),
            ## 1x1 convs on 6x6 images
            torch.nn.Conv2d(Hn*hid2, Hn*hid2, 1,padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(Hn*hid2),
            torch.nn.Conv2d(Hn*hid2, Hn*hid2, 1,padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(Hn*hid2),
        )
        self.fc_dim = Hn*hid2
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.fc_dim, output_dim),
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(hidfc),
            # torch.nn.Linear(hidfc, output_dim),
            # torch.nn.Tanh(),
        )
    def forward(self, input):
        N = input.size()[0]
        input = input.permute(0,3,1,2)
        tmp = self.conv(input)
        ## (N, fc_dim, 6,6) -> (N, hid2, -1)
        tmp = tmp.view(N, self.fc_dim, -1)
        ## average pool across channels
        tmp = tmp.mean(-1).view(N, self.fc_dim)
        return self.fc(tmp)