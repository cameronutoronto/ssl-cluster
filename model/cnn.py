
from base import model
import torch 

class cnn(model):
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


    def forward(self, input):
        input = input.permute(0,3,1,2)
        tmp = self.conv(input)
        tmp = tmp.view(-1,self.fc_dim)
        return self.fc(tmp)
    def parameters(self):
        return list(self.conv.parameters())+list(self.fc.parameters())
    def zero_grad(self):
        self.fc.zero_grad()
        self.conv.zero_grad()
    


class cnn2(model):
    """docstring for fc"""
    def __init__(self, Hn=1, input_dim=[28,28,1],output_dim=10):
        super(cnn2, self).__init__()
        hid1=48
        hid2=96
        hid3=144
        hidfc=256
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[-1], Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(Hn*hid1, Hn*hid1, 3,padding=1),
            # torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid1, Hn*hid1, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.Conv2d(Hn*hid1, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=1),
            # torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid2, Hn*hid2, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.Conv2d(Hn*hid2, Hn*hid3, 3,padding=1),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(Hn*hid3, Hn*hid3, 3,padding=1),
            # torch.nn.ReLU(),
            torch.nn.Conv2d(Hn*hid3, Hn*hid3, 3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            #
        )
        D = input_dim[0]
        for _ in xrange(3):
            D = D//2

        self.fc_dim = Hn*hid3 * D*D
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.fc_dim, hidfc),
            torch.nn.ReLU(),
            torch.nn.Linear(hidfc, output_dim),
            # torch.nn.Tanh(),
        )


    def forward(self, input):
        input = input.permute(0,3,1,2)
        tmp = self.conv(input)
        tmp = tmp.view(-1,self.fc_dim)
        return self.fc(tmp)
    def parameters(self):
        return list(self.conv.parameters())+list(self.fc.parameters())
    def zero_grad(self):
        self.fc.zero_grad()
        self.conv.zero_grad()
    