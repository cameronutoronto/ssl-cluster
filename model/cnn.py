
from base import model
import torch 

class cnn(model):
    """docstring for fc"""
    def __init__(self, Hn, input_dim=[28,28,1],output_dim=10):
        super(cnn, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, Hn, 3,padding=1),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn, Hn, 3,padding=1),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Hn, Hn, 3,padding=1),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.ReLU(),
        )
        self.fc_dim = Hn * 3*3
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
    
        