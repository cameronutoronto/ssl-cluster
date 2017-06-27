
from base import model
import torch 

class fc(model):
    """docstring for fc"""
    def __init__(self, Hn, input_dim=28*28,output_dim=10):
        super(fc, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, Hn),
            torch.nn.ReLU(),
            torch.nn.Linear(Hn, Hn),
            torch.nn.ReLU(),
            torch.nn.Linear(Hn, Hn),
            torch.nn.ReLU(),
            torch.nn.Linear(Hn, output_dim),
        )
        self.model = model

    def forward(self, input):
        return self.model(input.view(-1,self.input_dim))

    def parameters(self):
        return self.model.parameters()
        
    def zero_grad(self):
        self.model.zero_grad()

    def eval(self):
        self.model.eval()
    def train(self):
        self.model.train()