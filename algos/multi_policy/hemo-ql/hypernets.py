import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """Code adapted from the original implementation of the paper:

    Recomposing the Reinforcement Learning Building-Blocks with Hypernetworks. 2021. 
    https://arxiv.org/abs/2106.06842
    https://github.com/keynans/HypeRL
    """
    def __init__(self, latent_dim, output_dim_in, output_dim_out, sttdev):
        super(Head, self).__init__()
        
        latent_dim = 1024
        self.output_dim_in = output_dim_in
        self.output_dim_out = output_dim_out
     
        self.W1 = nn.Linear(latent_dim, output_dim_in * output_dim_out)
        self.b1 = nn.Linear(latent_dim, output_dim_out)
        self.s1 = nn.Linear(latent_dim, output_dim_out)

        self.init_layers(sttdev)

    def forward(self, x):

        # weights, bias and scale for dynamic layer
        w = self.W1(x).view(-1, self.output_dim_out, self.output_dim_in)
        b = self.b1(x).view(-1, self.output_dim_out, 1)
        s = 1. + self.s1(x).view(-1, self.output_dim_out, 1) 
                    
        return w, b, s
   
    def init_layers(self, stddev):

        torch.nn.init.uniform_(self.W1.weight, -stddev, stddev)
        torch.nn.init.uniform_(self.b1.weight, -stddev, stddev)
        torch.nn.init.uniform_(self.s1.weight, -stddev, stddev)

        torch.nn.init.zeros_(self.W1.bias)
        torch.nn.init.zeros_(self.b1.bias)
        torch.nn.init.zeros_(self.s1.bias)