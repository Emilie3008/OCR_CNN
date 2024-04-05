import torch.nn as nn
import torch

class C1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(C1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel)

    def forward(self, x):
        return self.conv(x)
    
class S2(nn.Module):
    def __init__(self, out_channels, kernel=(2, 2)):
        super(S2, self).__init__()
        self.kernel = kernel
        self.coefficient = nn.Parameter(torch.randn(out_channels))
        self.bias = nn.Parameter(torch.randn(out_channels))
 
    def forward(self, x):
        x = nn.functional.avg_pool2d(x, kernel_size=self.kernel, stride=self.kernel)*self.kernel[0]*self.kernel[1]
        coeff =  self.coefficient.view(1, -1, 1, 1)
        biais = self.bias.view(1, -1, 1, 1)
        x = x *coeff + biais
        return x
    
class C3(nn.Module):
    
    map_S2 = [[0, 1, 2],
              [1, 2, 3],
              [2, 3, 4],
              [3, 4, 5],
              [0, 4, 5],
              [0, 1, 5],
              [0, 1, 2, 3],
              [1, 2, 3, 4],
              [2, 3, 4, 5],
              [0, 3, 4, 5],
              [0, 1, 4, 5],
              [0, 1, 2, 5],
              [0, 1, 3, 4],
              [1, 2, 4, 5],
              [0, 2, 3, 5],
              [0, 1, 2, 3, 4, 5]
              ]
    
    def __init__(self, in_channels, out_channels, kernel = 5, connected=map_S2):
        super(C3, self).__init__()
        self.connections = connected
        assert len(self.connections)==out_channels and len(self.connections[-1])==in_channels   

        self.kernel = kernel

    def forward(self, x):
        batch_size, _, height, width = x.size()
        output = torch.zeros(batch_size, len(self.connections), height - self.kernel + 1, width - self.kernel + 1, device=x.device)

        for i, connected_indices in enumerate(self.connections):
            connected_input = x[:, connected_indices, :, :]
            output[:, i:i+1, :, :] =  nn.Conv2d(len(self.connections[i]), 1,
                                                 kernel_size=self.kernel)(connected_input)
        return output
    

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.c1 = C1(in_channels=1, out_channels=6, kernel=5)
        self.s2 = S2(out_channels=6, kernel=(2, 2))
        self.c3 = C3(in_channels=6, out_channels=16)
        self.s4 = S2(out_channels=16, kernel=(2, 2))
        self.c5 = nn.Linear(5*5*16, 120)
        self.c6 = nn.Linear(120, 84)
        self.output_layer = nn.Linear(84, 10)

        
    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = self.s2(x)
        x = torch.relu(self.c3(x))
        x = self.s4(x)

        # Flatten before Dense Layer
        x = x.view(-1, 16*5*5)
        x = torch.relu(self.c5(x))
        x = torch.relu(self.c6(x))
        x = self.output_layer(x)
        return x
    
def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


# leNet5 = LeNet5()

# # Affichage du nombre de paramètres pour chaque couche
# print("Nombre de paramètres pour la couche C1:", count_parameters(leNet5_.c1))
# print("Nombre de paramètres pour la couche S2:", count_parameters(leNet5_.s2))
# print("Nombre de paramètres pour la couche C3:", count_parameters(leNet5_.c3))
# print("Nombre de paramètres pour la couche S4:", count_parameters(leNet5_.s4))
# print("Nombre de paramètres pour la couche C5:", count_parameters(leNet5_.c5))
# print("Nombre de paramètres pour la couche F6:", count_parameters(leNet5_.c6))
# print("Nombre de paramètres pour la couche de sortie:", count_parameters(leNet5_.output_layer))