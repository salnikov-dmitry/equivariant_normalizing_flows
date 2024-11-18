import torch
from torch import nn
from cyclic_group import CyclicGroup

class EquivariantLinearLayer(nn.Module):
    def __init__(self, n, device):
        super().__init__()
        self.n  = n
        self.device = device
        self.Cn = CyclicGroup(n=self.n)
        self.number_of_weights = n//2 + 1
        weight_tensor = torch.Tensor(self.number_of_weights).to(self.device)
        self.weight = nn.Parameter(weight_tensor)
        torch.nn.init.zeros_(self.weight)
        self.matrix = self._matrix().to(self.device)
    
    def _matrix(self):
        mat = torch.zeros((self.number_of_weights, self.n))
        for i in range(2):
            mat[i, i] = torch.tensor(1)
        for i in range(1, self.n//2):
            mat[1+i, 2*i] = torch.tensor(1)
            mat[1+i, 2*i+1] = torch.tensor(1)
        return mat
    
    def forward(self, z):
        weight = self.weight.matmul(self.matrix)
        mat_weight = torch.diag(torch.exp(weight))
        y = z.matmul(mat_weight)
        x = y.matmul(self.Cn.O_inverse.to(self.device))
        log_jacobian = torch.sum(weight)
        return x, log_jacobian