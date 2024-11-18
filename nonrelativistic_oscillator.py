import torch
from cyclic_group import CyclicGroup

class NonRelativisticOscillator:
    def __init__(self, tau, number_of_nodes, device):
        self.tau = tau
        self.n = number_of_nodes
        self.beta = tau * number_of_nodes
        self.K  = lambda x: x**2 / 2 / self.tau
        self.V  = lambda x: self.tau * x**2 / 2
        self.Cn = CyclicGroup(n=self.n)
        self.device = device
    
    def Action(self, x):
        return torch.sum(self.K(x - torch.roll(x,1,1)) + self.V(x), axis=1)
  
    def Loss(self, x, log_jacobian):
        return self.Action(x).mean() - log_jacobian
    
    def Eigenvalues(self):
        res = torch.zeros(self.n)
        res[0] = self.tau
        res[1] = self.tau + (4/self.tau)
        for i in range(1, self.n//2):
            temp_ = self.tau + (4/self.tau)*(torch.sin(torch.tensor(torch.pi*i/self.n)))**2
            res[2*i]   = temp_
            res[2*i+1] = temp_
        return res.to(self.device)
    
    def MinLoss(self):
        return 0.5*(self.n + torch.sum(torch.log(self.Eigenvalues())))
    
    def AnalyticalLogJacobian(self):
        return torch.sum(torch.log(self.Eigenvalues()**(-0.5))).to(self.device)
    
    def matrixA(self):
        res = (self.tau + 2/self.tau)*torch.eye(self.n)
        res -= self.Cn.regular_representation(1)/self.tau
        res -= self.Cn.regular_representation(-1)/self.tau
        return res.to(self.device)
    
    def AnalyticTransformation(self, z):
        y = z*self.Eigenvalues()**(-0.5)
        x = y.matmul(self.Cn.O_inverse.to(self.device))
        return x
    
    def correlation_function(self, x):
        return lambda s: torch.mean(torch.roll(x,s,1)*x)
    
    def expectation_energy(self, x):
        G0 = self.correlation_function(x)(0)
        G1 = self.correlation_function(x)(1)
        mean_kinetic_energy = 0.5*(1 - 2*(G0 - G1)/self.tau) / self.tau
        mean_potential_energy = 0.5*G0
        mean_energy = mean_kinetic_energy + mean_potential_energy
        return torch.concatenate([torch.unsqueeze(mean_kinetic_energy,0),
                                  torch.unsqueeze(mean_potential_energy,0),
                                  torch.unsqueeze(mean_energy,0)])