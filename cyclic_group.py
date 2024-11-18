import torch

def rotation_matrix(theta):
    return torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                         [torch.sin(theta),  torch.cos(theta)]])

class CyclicGroup:
    def __init__(self, n):
        self.n = n
        self._phi = 2*torch.pi/self.n
        self._arange = torch.arange(self.n).reshape(-1,1)
        self._O()
        # T_irreducible(g) = O^-1 * T_regular(g) * O
        # T_regular(g) = O * T_irreducible(g) * O^-1
        # O^-1 = O^T
        self._flip()
        
    def _O(self):
        res = self._arange.matmul(torch.arange(2).reshape(1,-1))
        res = (-1)**res
        res[:, 1] = (-1)*res[:, 1]
        
        for i in range(1, self.n//2):
            res = torch.concatenate([res, 
                                     (torch.sqrt(torch.tensor([2]))*
                                      torch.cos(i*self._phi*
                                                self._arange))],
                                    axis=1)
            res = torch.concatenate([res, 
                                     (torch.sqrt(torch.tensor([2]))*
                                      torch.sin(i*self._phi*
                                                self._arange))],
                                    axis=1)
            
        self.O = res/torch.sqrt(torch.tensor([self.n]))
        self.O_inverse = self.O.T
        
    def regular_representation(self, k):
        return torch.roll(torch.eye(self.n),-k,0)
    
    def sum_of_irreducible(self,k):
        res = torch.eye(self.n)
        res[:2, :2] = torch.diag(torch.tensor([1, (-1)**k]))
        for i in range(1, self.n//2):
            res[2*i:2*(i+1), 2*i:2*(i+1)] = rotation_matrix(-torch.tensor(i*k*self._phi))
        return res
    
    def _flip(self):
        res = torch.eye(self.n)
        res[1:, :] = res[1:, :].flip(0)
        self.flip_regular = res
        self.flip_sum_of_irreducible = self.O_inverse.matmul(res.matmul(self.O))