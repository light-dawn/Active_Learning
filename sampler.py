class HybridSampler(Sampler):
    def __init__(self, budget):
        self.budget = budget
    
    def sample(self, lossnet, data_loader, device):
        raise NotImplementedError
        

