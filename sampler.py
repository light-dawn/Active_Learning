import numpy as np
import random

# class HybridSampler:
#     def __init__(self, budget):
#         self.budget = budget
    
#     def sample(self, lossnet, data_loader, device):
#         raise NotImplementedError

class SamplerUtils:
    def get_sampler(sampler_name, budget):
        if sampler_name == "random":
            return RandomSampler(budget)
        elif sampler_name == "hybrid":
            return HybridSampler(budget)
        elif sampler_name == "diversity":
            return DiversitySampler(budget)
        elif sampler_name == "loss":
            return LossPredictionSampler(budget)
        elif sampler_name == "coreset":
            return CoreSetSampler(budget)
        elif sampler_name == "entropy":
            return EntropySampler(budget)
        else:
            return None


class RandomSampler:
    def __init__(self, budget):
        self.budget = budget

    def sample(self, dataloader):
        all_indices = []
        for _, _, indices in dataloader:
            all_indices.extend(indices)
        query_indices = random.sample(all_indices, self.budget)
        # call .item() to get the value of the indices instead of type Tensor
        query_indices = [index.item() for index in query_indices]
        return query_indices


class HybridSampler:
    def __init__(self, budget):
        self.budget = budget

    def sample(self, dataloader):
        all_indices = []
        for _, _, indices in dataloader:
            all_indices.extend(indices)
        query_indices = random.sample(all_indices, self.budget)
        return query_indices


class DiversitySampler:
    def __init__(self, budget):
        self.budget = budget

    def sample(self, dataloader):
        all_indices = []
        for _, _, indices in dataloader:
            all_indices.extend(indices)
        query_indices = random.sample(all_indices, self.budget)
        return query_indices


class LossPredictionSampler:
    def __init__(self, budget):
        self.budget = budget

    def sample(self, dataloader):
        all_indices = []
        for _, _, indices in dataloader:
            all_indices.extend(indices)
        query_indices = random.sample(all_indices, self.budget)
        return query_indices


class CoreSetSampler:
    def __init__(self, budget):
        self.budget = budget

    def sample(self, dataloader):
        all_indices = []
        for _, _, indices in dataloader:
            all_indices.extend(indices)
        query_indices = random.sample(all_indices, self.budget)
        return query_indices



class EntropySampler:
    def __init__(self, budget):
        self.budget = budget

    def sample(self, dataloader):
        all_indices = []
        for _, _, indices in dataloader:
            all_indices.extend(indices)
        query_indices = random.sample(all_indices, self.budget)
        return query_indices


        
        


            


