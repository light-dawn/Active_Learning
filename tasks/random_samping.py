from tasks.base_tasks import DeepActiveTask
from sampler import RandomSampler 


class RandomSegPipeline(DeepActiveTask):
    def __init__(self, task_name, model, dataset, optimizer, criterion, epochs, batch_size, 
                 init_budget, budget, cycles):
        super().__init__(task_name, model, dataset, optimizer, criterion, epochs, batch_size, 
                         init_budget, budget, cycles)
        self.sampler = RandomSampler(self.budget)
    
    def run(self):
        for cycle in range(self.cycles):
            print("Cycle: ", cycle + 1)
            # 构造DataLoader
            labeled_loader = self.get_cur_data_loader(part="labeled")
            unlabeled_loader = self.get_cur_data_loader(part="unlabeled")
            self.train(labeled_loader)
            query_indices = self.sampler.sample(unlabeled_loader)
            self.query_and_move(query_indices)
            print("Query and move data...")



