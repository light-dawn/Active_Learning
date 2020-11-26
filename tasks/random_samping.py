from tasks.base_tasks import DeepActiveSegTask, DeepActiveClaTask
from sampler import RandomSampler 


class RandomSegPipeline(DeepActiveSegTask):
    def __init__(self, task_name, model, dataset, optimizer, criterion, epochs, batch_size, 
                 init_budget, budget, cycles):
        super().__init__(task_name, model, dataset, optimizer, criterion, epochs, batch_size, 
                         init_budget, budget, cycles)
        self.sampler = RandomSampler(budget)
    
    def run(self):
        # 构造DataLoader
        train_loader = self.get_cur_train_loader()
        for cycle in self.cycles:
            self.train(train_loader)
            query_indices = self.sampler.sample(self.budget)
            self.query_and_move()



