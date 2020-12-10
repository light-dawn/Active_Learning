import numpy as np
import random
from contextlib import contextmanager
import json
from models import unet
from utils.network import register_embedding_hook


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
             # call .item() to get the value of the indices instead of type Tensor
            all_indices.extend([index.item() for index in indices])
        query_indices = random.sample(all_indices, self.budget)
        return query_indices


class HybridSampler:
    def __init__(self, budget, lossnet, task_model, device):
        self.budget = budget
        self.task_model = task_model
        self.lossnet = lossnet
        self.device = device

    def get_embedding_layer(self):
        return list(self.model.children())[4]

    def get_feature_embedding_and_pred_loss(self):
        self.model.eval()
        self.lossnet.eval()
        batch_embeddings = []
        all_indices = []
        with torch.no_grad(), register_embedding_hook(self.get_embedding_layer(), batch_embeddings):
            embeddings = torch.tensor([], device=self.device)
            all_pred_loss = torch.tensor([], device=self.device)
            for data, _, indices in dataloader:
                all_indices.extend(indices)
                data = data.to(self.device)
                _, feature_maps = self.model(data)
                pred_loss = self.lossnet(feature_maps)
                embeddings = torch.cat([embeddings, batch_embeddings.pop()])
                all_pred_loss = torch.cat([all_pred_loss, pred_loss])
                assert len(batch_embeddings) == 0, "Pop batch embeddings failed."
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        return embeddings, all_pred_loss, np.asarray(all_indices)

    def sample(self, labeled_dataloader, unlabeled_dataloader):
        embedding_labeled, _, _ = self.get_feature_embedding(labeled_dataloader)
        # The indices of the unlabeled embedding and the unlabeled indices are matched.
        embedding_unlabeled, all_pred_loss, unlabeled_indices = self.get_feature_embedding(unlabeled_dataloader)
        labeled_centroid = embedding_labeled.mean(0)
        new_items = torch.zero_like(labeled_centroid, device=self.device)
        remaining_unpicked = torch.ones(embedding_unlabeled.shape[0], dtype=torch.bool, device=self.device)
        points_to_label = torch.empty(self.budget, dtype=torch.long, device=self.device)
        N = embedding_labeled.shape[0]  # number of previously labeled items
        M = 0  # new labeled count
        for i in range(self.budget):
            cur_centroid = (N - 1) / (N + M) * labeled_centroid + 1 / (M + N) * new_items
            unlabeled_items = embedding_unlabeled[remaining_unpicked]
            all_pred_loss = all_pred_loss[remaining_unpicked]
            dists = torch.norm(unlabeled_items - cur_centroid, p=2, dim=1)
            dists_argsort = np.argsort(dists.numpy())
            loss_argsort = np.argsort(all_pred_loss.numpy())
            assert len(dists_argsort) == len(loss_argsort)
            ranks = torch.empty(len(dists_args), dtype=torch.long, device=self.device)
            for i in range(len(ranks)):
                ranks[i] = np.argwhere(dists_argsort == i)[0][0] + np.argwhere(loss_argsort == i)[0][0]
            selected_point = ranks.argmax()
            new_items += unlabeled_items[selected_point]
            M += 1
            points_to_label[i] = unlabeled_indices[selected_point]
            _tmp = torch.arange(remaining_unpicked.shape[0], device=self.device)
            _tmp2 = _tmp[remaining_unpicked][selected_point]
            assert remaining_unpicked[_tmp2] == 1, "Can only select 1 point in each iteration."
            remaining_unpicked[_tmp2] = 0
            assert remaining_unpicked[_tmp2] == 0, "Label new point failed."
        assert (~remaining_unpicked).sum() == self.budget, "The number of queried indices does not match the budget."
        query_indices = points_to_label.data
        return query_indices


class OMedALSampler:
    def __init__(self, budget, task_model, device):
        self.budget = budget
        self.model = task_model
        self.device = device

    def get_feature_embedding(self, dataloader):
        self.model.eval()
        batch_embeddings = []
        all_indices = []
        with torch.no_grad(), register_embedding_hook(self.get_embedding_layer(), batch_embeddings):
            embeddings = torch.tensor([]).to(self.device)
            for data, _, indices in dataloader:
                all_indices.extend(indices)
                data = data.to(self.device)
                self.model(data)
                embeddings = torch.cat([embeddings, batch_embeddings.pop()])
                assert len(batch_embeddings) == 0, "Pop batch embeddings failed."
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        return embeddings, np.asarray(all_indices)
        
    def get_embedding_layer(self):
        return list(self.model.children())[4]

    def sample(self, labeled_dataloader, unlabeled_dataloader):
        embedding_labeled, _ = self.get_feature_embedding(labeled_dataloader)
        # The indices of the unlabeled embedding and the unlabeled indices are matched.
        embedding_unlabeled, unlabeled_indices = self.get_feature_embedding(unlabeled_dataloader)
        labeled_centroid = embedding_labeled.mean(0)
        new_items = torch.zero_like(labeled_centroid, device=self.device)
        remaining_unpicked = torch.ones(embedding_unlabeled.shape[0], dtype=torch.bool, device=self.device)
        points_to_label = torch.empty(self.budget, dtype=torch.long, device=self.device)
        N = embedding_labeled.shape[0]  # number of previously labeled items
        M = 0  # new labeled count
        for i in range(self.budget):
            cur_centroid = (N - 1) / (N + M) * labeled_centroid + 1 / (M + N) * new_items
            unlabeled_items = embedding_unlabeled[remaining_unpicked]
            dists = torch.norm(unlabeled_items - cur_centroid, p=2, dim=1)
            selected_point = dists.argmax()
            new_items += unlabeled_items[selected_point]
            M += 1
            points_to_label[i] = unlabeled_indices[selected_point]
            _tmp = torch.arange(remaining_unpicked.shape[0], device=self.device)
            _tmp2 = _tmp[remaining_unpicked][selected_point]
            assert remaining_unpicked[_tmp2] == 1, "Can only select 1 point in each iteration."
            remaining_unpicked[_tmp2] = 0
            assert remaining_unpicked[_tmp2] == 0, "Label new point failed."
        assert (~remaining_unpicked).sum() == self.budget, "The number of queried indices does not match the budget."
        query_indices = points_to_label.data
        return query_indices


class LossPredictionSampler:
    def __init__(self, budget, lossnet, device):
        self.budget = budget
        self.lossnet = lossnet
        self.device = device

    def sample(self, dataloader):
        lossnet.eval()
        all_indices, all_pred_loss = [], []
        for data, _, indices in dataloader:
            with torch.no_grad():
                data = data.to(self.device)
                pred_loss = self.lossnet(data)
            pred_loss = pred_loss.cpu().data
            print("Type pred_loss: ", type(pred_loss))
            all_pred_loss.extend(pred_loss)
            all_indices.extend([index.item() for index in indices])
        all_pred_loss = torch.stack(all_pred_loss)
        print("all_pred_loss size: ", all_pred_loss.size())
        all_pred_loss = all_pred_loss.view(-1)
        print("all_pred_loss size: ", all_pred_loss.size())
        # (values, indices)
        _, topk_indices = torch.topk(all_pred_loss, self.budget)
        query_indices = np.asarray(all_indices)[topk_indices]
        return query_indices


class CoreSetSampler:
    def __init__(self, budget):
        self.budget = budget

    def sample(self, dataloader):
        raise NotImplementedError


class EntropySampler:
    def __init__(self, budget):
        self.budget = budget

    def sample(self, dataloader):
        raise NotImplementedError


if __name__ == "__main__":
    with open("demo_cfg.json", "r") as f:
        config = json.loads(f.read())
    model = unet.UNet(n_channels=config["model"]["n_channels"], n_classes=config["model"]["n_classes"])
    sampler = OMedALSampler(100, model)
    layer = sampler.get_embedding_layer()
    print(layer)
    # print(len(layer))


        
        


            


