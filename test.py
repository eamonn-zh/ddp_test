import torch
import torch.nn as nn
from tqdm import tqdm
import torch.distributed as dist
import torch.utils.data.distributed
import sys
import time


# dataset
class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.data = torch.randn(100, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return 100

# model
class Model(nn.Module):

    def __init__(self, input_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, input):
        return self.fc(input)


if __name__=="__main__":

    # check the nccl backend
    if not dist.is_nccl_available():
        print("Error: nccl backend not available.")
        sys.exit(1)

    input_size = 5
    batch_size = 30

    # prepare the dataset
    print("Init dataset")
    dataset = RandomDataset(input_size)

    # get the process rank and the world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print("Init dataset sampler")
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    
    print("Init dataloader")
    rand_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size // world_size, sampler=train_sampler)

    print("Init model")
    model = Model(input_size)
    
    print("Init nccl group")
    dist.init_process_group(backend="nccl", init_method="env://")

    device = torch.device('cuda', rank)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    print("Start training ...")
    for data in tqdm(rand_loader):
        input = data.to(device)
        output = model(input)
        time.sleep(1)
