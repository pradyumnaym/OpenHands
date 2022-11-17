import torch
from openhands.datasets.continuous import DGSDataset

dataset = DGSDataset(
        train_file= "/data/cslr_datasets/DGS_CORPUS/final_dataset.csv",
        split_file= "/data/cslr_datasets/DGS_CORPUS/final_dataset.csv",
        root_dir= "/data/cslr_datasets/DGS_CORPUS/",
        modality= "rgb",
        splits= "train",
        use_english= False,
        transforms=None
)

dl = torch.utils.data.DataLoader(
    dataset=dataset,
    collate_fn=dataset.collate_fn,
    batch_size= 32,
    shuffle= True,
    num_workers= 1,
    pin_memory= True,
    drop_last= False,
)

print(next(iter(dl)))
