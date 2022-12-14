## Test Code to check whether Phoenix Dataset Classes are working
import torch
from openhands.datasets.continuous import Phoenix14TDataset

# dataset = Phoenix14Dataset(
#           train_file = '/data/OpenHands/openhands/datasets/assets/phoenix14_metadata/train.corpus.csv',
#           split_file = '/data/OpenHands/openhands/datasets/assets/phoenix14_metadata/dev.corpus.csv',
#           root_dir = '/data/cslr_datasets/PHOENIX-2014/phoenix2014-release/phoenix-2014-multisigner/',
#           modality = 'rgb',
#           splits = 'val',
#           transforms = None
# )

dataset = Phoenix14TDataset(
          train_file = '/data/OpenHands/openhands/datasets/assets/phoenix14-t_metadata/PHOENIX-2014-T.train.corpus.csv',
          split_file = '/data/OpenHands/openhands/datasets/assets/phoenix14-t_metadata/PHOENIX-2014-T.dev.corpus.csv',
          root_dir = '/data/cslr_datasets/PHOENIX-2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/',
          modality = 'rgb',
          splits = 'val',
          transforms = None
)


dl = torch.utils.data.DataLoader(
    dataset=dataset,
    collate_fn=dataset.collate_fn,
    batch_size= 2,
    shuffle= False,
    num_workers= 1,
    pin_memory= True,
    drop_last= False,
)


d = next(iter(dl))
print('frames: ', d['frames'].shape)
print('gloss: ', d['gloss'].shape)
print('text: ', d['text'].shape)
print('files: ', len(d['files']))
print('dataset_names: ', len(d['dataset_names']))

