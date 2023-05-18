from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import gg_collate

def get_dataset_class(name):
    if name == "genea2022":
        from data_loaders.gesture.data.dataset import Genea2022
        return Genea2022
    elif name == "genea2023":
        from data_loaders.gesture.data.dataset import Genea2023
        return Genea2023
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if name in ['genea2022', 'genea2023']:
        return gg_collate
    else:
        return all_collate

def get_dataset(name, num_frames, split='train', hml_mode='train'):
    DATA = get_dataset_class(name)
    dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train'):
    dataset = get_dataset(name, num_frames, split, hml_mode)
    collate = get_collate_fn(name, hml_mode)
    
    shuffled = True if split == 'train' else False
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffled,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader