import torch
import numpy as np
from lipreading.preprocess import *
from lipreading.dataset_ud_base import MyDataset
from lipreading.dataset_ud_adapt import AdaptDataset


def pad_packed_collate(batch):
    if len(batch[0]) == 2:
        use_boundary = False
        data_tuple, lengths, labels_tuple = zip(
            *[(a, a.shape[0], b) for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])
    elif len(batch[0]) == 3:
        use_boundary = True
        data_tuple, lengths, labels_tuple, boundaries_tuple = zip(
            *[(a, a.shape[0], b, c) for (a, b, c) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])

    if data_tuple[0].ndim == 1:
        max_len = data_tuple[0].shape[0]
        data_np = np.zeros((len(data_tuple), max_len))
        for idx in range(len(data_np)):
            data_np[idx][:data_tuple[idx].shape[0]] = data_tuple[idx]
    elif data_tuple[0].ndim == 3:
        data_np = data_tuple

    data = torch.FloatTensor(data_np)

    if use_boundary:
        boundaries_np = np.zeros((len(boundaries_tuple), len(boundaries_tuple[0])))
        for idx in range(len(data_np)):
            boundaries_np[idx] = boundaries_tuple[idx]
        boundaries = torch.FloatTensor(boundaries_np).unsqueeze(-1)

    labels = torch.LongTensor(labels_tuple)

    if use_boundary:
        return data, lengths, labels, boundaries
    else:
        return data, lengths, labels


def get_preprocessing_pipelines(modality):
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    if modality == 'video':
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
        preprocessing['train'] = Compose([
            Normalize(0.0, 255.0),
            RandomCrop(crop_size),
            HorizontalFlip(0.5),
            Normalize(mean, std),
            TimeMask(T=0.6 * 25, n_mask=1)
        ])

        preprocessing['val'] = Compose([
            Normalize(0.0, 255.0),
            CenterCrop(crop_size),
            Normalize(mean, std)])

        preprocessing['test'] = preprocessing['val']
        # preprocessing['train'] = preprocessing['val']

    elif modality == 'audio':

        preprocessing['train'] = Compose([
            AddNoise(noise=np.load('./data/babbleNoise_resample_16K.npy')),
            NormalizeUtterance()])

        preprocessing['val'] = NormalizeUtterance()

        preprocessing['test'] = NormalizeUtterance()

    return preprocessing


def get_data_loaders(args):
    preprocessing = get_preprocessing_pipelines(args.modality)

    # create dataset object for each partition
    partitions = ['test'] if args.test else ['train', 'val', 'test']
    dsets = {partition: MyDataset(
        modality=args.modality,
        data_partition=partition,
        data_dir=args.data_dir,
        label_fp=args.label_path,
        annonation_direc=args.annonation_direc,
        preprocessing_func=preprocessing[partition],
        data_suffix='.npz',
        use_boundary=args.use_boundary,
    ) for partition in partitions}
    dset_loaders = {x: torch.utils.data.DataLoader(
        dsets[x],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pad_packed_collate,
        pin_memory=True,
        num_workers=args.workers,
        worker_init_fn=np.random.seed(1)) for x in partitions}
    return dset_loaders


def get_adp_data_loaders(args):
    preprocessing = get_preprocessing_pipelines(args.modality)

    # create dataset object for each partition
    partitions = ['test'] if args.test else ['train', 'val', 'test']
    dsets = {partition: AdaptDataset(
        modality=args.modality,
        data_partition=partition,
        data_dir=args.data_dir,
        label_fp=args.label_path,
        subject=args.subject,
        adapt_min=args.adapt_min,
        fold=args.fold,
        annonation_direc=args.annonation_direc,
        preprocessing_func=preprocessing[partition],
        use_boundary=args.use_boundary,
    ) for partition in partitions}
    dset_loaders = {x: torch.utils.data.DataLoader(
        dsets[x],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pad_packed_collate,
        pin_memory=True,
        num_workers=args.workers,
        worker_init_fn=np.random.seed(1)) for x in partitions}
    return dset_loaders
