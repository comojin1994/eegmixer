from torch.utils.data import DataLoader, ConcatDataset
from typing import Optional
from datasets.datasets.bcic2a import BCIC2a
from datasets.datasets.bcic2b import BCIC2b
from datasets.datasets.zhou import Zhou
import datasets.eeg_transforms as e_transforms

dataset_dict = {
    "bcic2a": BCIC2a,
    "bcic2b": BCIC2b,
    "zhou": Zhou,
}

transform_dict = {
    "bcic2a": {
        "train": [
            e_transforms.ToTensor(),
        ],
        "test": [
            e_transforms.ToTensor(),
        ],
    },
    "bcic2b": {
        "train": [
            e_transforms.ToTensor(),
        ],
        "test": [
            e_transforms.ToTensor(),
        ],
    },
    "zhou": {
        "train": [
            e_transforms.ToTensor(),
        ],
        "test": [
            e_transforms.ToTensor(),
        ],
    },
}


class DatasetMaker:

    def __init__(self, dataset_name):
        self.dataset = dataset_dict[dataset_name]
        self.transform = transform_dict[dataset_name]

    def load_dataset(self, args, target_subject: Optional[int] = None):
        train_dataset = self.dataset(
            args=args,
            target_subject=target_subject,
            is_test=False,
            transform=e_transforms.Compose(self.transform["train"]),
        )

        test_dataset = self.dataset(
            args=args,
            target_subject=target_subject,
            is_test=True,
            transform=e_transforms.Compose(self.transform["test"]),
        )

        return train_dataset, test_dataset

    def load_data(self, args, target_subject: Optional[int] = None):
        train_dataset = self.dataset(
            args=args,
            target_subject=target_subject,
            is_test=False,
            transform=e_transforms.Compose(self.transform["train"]),
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        test_dataset = self.dataset(
            args=args,
            target_subject=target_subject,
            is_test=True,
            transform=e_transforms.Compose(self.transform["test"]),
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return train_dataloader, test_dataloader


def load_data(target_subject, args):
    if args.sub_datasets is None:
        return load_single_data(target_subject, args)
    else:
        return load_multi_data(target_subject, args)


def load_single_data(target_subject, args):

    dataset = DatasetMaker(args.dataset)
    train_dataloader, test_dataloader = dataset.load_data(args, target_subject)

    return train_dataloader, test_dataloader


def load_dataloader(train_dataset, test_dataset, args):

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataloader, test_dataloader


def load_multi_data(target_subject, args):

    datasets = []
    main_train_dataset, test_dataset = DatasetMaker(args.dataset).load_dataset(
        args, target_subject
    )
    datasets.append(main_train_dataset)

    for dataset in args.sub_datasets:
        args.mode = "ssl"
        sub_train_dataset, _ = DatasetMaker(dataset).load_dataset(args, target_subject)
        datasets.append(sub_train_dataset)
    train_dataset = ConcatDataset(datasets)

    train_dataloader, test_dataloader = load_dataloader(
        train_dataset, test_dataset, args
    )

    return train_dataloader, test_dataloader
