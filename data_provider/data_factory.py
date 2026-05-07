from torch.utils.data import DataLoader

from data_provider.data_loader import (
    Dataset_Custom,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Futures,
)

data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "custom": Dataset_Custom,
    "Futures": Dataset_Futures,
}


def data_provider(args, flag):
    dataset_cls = data_dict[args.data]
    dataset = dataset_cls(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=1,
        freq=args.freq,
    )
    print(flag, len(dataset))
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=flag != "test",
        num_workers=args.num_workers,
        drop_last=False,
    )
    return dataset, data_loader
