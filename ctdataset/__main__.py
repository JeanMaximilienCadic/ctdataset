if __name__ == "__main__":
    import argparse
    from ctdataset.dataset import CTDataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    parser = argparse.ArgumentParser(description='Process some integers.')
    ########################################### LOADER RELATED #########################################################
    parser.add_argument('ply_folder')
    ################################################ HP ################################################################
    parser.add_argument('--bs', default=10, type=int, help="100,30")
    parser.add_argument('--no_shuffle', action="store_true")
    parser.add_argument('--dense', action="store_true")
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--no_caching', action="store_true", type=str)

    args = parser.parse_args()
    # Loader
    dataset = CTDataset(ply_folder=args.ply_folder,
                        dim=args.dim,
                        caching=not args.no_caching)
    # Model variables
    loader = DataLoader(dataset=dataset,
                        batch_size=args.bs,
                        num_workers=args.num_workers,
                        shuffle=not args.no_shuffle,
                        persistent_workers=True,
                        pin_memory=True,
                        drop_last=True)

    for epoch in range(10):
        for i, (x, y, _, _, _) in tqdm(enumerate(loader), total=len(loader), desc=f">> Epoch {epoch}"):
            x = x.to(args.device)
            y = y.to(args.device)
