# CTDataset
![](img/image.png)

<p align="center">
  <a href="#code-structure">Code</a> •
  <a href="#how-to">How To Use</a> •
  <a href="#docker">Docker </a> •
</p>


### Code structure
```python
from setuptools import setup
from ctdataset import __version__

setup(
    name='ctdataset',
    version=__version__,
    long_description="",
    packages=[
        "ctdataset",
        "ctdataset.dataset",
    ],
    include_package_data=True,
    url='https://github.com/JeanMaximilienCadic/ctnet',
    license='MIT',
    author='Jean Maximilien Cadic',
    python_requires='>=3.6',
    install_requires=[d.rsplit()[0] for d in open("requirements.txt").readlines()],
    author_email='support@cadic.jp',
    description='GNU Tools for python',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
    ]
)


```

The main execution:
```python
import argparse
from ctdataset.dataset import CTDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from ctnet import CTNet
from torch.optim import Adam
from torch.nn import BCELoss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    ########################################### LOADER RELATED #########################################################
    parser.add_argument('ply_folder')
    ################################################ HP ################################################################
    parser.add_argument('--dim', default=96, type=int, help="64,96")
    parser.add_argument('--bs', default=10, type=int, help="100,30")
    parser.add_argument('--strech_box', action="store_true")
    parser.add_argument('--no_shuffle', action="store_true")
    parser.add_argument('--dense', action="store_true")
    parser.add_argument('--lr', default=pow(10, -3), type=float)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--t', default=0.9, type=float)
    parser.add_argument('--device', default="cuda", type=str)

    args = parser.parse_args()

    # Model
    model = CTNet(args.dim, id=f"ctnet{args.dim}{'_dense_' if args.dense else '_'}streched_{args.strech_box}").cuda()
    # Loader
    dataset = CTDataset(ply_folder=args.ply_folder, dim=args.dim)
    # Model variables
    loader = DataLoader(dataset=dataset,
                        batch_size=args.bs,
                        num_workers=args.num_workers,
                        shuffle=not args.no_shuffle,
                        persistent_workers=True,
                        pin_memory=True,
                        drop_last=True)

    # self.train_loader.dataset.shuffle() if self.shuffle else None
    optimizer=Adam(model.parameters(), lr=pow(10, -4))
    model.train()
    criterion = BCELoss()
    for epoch in range(10):
        for i, (x, y, _, _, _) in tqdm(enumerate(loader), total=len(loader), desc=f">> Epoch {epoch}"):
            x = x.cuda()
            y = y.cuda()

            optimizer.zero_grad()
            x, y = x.cuda(), y.cuda()
            _y = model(x)
            loss = criterion(_y, y)
            loss = loss / _y.size(1)  # average the loss by minibatch
            loss.backward()
```
### How to
```bash
# Clone this repository and install the code
$ https://github.com/JeanMaximilienCadic/ctdataset

# Go into the repository
$ cd ctdataset

# Install with python (not recommended)
$ python setup.py install
```
### Docker
```
docker run --gpus all --rm  -v $(pwd):$(pwd) -p 8888:8888 -it tensorflow/tensorflow:latest-gpu-jupyter sh
```

Run a test:
```python
python -m ctdataset
```
