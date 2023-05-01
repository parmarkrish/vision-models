'''
Contains MNIST and CIFAR-10 implementations.
PyTorch already implements this, but I've reimplemented it here for learning purposes.
'''
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader 
import torchvision.transforms as T

import numpy as np
from PIL import Image
import os
import utils

class CIFAR10(Dataset):
    def __init__(self, root='data', train=True, transform=None):
        self.transform = transform
        utils.download_and_extract('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', root=root)

        meta = os.path.join(root, 'cifar-10-batches-py', 'batches.meta')
        self.classes = utils.unpickle(meta, encoding='latin1')['label_names']

        num_imgs = 50000 if train else 10000
        self.X, self.y = np.empty((num_imgs, 3072), dtype=np.ubyte), np.empty(num_imgs, dtype=np.ubyte)
        for i in range(5):
            data_batch = f'data_batch_{i+1}' if train else 'test_batch'
            batch_path = os.path.join(root, 'cifar-10-batches-py', data_batch)
            data_dict = utils.unpickle(batch_path)
            self.X[i*10000:(i+1)*10000], self.y[i*10000:(i+1)*10000] = data_dict[b'data'], data_dict[b'labels']
            if not train: break
        self.X = np.moveaxis(self.X.reshape(-1, 3, 32, 32), 1, -1)
        
    def __getitem__(self, index):
        img = Image.fromarray(self.X[index])
        return self.transform(img) if self.transform else img, self.y[index]
    
    def __len__(self):
        return len(self.y)

class MNIST(Dataset):
    def __init__(self, root='data', train=True, transform=None):
        self.transform = transform

        utils.download_and_extract('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', root=root)
        utils.download_and_extract('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', root=root)
        utils.download_and_extract('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', root=root)
        utils.download_and_extract('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', root=root)

        img_filename = f"{'train' if train else 't10k'}-images-idx3-ubyte"
        label_filename = f"{'train' if train else 't10k'}-labels-idx1-ubyte"

        with open(os.path.join(root, img_filename), 'rb') as f1, open(os.path.join(root, label_filename), 'rb') as f2:
            self.X = np.frombuffer(f1.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
            self.y = np.frombuffer(f2.read(), dtype=np.uint8, offset=8)

    def __getitem__(self, index):
        img = Image.fromarray(self.X[index])
        return self.transform(img) if self.transform else img, self.y[index]

    def __len__(self):
        return len(self.y)

def create_dataloaders(dataset_string, batch_size):
    DatasetClass = globals()[dataset_string]

    transform = T.ToTensor()

    train_dataset = DatasetClass(transform=transform)
    test_dataset = DatasetClass(train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return {'train': train_loader, 'test': test_loader}