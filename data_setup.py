import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

def unpickle(file, encoding='bytes'):
    import pickle
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding=encoding)
    return data_dict

class CIFAR10(Dataset):
    def __init__(self, root='data', train=True, transform=None):
        import requests, tarfile, os
        self.transform = transform
        if not os.path.exists(os.path.join(root, 'cifar-10-batches-py')):
            print('Downloading dataset...')
            url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            response = requests.get(url, stream=True)
            tarfile.open(fileobj=response.raw, mode='r|gz').extractall(path=root)

        meta = os.path.join(root, 'cifar-10-batches-py', 'batches.meta')
        self.classes = unpickle(meta, encoding='latin1')['label_names']

        num_imgs = 50000 if train else 10000
        self.X, self.y = np.empty((num_imgs, 3072), dtype=np.ubyte), np.empty(num_imgs, dtype=np.ubyte)
        for i in range(5):
            data_batch = f'data_batch_{i+1}' if train else 'test_batch'
            batch_path = os.path.join(root, 'cifar-10-batches-py', data_batch)
            data_dict = unpickle(batch_path)
            self.X[i*10000:(i+1)*10000], self.y[i*10000:(i+1)*10000] = data_dict[b'data'], data_dict[b'labels']
            if not train: break
        self.X = np.moveaxis(self.X.reshape(-1, 3, 32, 32), 1, -1)
        
    def __getitem__(self, index):
        img = Image.fromarray(self.X[index])
        return self.transform(img) if self.transform else img, self.y[index]
    
    def __len__(self):
        return len(self.y)