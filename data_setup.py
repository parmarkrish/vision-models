import torch
from torch.utils.data import Dataset
import numpy as np

class CIFAR10(Dataset):
    def __init__(self, root='data', train=True, transform=None):
        import requests, tarfile, os
        if not os.path.exists(os.path.join('data', 'cifar-10-batches-py')):
            print('in if')
            url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            response = requests.get(url, stream=True)
            tarfile.open(fileobj=response.raw, mode='r|gz').extractall(path=root)

        import pickle
        num_imgs = 50000 if train else 10000
        self.X, self.y = np.empty((num_imgs, 3072), dtype=np.ubyte), np.empty(num_imgs, dtype=np.ubyte)
        for i in range(5):
            data_batch = f'data_batch_{i+1}' if train else 'test_batch'
            with open(os.path.join(root, 'cifar-10-batches-py', data_batch), 'rb') as f:
                data_dict = pickle.load(f, encoding='bytes')
            self.X[i*10000:(i+1)*10000], self.y[i*10000:(i+1)*10000] = data_dict[b'data'], data_dict[b'labels']
            if not train: break
        self.X = self.X.reshape(-1, 3, 32, 32)
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.y)