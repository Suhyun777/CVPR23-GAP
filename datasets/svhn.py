import os
import pickle
from PIL import Image
import h5py
import json
import numpy as np
from tqdm import tqdm
import requests
import tarfile
import glob
import shutil
import collections
from scipy.io import loadmat

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchvision.datasets.utils import download_file_from_google_drive


class SVHN(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = SVHNClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, class_augmentations=class_augmentations,
            download=download)
        super(SVHN, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)

class SVHNClassDataset(ClassDataset):
    folder = 'svhn'

    train_url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
    test_url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
    
    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(SVHNClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)
        
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root,
            self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('SVHN integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        data = self.data[class_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return SVHNDataset(index, data, class_name,
            transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def download(self):
        import tarfile

        if self._check_integrity():
            return

        chunkSize = 1024
        r = requests.get(self.test_url, stream=True)
        with open(self.root+'/test_32x32.mat', 'wb') as f:
            pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
            for chunk in r.iter_content(chunk_size=chunkSize):
                if chunk: # filter out keep-alive new chunks
                    pbar.update (len(chunk))
                    f.write(chunk)

        data = loadmat(self.root+'/test_32x32.mat')
        x_lst = data['X'].transpose(3,0,1,2)
        y_lst = data['y']
        x_per_cls = [[] for _ in range(10)]
        for i in range(len(y_lst)):
            x = x_lst[i]
            y = y_lst[i][0] - 1
            x_per_cls[y].append(x)
                
        for split in ['test']:
            filename = os.path.join(self.root, self.filename.format(split))
            labels_filename = os.path.join(self.root, self.filename_labels.format(split))

            images = np.array([])
            classes = {}
            pre_idx = 0
            post_idx = 0

            for cls_id, cls_data in enumerate(tqdm(x_per_cls)):
                pre_idx = post_idx
                cls_data = np.array(cls_data)
                
                if images.shape[0] == 0:
                    images = cls_data
                else:
                    images = np.concatenate((images, cls_data), axis=0)

                post_idx = pre_idx + len(cls_data)
                classes[str(cls_id)] = list(range(pre_idx, post_idx))

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                for name, indices in classes.items():
                    group.create_dataset(name, data=images[indices])

            with open(labels_filename, 'w') as f:
                labels = sorted(list(classes.keys()))
                json.dump(labels, f)
                
class SVHNDataset(Dataset):
    def __init__(self, index, data, class_name,
                 transform=None, target_transform=None):
        super(SVHNDataset, self).__init__(index, transform=transform,
                                                  target_transform=target_transform)
        self.data = data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        target = self.class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)
