import os
import pickle
from PIL import Image
import h5py
import json

import numpy as np
from tqdm import tqdm
import requests
import zipfile
import glob
import shutil
import collections
from scipy.io import loadmat
from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchvision.datasets.utils import download_file_from_google_drive


class TrafficSign(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = TrafficSignClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, class_augmentations=class_augmentations,
            download=download)
        super(TrafficSign, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)

class TrafficSignClassDataset(ClassDataset):
    folder = 'traffic_sign'

    zip_url = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip'
    
    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(TrafficSignClassDataset, self).__init__(meta_train=meta_train,
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
            raise RuntimeError('TrafficSign integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        data = self.data[class_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return TrafficSignDataset(index, data, class_name,
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
        import zipfile

        if self._check_integrity():
            return

        chunkSize = 1024
        r = requests.get(self.zip_url, stream=True)
        with open(self.root+'/GTSRB_Final_Training_Images.zip', 'wb') as f:
            pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
            for chunk in r.iter_content(chunk_size=chunkSize):
                if chunk: # filter out keep-alive new chunks
                    pbar.update (len(chunk))
                    f.write(chunk)

        filename = os.path.join(self.root, 'GTSRB_Final_Training_Images.zip')
        traffic_sign_zip = zipfile.ZipFile(filename)
        traffic_sign_zip.extractall(self.root)
        traffic_sign_zip.close()
                
        for split in ['test']:
            filename = os.path.join(self.root, self.filename.format(split))
            labels_filename = os.path.join(self.root, self.filename_labels.format(split))

            cls_path_lst = glob.glob(os.path.join(self.root, 'GTSRB', 'Final_Training', 'Images', '*'))
            images = np.array([])
            classes = {}
            pre_idx = 0
            post_idx = 0

            for cls_path in tqdm(cls_path_lst):
                pre_idx = post_idx

                cls_data = []
                file_path_lst = glob.glob(os.path.join(cls_path, '*'))
                for file_path in file_path_lst:
                    if not 'csv' in file_path:
                        img = Image.open(file_path).convert('RGB')
                        img = np.asarray(img.resize((32, 32)))
                        cls_data.append(img)
                cls_data = np.array(cls_data)
                if images.shape[0] == 0:
                    images = cls_data
                else:
                    images = np.concatenate((images, cls_data), axis=0)

                post_idx = pre_idx + len(cls_data)
                cls_id = int(cls_path.split('/')[-1])
                classes[str(cls_id)] = list(range(pre_idx, post_idx))

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                for name, indices in classes.items():
                    group.create_dataset(name, data=images[indices])

            with open(labels_filename, 'w') as f:
                labels = sorted(list(classes.keys()))
                json.dump(labels, f)
                
class TrafficSignDataset(Dataset):
    def __init__(self, index, data, class_name,
                 transform=None, target_transform=None):
        super(TrafficSignDataset, self).__init__(index, transform=transform,
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
