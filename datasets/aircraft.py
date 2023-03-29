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
from torchmeta.datasets.utils import get_asset


class AirCraft(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = AirCraftClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, class_augmentations=class_augmentations,
            download=download)
        super(AirCraft, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)

class AirCraftClassDataset(ClassDataset):
    folder = 'aircraft'

    tar_url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    
    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(AirCraftClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)
        
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root,
            self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        self._data_file = None
        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('AirCraft integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index % self.num_classes]
        data = self.data[label]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return AirCraftDataset(index, data, label,
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
        r = requests.get(self.tar_url, stream=True)
        with open(self.root+'/fgvc-aircraft-2013b.tar.gz', 'wb') as f:
            pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
            for chunk in r.iter_content(chunk_size=chunkSize):
                if chunk: # filter out keep-alive new chunks
                    pbar.update (len(chunk))
                    f.write(chunk)

        filename = os.path.join(self.root, 'fgvc-aircraft-2013b.tar.gz')
        with tarfile.open(filename, 'r') as f:
            f.extractall(self.root)

        # Cropping images with bounding box same as meta-dataset.
        bboxes_path = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data', 'images_box.txt')
        with open(bboxes_path, 'r') as f:
            names_to_bboxes = [line.split('\n')[0].split(' ') for line in f.readlines()]
            names_to_bboxes = dict((name, map(int, (xmin, ymin, xmax, ymax))) for name, xmin, ymin, xmax, ymax in names_to_bboxes)
            
        # Retrieve mapping from filename to cls
        cls_trainval_path = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data', 'images_variant_trainval.txt')
        with open(cls_trainval_path, 'r') as f:
            filenames_to_clsnames = [line.split('\n')[0].split(' ', 1) for line in f.readlines()]
            
        cls_test_path = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data', 'images_variant_test.txt')
        with open(cls_test_path, 'r') as f:
            filenames_to_clsnames += [line.split('\n')[0].split(' ', 1) for line in f.readlines()]
            
        filenames_to_clsnames = dict(filenames_to_clsnames)
        clss_to_names = collections.defaultdict(list)
        for filename, cls in filenames_to_clsnames.items():
            clss_to_names[cls].append(filename)
                
        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            labels = get_asset(self.folder, '{}.json'.format(split))
            labels_filename = os.path.join(self.root, self.filename_labels.format(split))
            
            with open(labels_filename, 'w') as f:
                json.dump(labels, f)

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                for i, label in enumerate(tqdm(labels, desc=filename)):
                    images = []
                    for file in clss_to_names[label]:
                        file_path = os.path.join(self.root,
                                                'fgvc-aircraft-2013b',
                                                'data',
                                                'images',
                                                '{}.jpg'.format(file))
                        img = Image.open(file_path)
                        bbox = names_to_bboxes[file]
                        img = np.asarray(img.crop(bbox).resize((32, 32)), dtype=np.uint8)
                        images.append(img)
                        
                    dataset = group.create_dataset(label, (len(images), 32, 32, 3))
                    
                    for j, image in enumerate(images):
                        dataset[j] = image
                
class AirCraftDataset(Dataset):
    def __init__(self, index, data, label,
                 transform=None, target_transform=None):
        super(AirCraftDataset, self).__init__(index, transform=transform,
                                                  target_transform=target_transform)
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index].astype(np.uint8)).convert('RGB')
        target = self.label

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)
