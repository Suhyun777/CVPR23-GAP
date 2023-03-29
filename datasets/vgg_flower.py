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


class VggFlower(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = VggFlowerClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, class_augmentations=class_augmentations,
            download=download)
        super(VggFlower, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class VggFlowerClassDataset(ClassDataset):
    folder = 'vgg_flower'

    tgz_url = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
    mat_url = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
    
    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(VggFlowerClassDataset, self).__init__(meta_train=meta_train,
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
            raise RuntimeError('VggFlower integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        data = self.data[class_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return VggFlowerDataset(index, data, class_name,
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
        r = requests.get(self.tgz_url, stream=True)
        with open(self.root+'/102flowers.tgz', 'wb') as f:
            pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
            for chunk in r.iter_content(chunk_size=chunkSize):
                if chunk: # filter out keep-alive new chunks
                    pbar.update (len(chunk))
                    f.write(chunk)
                    
        chunkSize = 1024
        r = requests.get(self.mat_url, stream=True)
        with open(self.root+'/imagelabels.mat', 'wb') as f:
            pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
            for chunk in r.iter_content(chunk_size=chunkSize):
                if chunk: # filter out keep-alive new chunks
                    pbar.update (len(chunk))
                    f.write(chunk)

        filename = os.path.join(self.root, '102flowers.tgz')
        with tarfile.open(filename, 'r') as f:
            f.extractall(self.root)

        imagelabels_path = self.root+'/imagelabels.mat'
        with open(imagelabels_path, 'rb') as f:
            labels = loadmat(f)['labels'][0]

        filepaths = collections.defaultdict(list)
        for i, label in enumerate(labels):
            filepaths[label].append(os.path.join(self.root+'/jpg', 'image_{:05d}.jpg'.format(i + 1)))
        
        splits = {}
        splits['train'] = [
            "090.canna lily", "038.great masterwort", "080.anthurium", "030.sweet william",
            "029.artichoke", "012.colt's foot", "043.sword lily", "027.prince of wales feathers",
            "004.sweet pea", "064.silverbush", "031.carnation", "099.bromelia", 
            "008.bird of paradise", "067.spring crocus", "095.bougainvillea", "077.passion flower",
            "078.lotus", "061.cautleya spicata", "088.cyclamen", "074.rose", "055.pelargonium",
            "032.garden phlox", "021.fire lily", "013.king protea", "079.toad lily", "070.tree poppy",
            "051.petunia", "069.windflower", "014.spear thistle", "060.pink-yellow dahlia?",
            "011.snapdragon", "039.siam tulip", "063.black-eyed susan", "037.cape flower",
            "036.ruby-lipped cattleya", "028.stemless gentian", "048.buttercup", "007.moon orchid",
            "093.ball moss", "002.hard-leaved pocket orchid", "018.peruvian lily", "024.red ginger",
            "006.tiger lily", "003.canterbury bells", "044.poinsettia", "076.morning glory",
            "075.thorn apple", "072.azalea", "052.wild pansy", "084.columbine", "073.water lily",
            "034.mexican aster", "054.sunflower", "066.osteospermum", "059.orange dahlia",
            "050.common dandelion", "091.hippeastrum", "068.bearded iris", "100.blanket flower",
            "071.gazania", "081.frangipani", "101.trumpet creeper", "092.bee balm", 
            "022.pincushion flower", "033.love in the mist", "087.magnolia", "001.pink primrose",
            "049.oxeye daisy", "020.giant white arum lily", "025.grape hyacinth", "058.geranium"
            ]
        splits['valid'] = [
            "010.globe thistle", "016.globe-flower", "017.purple coneflower", "023.fritillary",
            "026.corn poppy", "047.marigold", "053.primula", "056.bishop of llandaff", 
            "057.gaura", "062.japanese anemone", "082.clematis", "083.hibiscus", 
            "086.tree mallow", "097.mallow", "102.blackberry lily"
            ]
        splits['test'] = [
            "005.english marigold", "009.monkshood", "015.yellow iris", "019.balloon flower",
            "035.alpine sea holly", "040.lenten rose", "041.barbeton daisy", "042.daffodil",
            "045.bolero deep blue", "046.wallflower", "065.californian poppy", "085.desert-rose",
            "089.watercress", "094.foxglove", "096.camellia", "098.mexican petunia"
            ]
        
        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            labels_filename = os.path.join(self.root, self.filename_labels.format(split))

            images = np.array([])
            classes = {}
            pre_idx = 0
            post_idx = 0

            for cls_name in tqdm(splits[split]):
                pre_idx = post_idx

                cls_id = int(cls_name[:3])
                cls_data = []
                for file_path in filepaths[cls_id]:
                    img = Image.open(file_path)
                    img = np.asarray(img.resize((32, 32)))
                    cls_data.append(img)
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

class VggFlowerDataset(Dataset):
    def __init__(self, index, data, class_name,
                 transform=None, target_transform=None):
        super(VggFlowerDataset, self).__init__(index, transform=transform,
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
