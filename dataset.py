# tch-light datamodule
import json
import torchvision.datasets as dset
import torch
import torch.utils.data as dsutil
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2

from pytorch_lightning import LightningDataModule
from pathlib import Path
from PIL import Image

class ImageFolder(Dataset):
    def __init__(self, root, transform, lab_dict, subset_ratio=1.0):
        self.inv_lab_dict = {} # 'inverse' mapping folder name of imagenet into class label id
        # v : (fd_name, lab_str)
        for k, v in lab_dict.items():
            self.inv_lab_dict[v[0]] = (k, v[1])

        rt = Path(root)
        self.trfs = transform
        self.data_lst = []
        for fold_p in rt.glob("*"):
            paths = list( fold_p.glob("*.JPEG") )
            # support subset for each class
            n_ims = int(subset_ratio * len(paths))
            paths = paths[:n_ims]
            fd_names = [fold_p.stem] * len(paths)
            self.data_lst.extend( list(zip(paths, fd_names)) )

    def __getitem__(self, index):
        path, fd_name = self.data_lst[index]
        img = Image.open( str(path) ).convert('RGB')
        if self.trfs is not None:
            img = self.trfs(img)

        lab_info = self.inv_lab_dict[fd_name]
        lab = lab_info[0]
        lab_str = lab_info[1]
        return img, int(lab), lab_str

    def __len__(self):
        return len(self.data_lst)


class ImageNet_ds(LightningDataModule):

    def __init__(self, tra_ds_rt='./tiny-imagenet-200/train', tst_ds_rt='./tiny-imagenet-200/test', img_sz=(224, 224)):
        super().__init__()
        with open('/content/imagenet_class_index.json', 'r') as f_ptr:
          self.lab_dict = json.load(f_ptr)
        self.tra_ds_rt = tra_ds_rt
        self.tst_ds_rt = tst_ds_rt
        self._train_trfs = v2.Compose([v2.Resize(img_sz), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self._test_trfs = v2.Compose([v2.Resize(img_sz), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


    @property
    def train_transform(self):
        return self._train_trfs

    @train_transform.setter
    def train_transform(self, new_trfs):
        self._train_trfs = new_trfs

    @property
    def test_transform(self):
        return self._test_trfs

    @test_transform.setter
    def test_transform(self, new_trfs):
        self._test_trfs = new_trfs

    def get_cls_by_id(self, cls_idx):
      return self.lab_dict[cls_idx]

    def setup(self, stage = 'train', subset_ratio=1.0):
      if stage == 'train':
        self.tra_img_dset = ImageFolder(root=self.tra_ds_rt, transform=self._train_trfs, lab_dict=self.lab_dict, subset_ratio=subset_ratio)
      else:
        self.tst_img_dset = ImageFolder(root=self.tst_ds_rt, transform=self._test_trfs, lab_dict=self.lab_dict, subset_ratio=subset_ratio)

    # overwrite base class methods
    def train_dataloader(self, batch_size=32, shuffle=True, num_workers=2, pin_memory=True):
        return dsutil.DataLoader(self.tra_img_dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

    def test_dataloader(self, batch_size=32, shuffle=True, num_workers=2, pin_memory=True):
        return dsutil.DataLoader(self.tst_img_dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)