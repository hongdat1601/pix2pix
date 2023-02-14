import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

class BaseDataset(Dataset):
  def __init__(self, 
               img_names, 
               img_dir, 
               mask_dir, 
               ext_img="jpg", 
               ext_mask="jpg", 
               transform=None, 
               target_transform=None):
    self.img_dir = img_dir
    self.mask_dir = mask_dir
    self.transform = transform
    self.target_transform = target_transform
    self.img_names = img_names
    self.ext_mask = ext_mask
    self.ext_img = ext_img

  def __len__(self):
    return len(self.img_names)

  def __getitem__(self, idx):
    img = read_image(os.path.join(self.img_dir, self.img_names[idx] + f'.{self.ext_img}'))
    mask = read_image(os.path.join(self.mask_dir, self.img_names[idx] + f'.{self.ext_mask}'))
    
    if self.transform:
      mask = self.transform(mask)
    
    if self.target_transform:
      img = self.target_transform(img)

    return mask, img
  
def load_data(img_dir, 
              mask_dir, 
              ext_img="jpg", 
              ext_mask="jpg", 
              transform=None, 
              target_transform=None, 
              batch_size=4, 
              test_split=0.2):
    # Load train data
    imgs = os.listdir(img_dir)
    imgs = [i.split('.')[0] for i in imgs]
    np.random.seed(42)
    np.random.shuffle(imgs)

    train_max_idx = int(len(imgs) * (1 - test_split))
    train_imgs = imgs[:train_max_idx]
    val_imgs = imgs[train_max_idx:]

    train_dataset = BaseDataset(
       img_names=train_imgs,
       img_dir=img_dir,
       mask_dir=mask_dir,
       ext_img=ext_img,
       ext_mask=ext_mask,
       transform=transform,
       target_transform=target_transform
    )

    # Load val data

    val_dataset = BaseDataset(
       img_names=val_imgs,
       img_dir=img_dir,
       mask_dir=mask_dir,
       ext_img=ext_img,
       ext_mask=ext_mask,
       transform=transform,
       target_transform=target_transform
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return (train_dataloader, val_dataloader)
