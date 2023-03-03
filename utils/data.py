import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode

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
    img = read_image(os.path.join(self.img_dir, self.img_names[idx] + f'.{self.ext_img}'), 
                     ImageReadMode.RGB)
    mask = read_image(os.path.join(self.mask_dir, self.img_names[idx] + f'.{self.ext_mask}'), 
                     ImageReadMode.RGB)
    
    if self.transform:
      mask = self.transform(mask)
    
    if self.target_transform:
      img = self.target_transform(img)

    return mask, img
  
def get_data(file_path, img_dir, mask_dir, ext_img, ext_mask, transform, target_transform, batch_size):
    with open(file_path, 'r') as f:
        imgs = f.read()
    imgs = imgs.split('\n')

    try:
        imgs.remove('')
    except:
        pass

    dataset = BaseDataset(
        img_names=imgs,
        img_dir=img_dir,
        mask_dir=mask_dir,
        ext_img=ext_img,
        ext_mask=ext_mask,
        transform=transform,
        target_transform=target_transform
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
  
def load_data(img_dir, 
              mask_dir, 
              ext_img="jpg", 
              ext_mask="jpg", 
              transform=None, 
              target_transform=None, 
              batch_size=4):
    # Load train data
    train_dataloader = get_data('./train.txt', 
                             img_dir, 
                             mask_dir, 
                             ext_img, 
                             ext_mask, 
                             transform, 
                             target_transform,
                             batch_size)

    # Load val data
    test_dataloader = get_data('./test.txt',
                                img_dir,
                                mask_dir,
                                ext_img,
                                ext_mask,
                                transform,
                                target_transform,
                                batch_size)

    return (train_dataloader, test_dataloader)
