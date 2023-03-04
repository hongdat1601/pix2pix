import argparse
import torchvision.transforms as T
from models.discriminator.patch_gan import PatchGan
from models.generator.attention_unet import AttentionUnet
from models.generator.unet import UnetGenerator
from models.pix2pix_unet import Pix2Pix
import torch.nn as nn
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from utils.data import get_data

if __name__ == "__main__":
    # Add Arguments
    parser = argparse.ArgumentParser(description="Training pix2pix model.")

    parser.add_argument("--img-dir", required=True, help="Image folder path.")
    parser.add_argument("--mask-dir", required=True, help="Mask folder path.")
    parser.add_argument("--generator", required=False, choices=['unet', 'attention-unet'], help="Select generator model.", default="unet")
    parser.add_argument("--ext-img", required=False, help="Image file extension.", default="jpg")
    parser.add_argument("--ext-mask", required=False, help="Mask file extension.", default="jpg")
    parser.add_argument("--batch-size", required=False, default=4, type=int)
    parser.add_argument("--resize", required=False, help="Image's size to resize.", default=256, type=int)
    parser.add_argument("--device", required=False, default="cpu", choices=['cpu', 'cuda'])

    args = parser.parse_args()

    if not os.path.exists('./result'):
        os.mkdir('./result')
        print("Created result folder")

    # Load data
    transform = T.Compose([
        T.Resize((args.resize, args.resize)),
        T.Lambda(lambda x : (x / 127.5) - 1.0)
    ])

    file_path = './data/test.txt'
    img_dir = args.img_dir
    mask_dir = args.mask_dir
    ext_img = args.ext_img
    ext_mask = args.ext_mask
    batch_size = args.batch_size
    transform = transform
    target_transform = transform

    test_dataloader = get_data(file_path, img_dir, mask_dir, ext_img, ext_mask, transform, target_transform, batch_size)

    # Config model
    generator = None
    if args.generator == 'unet':
        generator = UnetGenerator(3, 3, 64, use_dropout=False)
    else:
        generator = AttentionUnet(3, 3)

    model = Pix2Pix(generator=generator,
                    discriminator=PatchGan(6),
                    gen_loss=nn.L1Loss(),
                    dis_loss=nn.BCELoss())
    
    # Test
    device = torch.device(args.device)
    for i, (input, target) in enumerate(tqdm(test_dataloader)):
        pred = model.generate(input, device)

        fig, ax = plt.subplots(3, 1, figsize=(10, 10))

        grid_input = make_grid(input, pad_value=2, normalize=True)
        ax[0].imshow(grid_input.permute(1, 2, 0))
        ax[0].set_title('Input')

        grid_target = make_grid(target, pad_value=2, normalize=True)
        ax[1].imshow(grid_target.permute(1, 2, 0))
        ax[1].set_title('Target')

        grid_pred = make_grid(pred, pad_value=2, normalize=True)
        ax[2].imshow(grid_pred.permute(1, 2, 0))
        ax[2].set_title('Generated Image')

        fig.savefig(f'./result/{i}.png')
