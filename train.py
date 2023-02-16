import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from utils.data import load_data
from models.pix2pix_unet import Pix2Pix
from models.generator.unet import UnetGenerator
from models.generator.attention_unet import AttentionUnet
from models.discriminator.patch_gan import PatchGan

if __name__ == "__main__":
    # Add Arguments
    parser = argparse.ArgumentParser(description="Training pix2pix model.")

    parser.add_argument("--img-dir", required=True, help="Image folder path.")
    parser.add_argument("--mask-dir", required=True, help="Mask folder path.")
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--generator", required=False, choices=['unet', 'attention-unet'], help="Select generator model.", default="unet")
    parser.add_argument("--ext-img", required=False, help="Image file extension.", default="jpg")
    parser.add_argument("--ext-mask", required=False, help="Mask file extension.", default="jpg")
    parser.add_argument("--batch-size", required=False, default=4)
    parser.add_argument("--resize", required=False, help="Image's size to resize.", default=256, type=int)
    parser.add_argument("--test-split", required=False, default=0.2)
    parser.add_argument("--verbose", required=False, default=5)
    parser.add_argument("--draw-model", required=False, default=False, type=bool)

    args = parser.parse_args()
    print(args.img_dir)

    # Load data
    transform = T.Compose([
        T.Resize((args.resize, args.resize)),
        T.Lambda(lambda x : (x / 127.5) - 1.0)
    ])


    train_dataloader, val_dataloader = load_data(img_dir= args.img_dir,
                                                 mask_dir=args.mask_dir,
                                                 ext_img=args.ext_img,
                                                 ext_mask=args.ext_mask,
                                                 batch_size=args.batch_size,
                                                 test_split=args.test_split,
                                                 transform=transform,
                                                 target_transform=transform)

    # Create Model
    generator = UnetGenerator(3, 3, 64, use_dropout=False)
    if args.generator == "attention-unet":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator = AttentionUnet(3, 3)


    model = Pix2Pix(generator=generator,
                    discriminator=PatchGan(6),
                    gen_loss=nn.L1Loss(),
                    dis_loss=nn.BCELoss())

    # Save Model Structure
    if args.draw_model:
        model.save_structure_model(args.batch_size, args.resize)

    # Train Model
    model.train_pix2pix(train_dataloader=train_dataloader, 
                        epochs=args.epochs, 
                        verbose=args.verbose)