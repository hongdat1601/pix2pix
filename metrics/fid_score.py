import torch
from torch import nn
from torchvision.models import inception_v3
import torchvision.transforms  as T
import PIL
import pickle
import multiprocessing
import numpy as np
import glob
import os
from scipy import linalg
import warnings
from tqdm import tqdm


class PartialInceptionNetwork(nn.Module):

    def __init__(self, transform_input=True):
        super().__init__()
        self.inception_network = inception_v3(pretrained=True)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                             ", but got {}".format(x.shape)
        x = x * 2 -1 # Normalize to [-1, 1]

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1 
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)
        return activations


def get_activations(images, batch_size, device):
    """
    Calculates activations for last pool layer for all iamges
    --
        Images: torch.array shape: (N, 3, 299, 299), dtype: torch.float32
        batch size: batch size used for inception network
    --
    Returns: np array shape: (N, 2048), dtype: np.float32
    """
    assert images.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                              ", but got {}".format(images.shape)

    num_images = images.shape[0]
    inception_network = PartialInceptionNetwork()
    inception_network = inception_network.to(device)
    inception_network.eval()
    n_batches = int(np.ceil(num_images  / batch_size))
    inception_activations = np.zeros((num_images, 2048), dtype=np.float32)
    for batch_idx in tqdm(range(n_batches)):
        start_idx = batch_size * batch_idx
        end_idx = batch_size * (batch_idx + 1)

        ims = images[start_idx:end_idx]
        ims = ims.to(device)
        activations = inception_network(ims)
        activations = activations.detach().cpu().numpy()
        assert activations.shape == (ims.shape[0], 2048), "Expexted output shape to be: {}, but was: {}".format((ims.shape[0], 2048), activations.shape)
        inception_activations[start_idx:end_idx, :] = activations
    return inception_activations



def calculate_activation_statistics(images, batch_size, device):
    """Calculates the statistics used by FID
    Args:
        images: torch.tensor, shape: (N, 3, H, W), dtype: torch.float32 in range 0 - 1
        batch_size: batch size to use to calculate inception scores
    Returns:
        mu:     mean over all activations from the last pool layer of the inception model
        sigma:  covariance matrix over all activations from the last pool layer 
                of the inception model.
    """
    images = preprocess_images(images)
    act = get_activations(images, batch_size, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# Modified from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def preprocess_images(images):
    """Resizes and shifts the dynamic range of image to 0-1
    Args:
        images: np.array, shape: (N, 3, H, W), dtype: float32 between 0-1 or np.uint8
        use_multiprocessing: If multiprocessing should be used to pre-process the images
    Return:
        final_images: torch.tensor, shape: (N, 3, 299, 299), dtype: torch.float32 between 0-1
    """
    res = []
    trans = T.Resize((299, 299))

    for i in range(len(images)):
        tmp = trans(images[i])
        res.append(tmp)

    res = torch.stack(res)
    return res


def load_images(images_path):
    lst_imgs = os.listdir(images_path)
    imgs = []
    trans = T.ToTensor()

    for i in lst_imgs:
        im = PIL.Image.open(os.path.join(images_path, i))
        im = im.convert('RGB')
        im = im.resize((256, 256))
        imgs.append(trans(im))

    imgs = torch.stack(imgs)
    return imgs


def save_cahe(mu, sigma):
    tmp = {
        "mu": mu,
        "sigma": sigma
    }

    with open('./cache/fid_cache.pickle', 'wb') as f:
        pickle.dump(tmp, f)


def load_cache():
    with open('./cache/fid_cache.pickle', 'rb') as f:
        cache = pickle.load(f)
    return cache

def calculate_fid(original_imgs_path, gen_imgs, batch_size, device):
    """ Calculate FID between images1 and images2
    Args:
        original_imgs_path: str, Images folder path
        images2: torch.Tensor, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        use_multiprocessing: If multiprocessing should be used to pre-process the images
        batch size: batch size used for inception network
    Returns:
        FID (scalar)
    """
    if not os.path.exists('./cache/fid_cache.pickle'):
        original_imgs = load_images(original_imgs_path)
        print("Create FID Cache")
        mu1, sigma1 = calculate_activation_statistics(original_imgs, batch_size, device)
        save_cahe(mu1, sigma1)
    else:
        cache = load_cache()
        mu1 = cache["mu"]
        sigma1 = cache["sigma"]
    
    mu2, sigma2 = calculate_activation_statistics(gen_imgs, batch_size, device)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid