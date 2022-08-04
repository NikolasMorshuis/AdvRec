"""
Utilities for example srcripts.
"""
import pathlib
import yaml
import unittest
import pandas as pd
import numpy as np
from pathlib import Path, PurePath
from PIL import ImageDraw, Image, ImageEnhance
import torch
import unittest
import fastmri
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

def orig_kspace_to_images(kspace_torch, dim=(-2, -1)):
    kspace_shifted = torch.fft.ifftshift(kspace_torch, dim=dim)
    images = torch.fft.ifftn(kspace_shifted, dim=dim, norm='ortho')
    images = torch.fft.fftshift(images, dim=dim)
    return images


def image_to_orig_kspace(image):
    image = torch.fft.ifftshift(image, dim=(-2, -1))
    kspace = torch.fft.fftn(image, dim=(-2, -1), norm='ortho')
    kspace = torch.fft.fftshift(kspace, dim=(-2, -1))
    return kspace


def complex_channel_first(x):
    # From MR Augment
    # We assume a complex input image x
    x = torch.view_as_real(x)
    assert x.shape[-1] == 2
    if len(x.shape) == 3:
        # Single-coil (H, W, 2) -> (2, H, W)
        x = x.permute(2, 0, 1)
    else:
        # Multi-coil (C, H, W, 2) -> (2, C, H, W)
        assert len(x.shape) == 4
        x = x.permute(3, 0, 1, 2)
    return x


def complex_to_channels(image):
    image_real = torch.real(image)
    image_compl = torch.imag(image)
    new_image = torch.concat([image_real, image_compl], dim=1)
    return(new_image)


def channels_to_complex(image):
    image_size = image.size()
    new_image = torch.complex(image[:, 0:image_size[1]//2, :, :], image[:, image_size[1]//2:, :, :])
    return new_image


def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, Union[torch.Tensor], Union[torch.Tensor]]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


def load_args_from_config(args):
    config_file = args.config_file
    config_file = pathlib.Path(config_file)
    if config_file.exists():
        with config_file.open('r') as f:
            d = yaml.safe_load(f)
            for k,v in d.items():
                setattr(args, k, v)
    else:
        print('Config file does not exist.')
    return args


def get_elements_with_pathology(df, examples, one_per_patient=False):
    """
    This function returns the inidices where we have pathologies.
    If one_per_patient=True, it will just return one index per patient, where the index selects a slice in the middle of
    the pathologies
    :param df: Annotated dataset df
    :param examples: self.examples from the fastmri dataloaders, containing posixPath, slice_num and metadata of the slice
    :return: array of indices pointing to elements of examples that contain pathologies
    """
    # problem: what if examples is unstructured?
    indices_array = []
    patients_with_patholo = []
    fname_now = None
    indices_tmp = []
    append_indices=False
    for i in range(len(examples)):
        if i == len(examples)-1:
            append_indices=True
        file_path, slice, metadata = examples[i]
        fname = PurePath(file_path).parts[-1].split('.')[0]

        df_tmp = df[df.iloc[:,0]==fname]

        if fname in df_tmp.iloc[:,0].values:
            if slice in df_tmp.iloc[:,1].values:
                if fname_now is None:
                    fname_now = fname
                elif fname_now == fname:
                    pass
                elif fname_now != fname:
                    # Not good: last element will not be included
                    fname_now = fname
                    append_indices = True
                indices_tmp.append(i)
                patients_with_patholo.append(metadata['annotation']['fname'])
        if append_indices:
            if len(indices_tmp) >= 1:
                if one_per_patient:
                    middle_index = len(indices_tmp)//2
                    indices_array += indices_tmp[middle_index:middle_index+1]
                else:
                    indices_array += indices_tmp
                indices_tmp = []
            append_indices=False
    return indices_array, np.unique(patients_with_patholo)


def get_index_in_score_table(score, network_name, example_list, best_or_worst='worst'):
    df_score = pd.read_csv('./data_frames/{}_scores_{}.csv'.format(score, network_name), index_col=0)
    percent_of_orig = df_score.iloc[:, -2:].mean(1) / df_score.iloc[:, 0]
    best_scores_with_noise = percent_of_orig>percent_of_orig.quantile(0.95)
    worst_scores_with_noise = percent_of_orig < percent_of_orig.quantile(0.05)
    worst_cases = percent_of_orig[worst_scores_with_noise]
    best_cases = percent_of_orig[best_scores_with_noise]
    if best_or_worst == 'worst':
        cases_index = percent_of_orig.index[worst_scores_with_noise]
    elif best_or_worst == 'best':
        cases_index = percent_of_orig.index[best_scores_with_noise]
    examples_array = np.array(example_list)
    data_path = '/mnt/qb/work/baumgartner/jmorshuis45/data/fastmri_small/raw/knee_multicoil/multicoil_val/{}.h5'
    simple_np_array = np.arange(len(examples_array))
    index_array = []
    x0_array = []
    for element in cases_index:
        fname, slice_nr, dis, x0 = element.split('_')
        index = simple_np_array[np.logical_and(examples_array[:, 0] == Path(data_path.format(fname)),
                                               examples_array[:, 1] == int(slice_nr))]
        index_array.append(index[0])
        x0_array.append(int(x0))

    return index_array, x0_array


class Transform_kspace_to_modelinput():
    def __init__(self, model_type, use_sensmaps=False):
        self.model_type = model_type
        self.use_sensmaps = use_sensmaps  # Not yet implemented in public repository

    def make_model_input(self, kspace):
        if self.model_type == 'unet':
            images = orig_kspace_to_images(kspace)
            input_image = fastmri.rss(torch.abs(images), 1)
            input_image = input_image.unsqueeze(0)
            normalized_image, mean, std = normalize_instance(input_image, 1e-11)
            normalized_image = torch.clamp(normalized_image, -6, 6)
            return normalized_image, mean, std
        elif self.model_type == 'varnet':
            kspace = torch.view_as_real(kspace)
            return kspace, None, None



def error_per_angle(angles, ssim_loss, psnr_loss, mse_loss, target_angles):
    """
    Here we want to get a list of angles (e.g. [0,1,2,3,4,5] and the corresponding loss
    :param angles:
    :param loss:
    :return:
    """
    epsilon = 1e-8
    worst_score_array_ssim = []
    psnr_worst_score_array = []
    mse_worst_score_array = []
    for an in target_angles:
        smaller_angles = np.where(np.abs(angles)<an+epsilon, True, False)
        zero_out_outside_mask = np.where(np.abs(angles)<an+epsilon, mse_loss, 0)
        worst_score_mse = np.max(zero_out_outside_mask)#np.max(np.array(mse_loss)[smaller_angles])
        index_worst_score = np.argmax(zero_out_outside_mask)
        #worst_score_ssim = np.min(np.array(ssim_loss)[smaller_angles])
        #worst_score_psnr = np.min(np.array(psnr_loss)[smaller_angles])
        worst_score_ssim = ssim_loss[index_worst_score]
        worst_score_psnr = psnr_loss[index_worst_score]
        worst_score_array_ssim.append(worst_score_ssim)
        psnr_worst_score_array.append(worst_score_psnr)
        mse_worst_score_array.append(worst_score_mse)

    return worst_score_array_ssim, psnr_worst_score_array, mse_worst_score_array


def orig_kspace_to_images(kspace_torch):
    kspace_shifted = torch.fft.ifftshift(kspace_torch, dim=(-2, -1))
    images = torch.fft.ifftn(kspace_shifted, dim=(-2, -1), norm='ortho')
    images = torch.fft.fftshift(images, dim=(-2, -1))
    return images


class TestAffineMatrix(unittest.TestCase):
    def test_get_elements_with_pathology(self):
        # here we have some trouble with .h5 and not .h5 in the fname, it changes
        df = pd.read_csv('./.annotation_cache/knee640500fb.csv')
        examples = np.load('./.annotation_cache/dataset_val_examples.npy', allow_pickle=True)
        indices = get_elements_with_pathology(df, examples, True)
        fname_array = []
        for i in indices:
            file_path, slice, metadata = examples[i]
            fname = PurePath(file_path).parts[-1].split('.')[0]
            fname_array.append(fname)
        _, counts = np.unique(fname_array, return_counts=True)
        assert(np.max(counts)==1)
        assert(np.min(counts)==1)
        print('ok')