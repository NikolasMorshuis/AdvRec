import torch
import h5py
import numpy as np
import fastmri
from fastmri.data.mri_data import AnnotatedSliceDataset
from data.transforms import AdvDataTransform
import os
from argparse import ArgumentParser
import pandas as pd

import pytorch_lightning as pl
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.pl_modules import FastMriDataModule, UnetModule, VarNetModule
from fastmri_examples.unet.run_pretrained_unet_inference import download_model

from utils.utils import Transform_kspace_to_modelinput, error_per_angle

from augmentor.adv_rotation import AdvRotation
from augmentor.kspace_noise import KSpaceNoise
from augmentor.perturbation_finder import PerturbationFinder
from torchvision.transforms.functional import center_crop
from data.transforms import unnormalize

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils.utils import orig_kspace_to_images, get_elements_with_pathology

UNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/unet/"
VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
MODEL_FNAMES = {
    "unet_knee_sc": "knee_sc_leaderboard_state_dict.pt",
    "unet_knee_mc": "knee_mc_leaderboard_state_dict.pt",
    "unet_brain_mc": "brain_leaderboard_state_dict.pt",
    "varnet_knee_mc": "knee_leaderboard_state_dict.pt",
    "varnet_brain_mc": "brain_leaderboard_state_dict.pt",
}


def main():
    pl.seed_everything(42)

    # this creates a k-space mask that can be used to simulate undersampling
    mask_func = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    used_model_array = args.used_model_array  # possible choices: ['unet', 'varnet']

    if args.relevant_only == 'True':
        relevant_only = True  # True should be the standard in order to see effects on pathologies
    elif args.relevant_only == 'False':
        relevant_only = False

    used_transform = args.used_transform

    loss_name = args.loss_name

    state_dict_file = None

    for c_model, used_model in enumerate(used_model_array):
        get_model_input_from_kspace = Transform_kspace_to_modelinput(used_model)
        if not os.path.isdir('./model_dir'):
            os.makedirs('./model_dir')
        if used_model == 'unet':
            model = UnetModule(
                in_chans=1,
                out_chans=1,
                chans=256,  # In our model this should be 128, half the number of layers of the original
                num_pool_layers=4,
                drop_prob=0.0,
                lr=0.001,
                lr_step_size=40,
                lr_gamma=0.1,
                weight_decay=0.0,
            )
            if state_dict_file is None:
                model_path = './model_dir/{}'.format(MODEL_FNAMES['unet_knee_mc'])
                if not os.path.isfile(model_path):
                    url_root = UNET_FOLDER
                    download_model(url_root + MODEL_FNAMES['unet_knee_mc'], model_path)
                state_dict_file = model_path

        elif used_model == 'varnet':
            model = VarNetModule(
                num_cascades=12,
                pools=4,
                chans=18,
                sens_pools=4,
                sens_chans=8,
                lr=0.003,
                lr_step_size=40,
                lr_gamma=0.1,
                weight_decay=0.0,
            )
            if state_dict_file is None:
                model_path = './model_dir/{}'.format(MODEL_FNAMES['varnet_knee_mc'])
                if not os.path.isfile(model_path):
                    url_root = VARNET_FOLDER
                    download_model(url_root + MODEL_FNAMES['varnet_knee_mc'], model_path)
                state_dict_file = model_path

        checkpoint = torch.load(state_dict_file)

        if used_model == 'unet':
            model.unet.load_state_dict(checkpoint)
            network = model.unet.cuda()
        elif used_model == 'varnet':
            model.varnet.load_state_dict(checkpoint)
            network = model.varnet.cuda()

        train_transform = AdvDataTransform(mask_func=mask_func, use_seed=False)

        annotated_slice_dataset = AnnotatedSliceDataset(
            root=os.path.join(args.data_path, 'multicoil_val'),
            challenge="multicoil",
            subsplit="knee",
            transform=train_transform,
            annotation_version="640500fb",
            multiple_annotation_policy='all'
        )

        df = pd.read_csv('./.annotation_cache/knee640500fb.csv')
        patholo_indices, patients_with_patholo = get_elements_with_pathology(df, annotated_slice_dataset.examples,
                                                                             False)

        if used_transform == 'rotation':
            rot_ratio = 0.1  # 0.1 should be equivalent to 18 degrees

            augmentor_affine = AdvRotation(
                rot_ratio=rot_ratio
            )
            transformation_chain = [augmentor_affine]

        elif used_transform == 'noise':
            config_dict = {
                'epsilon': 5e-5,  # 1e-4 can also be k if we have a for-loop
                'masked_data_only': False,
                'attack_mode': 2,
            }
            augmentor_noise = KSpaceNoise(config_dict=config_dict)
            transformation_chain = [augmentor_noise]

        solver = PerturbationFinder(
            transform_list=transformation_chain,
            input_from_kspace=get_model_input_from_kspace,
            loss=loss_name
        )

        if used_transform == 'rotation':
            all_angles_losses = []
            relative_errors = [5 / 180]  # 5 degree
        else:
            # relative to the l2-norm of the coil, e.g. for 0.01 the ||z_i||_2 = 0.01 ||k_i||_2
            relative_errors = np.array([0, 0.002, 0.005, 0.01, 0.015, 0.02, 0.025])

        overall_ssim_score = []
        overall_psnr_score = []
        row_name = []
        index_array = patholo_indices

        for i in range(len(index_array)):
            data = annotated_slice_dataset.__getitem__(index_array[i])
            num_low_frequencies = data.num_low_frequencies
            # load the sensitivity map:
            acq = data.acq  # With this parameter we know where the kspace was sampled and where not!
            fname = data.fname
            patient_name = fname.split('.')[0]
            slice = data.slice_num

            # Get complex coil-images from k-space data:
            ims = orig_kspace_to_images(data.masked_kspace)

            complex_target = fastmri.rss(ims, 0)
            magnitude_target = torch.abs(complex_target)
            target_image = center_crop(magnitude_target, [320, 320])

            labels_for_slice = df[(df['file'] == data.fname.split('.')[0]) & (df['slice'] == data.slice_num)]
            kspace = data.masked_kspace.unsqueeze(0).cuda()

            for label in labels_for_slice.values:
                _, _, _, x0, y0, w, h, label_txt = label
                if y0 < 0:
                    y0 = 0
                x0, y0, w, h = int(x0), int(y0), int(w), int(h)
                position = [x0, y0, w, h]

                # in usual coordinates x0, y0 is the upper left corner
                x1 = x0 + w
                y1 = y0 + h

                # The mask for the sampled k-space data
                mask = data.mask.cuda()

                # The mask in image-space, indicating the location of the pathology
                mask2 = torch.zeros((1, 1, 320, 320)).bool().cuda()
                mask2[:, :, y0:y1, x0:x1] = 1

                # unique row name for dataframe the pathology for the dataframe
                row_name.append('{}_{}_{}_{}'.format(patient_name, slice, label_txt[:10], x0))

                ssim_score_array = []
                psnr_score_array = []
                for c, k in enumerate(relative_errors):
                    if used_transform == 'rotation':
                        rot_ratio = k
                        assert rot_ratio > 0, 'rot_ratio has to be larger than 0'
                        # we could also include a relevant only parameter in rotation
                        transformed_kspace, target, transformed_target, final_angles, loss_array, psnr_array, mse_array, best_angle = solver.transform_kspace_rotation(
                            kspace,
                            network, mask, eckpoints=[x0, x1, y0, y1], rot_ratio=rot_ratio,
                            used_model=used_model, acq=acq, num_low_frequencies=num_low_frequencies,
                            relevant_only=relevant_only)
                        all_angles_losses.append(loss_array)
                    elif used_transform == 'noise':
                        relative_error = k
                        transformed_kspace = \
                            solver.transform_kspace_noise(kspace, network, mask, mask2,
                                                          num_low_frequencies=num_low_frequencies,
                                                          used_model=used_model, acq=acq, relative_error=relative_error,
                                                          relevant_only=relevant_only, position=position, n_iter=10)

                    model_input, mean, std = get_model_input_from_kspace.make_model_input(transformed_kspace)
                    if used_model == 'unet':
                        output = network(model_input)
                        output_cropped = center_crop(output, [320, 320])
                        output_orig = unnormalize(output_cropped, mean, std)[0, 0].cpu().detach().numpy()

                    elif used_model == 'varnet':
                        output = network(model_input, torch.unsqueeze(mask, 0), num_low_frequencies)
                        output_cropped = center_crop(output, [320, 320])
                        output_orig = output_cropped[0].cpu().detach().numpy()

                    target_ssim = target_image.cpu().detach().numpy()
                    target_ssim_relreg = target_ssim[y0:y1, x0:x1]

                    output_ssim = output_orig
                    output_ssim_relreg = output_ssim[y0:y1, x0:x1]
                    ssim_score = structural_similarity(target_ssim_relreg, output_ssim_relreg,
                                                       data_range=target_ssim.max() - target_ssim.min())
                    psnr_score = peak_signal_noise_ratio(target_ssim_relreg, output_ssim_relreg,
                                                         data_range=target_ssim.max() - target_ssim.min())
                    if used_transform == 'rotation':
                        target_angles = [0, 1, 2, 3, 4, 5]
                        ssim_score_array, psnr_score_array, mse_score_array = error_per_angle(
                            final_angles, loss_array, psnr_array, mse_array, target_angles
                        )
                    elif used_transform == 'noise':
                        ssim_score_array.append(ssim_score)
                        psnr_score_array.append(psnr_score)
                    else:
                        assert False, '{} not implemented'.format(used_transform)
                    print('ssim-score array: ', ssim_score_array)
                    print('psnr-score array: ', psnr_score_array)
                overall_ssim_score.append(ssim_score_array)
                overall_psnr_score.append(psnr_score_array)
                print('image complete')

        # From here this is only to write the results in a table and store it in a way so we know which result is which
        if used_transform == 'rotation':
            column_names = target_angles
        else:
            column_names = relative_errors
        df_ssim = pd.DataFrame(overall_ssim_score, columns=column_names, index=row_name)
        df_psnr = pd.DataFrame(overall_psnr_score, columns=column_names, index=row_name)
        if used_transform == 'rotation':
            df_all_angles = pd.DataFrame(all_angles_losses, columns=final_angles, index=row_name)
            addition = 'rotation'
        elif used_transform == 'noise':
            addition = 'kspacenoise'
        if not os.path.isdir('./results'):
            os.makedirs('./results')
        if relevant_only:
            df_ssim.to_csv(
                './results/ssim_scores_{}_{}_{}x_{}.csv'.format(used_model, addition, args.accelerations[0], loss_name))
            df_psnr.to_csv(
                './results/psnr_scores_{}_{}_{}x_{}.csv'.format(used_model, addition, args.accelerations[0], loss_name))
        else:
            df_ssim.to_csv(
                './results/ssim_scores_{}_complimage_{}_{}x_{}.csv'.format(used_model, addition, args.accelerations[0],
                                                                           loss_name))
            df_psnr.to_csv(
                './results/psnr_scores_{}_complimage_{}_{}x_{}.csv'.format(used_model, addition, args.accelerations[0],
                                                                           loss_name))
    if used_transform == 'rotation':
        df_all_angles.to_csv(
            './results/all_angles_loss_{}_{}_{}x_{}.csv'.format(used_model, addition, args.accelerations[0], loss_name))
    print('end of function')


def build_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        help="Config File to use for experiment"
    )

    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced_fraction"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.04],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[8],
        type=int,
        help="Acceleration rates to use for masks",
    )

    parser.add_argument(
        "--relevant_only",
        default='True',
        type=str,
        help='Attack only annotated regions (relevant_only=True) or complete image (relevant_only=False)'
    )

    parser.add_argument(
        "--used_model_array",
        nargs='+',
        default=['varnet']  # ['unet', 'varnet']
    )

    parser.add_argument(
        "--used_transform",
        default='noise',
        choices=('noise', 'rotation'),
    )

    parser.add_argument(
        "--loss_name",
        default='MSE',
        choices=('MSE', 'SSIM', 'MSE+SSIM')
    )

    parser.add_argument(
        "--data_path",
        default=None,
        help="The path to the raw k-space data"
    )

    parser.add_argument(
        "--use_dataset_cache_file",
        default=False,
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = build_args()
    main()
