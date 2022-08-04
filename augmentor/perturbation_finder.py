import fastmri
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import center_crop

from utils.utils import orig_kspace_to_images, complex_to_channels, channels_to_complex, image_to_orig_kspace
from data.transforms import unnormalize
from fastmri.losses import SSIMLoss
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error


class adversarial_training(object):
    def __init__(self, transform_list, input_from_kspace, loss='MSE'):
        self.transforms = transform_list
        if loss == 'MSE' or loss == 'MSE+SSIM':
            self.loss1 = F.mse_loss
        if loss == 'SSIM' or loss == 'MSE+SSIM':
            self.loss2 = SSIMLoss().cuda()
        self.loss_name = loss
        self.input_from_kspace = input_from_kspace

    def transform_kspace_rotation(self, kspace, model, mask, eckpoints, rot_ratio=1, used_model='unet',
                                            acq=None, num_low_frequencies=None, relevant_only=True, step_size=0.1):
        """
        grid search through several rotation-angles, calculates the resp scores and returns the scores together with
        the worst-case transformed k-space.
        """
        model.eval()
        print('start adversarial attack:')
        loss_array = []
        psnr_array = []
        mse_array = []
        angle_array = []
        x0_int, x1_int, y0_int, y1_int = eckpoints

        mask_unsqueezed = torch.unsqueeze(mask, 0)

        # for if we want to start from several angles:
        angles = np.round(np.arange(-rot_ratio*180, rot_ratio*180, step_size), 2)
        angles = angles[1:]
        iterations = angles

        final_angles = []
        best_loss = 0
        best_param = None
        image = orig_kspace_to_images(kspace)
        image_ch = complex_to_channels(image)
        rotation_mask = torch.ones((1, 1, image_ch.shape[-2], image_ch.shape[-1])).float().cuda()

        target_image = fastmri.rss(torch.abs(image), 1)
        target_cropped = center_crop(target_image, [320, 320])
        target_ssim = target_cropped.cpu().detach().numpy()[0]
        if relevant_only:
            target_ssim_relreg = target_ssim[y0_int:y1_int, x0_int:x1_int]
        else:
            target_ssim_relreg = target_ssim

        for i in range(len(iterations)):
            for transform in self.transforms:
                transform.rot_ratio = rot_ratio
                if i == 0:
                    __ = transform.init_parameters()
                transform.param = torch.Tensor([angles[i]])

                # The rotation mask tells us which pixels are outside the visible area.
                # As we only considered very small angles (<= 5 degree) all pixels are within the visible area
                rotation_forwarded = transform.forward(rotation_mask)

                transformed_image = transform.forward(image_ch)  # affine matrix is calculated in forward function

                transformed_image = channels_to_complex(transformed_image)

                kspace_again = image_to_orig_kspace(transformed_image)

                if acq is not None:
                    kspace_again[:, :, :, :acq[0]] = 0
                    kspace_again[:, :, :, acq[1]:] = 0

                # put the sampling mask on the k-space
                unders_kspace = kspace_again * mask.squeeze(-1)
                target_image = target_image.unsqueeze(1)

                model_input, mean, std = self.input_from_kspace.make_model_input(unders_kspace)

                if used_model == 'unet':
                    output = model.forward(model_input)
                elif used_model == 'varnet':
                    output = model.forward(model_input, mask_unsqueezed, num_low_frequencies)
                    output = output.unsqueeze(0)
                output_backward = transform.backward(output)
                rotation_mask_backwarded = transform.backward(rotation_forwarded)
                output_center = TF.center_crop(output_backward, [320,320])
                rotation_mask_center = TF.center_crop(rotation_mask_backwarded, [320,320])

                seen_pixels = torch.isclose(rotation_mask_center,torch.tensor(1).float())
                if used_model == 'unet':
                    output_center = unnormalize(output_center, mean, std)

                output_center_ssim = output_center.cpu().detach().numpy()
                if relevant_only:
                    output_center_relreg = output_center_ssim[0,0,y0_int:y1_int, x0_int:x1_int]
                else:
                    output_center_relreg = output_center_ssim[0,0]
                ssim_loss = structural_similarity(target_ssim_relreg, output_center_relreg, data_range=target_ssim.max()-target_ssim.min())
                psnr_loss = peak_signal_noise_ratio(target_ssim_relreg, output_center_relreg, data_range=target_ssim.max()-target_ssim.min())
                mse_loss = mean_squared_error(target_ssim_relreg, output_center_relreg)

                print('loss: {}'.format(ssim_loss))
                angle=transform.param
                print('angle ', angle)
                angle_array.append(angle)
                loss_array.append(ssim_loss)
                psnr_array.append(psnr_loss)
                mse_array.append(mse_loss)
                if mse_loss > best_loss:
                    best_loss = mse_loss
                    best_param = transform.param

                model.zero_grad()
                final_angles.append(angle.item())

        # Make use of the best_param to find the worst-case perturbation
        transform.param = best_param
        best_angle = best_param
        image = orig_kspace_to_images(kspace)
        image = complex_to_channels(image)
        transformed_image = transform.forward(image)
        transformed_image = channels_to_complex(transformed_image)
        transformed_target = fastmri.rss(torch.abs(transformed_image), 1)
        kspace_again = image_to_orig_kspace(transformed_image)
        masked_kspace = kspace_again * mask.squeeze(-1)
        print('best angle: {}'.format(best_param))

        return masked_kspace, target_image, transformed_target, final_angles, loss_array, psnr_array, mse_array, best_angle

    def transform_kspace_noise(self, kspace, model, mask, mask2,
                                                num_low_frequencies=None, used_model='unet', acq=None, relative_error=0,
                                                relevant_only=True, position=None, n_iter=5):
        """
        This model tries to find the worst perturbation within the boundaries of the transforms in self.transforms

        :param kspace: the original k-space
        :param model: the model we want to attack
        :param mask: the mask to create the undersampled k-space
        :param mask2: the annotation of the anomality
        :param num_low_frequencies: number of low frequencies, input-parameter for varnet
        :param used_model: the model used
        :param acq: two integers indicating the minimum and maximum k-space column where data was sampled
        :param relative_error: the currently allowed relative error
        :param relevant_only: whether or not we attack only the relevant region (True) or the complete image (False)
        :param position: the x0, y0, x1, y1 position of the annotation, important for SSIM-loss
        :param n_iter: how many steps we want to invest for the adversarial attack.
        :return: transformed (worst-case) k-space
        """
        model.eval()
        if self.loss_name == 'SSIM' or self.loss_name == 'MSE+SSIM':
            assert position is not None, 'Position has to be given as argument when using SSIM'
            x0, y0, w, h = position
        print('start adversarial attack:')
        data_size = kspace.size()
        kspace_norms = torch.linalg.vector_norm(kspace, ord=2, dim=[-1,-2])
        loss_array = []
        mask_unsqueezed = torch.unsqueeze(mask, 0)
        orig_image = orig_kspace_to_images(kspace)
        target_image = fastmri.rss(torch.abs(orig_image), 1)

        mask = torch.permute(mask, (0, 1, 3, 2))
        transform = self.transforms[0]
        transform.acq = acq
        for i in range(n_iter+1):
            if i == 0:
                __ = transform.init_parameters(data_size, kspace_norms*relative_error)
            transform.train() # params need to get the gradient
            transformed_kspace = transform.forward(kspace)
            unders_kspace = transformed_kspace * mask
            if i == n_iter:
                # We have the undersampled transformed kspace, so we can stop here already.
                break
            model_input, mean, std = self.input_from_kspace.make_model_input(unders_kspace)
            loss_target = TF.center_crop(target_image, [320,320])
            if relative_error == 0:
                # We do not need to start the adversarial attack, since there won't be one
                break
            if i != n_iter:
                if used_model == 'unet':
                    output = model(model_input)
                    output_unnormalized = unnormalize(output, mean, std)
                    output_center = TF.center_crop(output_unnormalized, [320, 320])
                elif used_model == 'varnet':
                    output = model(model_input, mask_unsqueezed, num_low_frequencies)
                    output_center = TF.center_crop(output, [320, 320])
                    output_center = torch.unsqueeze(output_center, 0)
                if relevant_only:
                    if self.loss_name == 'MSE' or self.loss_name == 'MSE+SSIM':
                        loss1 = self.loss1(torch.abs(output_center)[mask2],
                            torch.abs(loss_target)[mask2[0]])
                    if self.loss_name == 'SSIM' or self.loss_name == 'MSE+SSIM':
                        # We need to keep the structural information here
                        loss2 = self.loss2(torch.abs(loss_target)[:,y0:y0+h, x0:x0+w],
                                            torch.abs(output_center)[:,:,y0:y0+h, x0:x0+w],
                                            torch.full([1], loss_target.max()).cuda())
                else:
                    if self.loss_name == 'MSE' or self.loss_name == 'MSE+SSIM':
                        loss1 = self.loss1(torch.abs(output_center), torch.abs(loss_target))
                    if self.loss_name == 'SSIM' or self.loss_name == 'MSE+SSIM':
                        loss2 = self.loss2(torch.abs(loss_target), torch.abs(output_center), torch.full([1], loss_target.max()).cuda())
                if self.loss_name == 'MSE':
                    loss = loss1
                elif self.loss_name == 'SSIM':
                    loss = loss2
                elif self.loss_name == 'MSE+SSIM':
                    if i == 0:
                        mse_baseline = torch.abs(loss1.detach().clone().requires_grad_(False))
                    loss = loss1/mse_baseline + loss2
                print('loss: {}'.format(loss.item()))
                loss.backward()
                transform.optimize_parameters(step_size=0.5, mask=mask)
                model.zero_grad()
                loss_array.append(loss.item())
        print('End adversarial attack')
        return unders_kspace
