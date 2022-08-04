import torch
import torchvision.transforms.functional as TF
from utils.utils import orig_kspace_to_images, complex_to_channels, channels_to_complex, image_to_orig_kspace


class AdvRotation:
    """
    Conducts a grid search through possible angles and finds the angle leading to the worst reconstruction
    """

    def __init__(self,
                 rot_ratio=0.1,
                 use_gpu=True, batch_size=1):

        self.batch_size = batch_size
        self.param = None
        self.rot_ratio = rot_ratio
        self.diff = None
        self.step_size = 1  # step size for optimizing data augmentation
        self.power_iteration = False
        self.affine_matrix = None
        self.was_initialized = False
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_parameters(self):
        rot_angle = self.create_init_affine_transformation()
        self.param = rot_angle
        self.was_initialized = True
        return rot_angle

    def backward(self, data):
        assert self.param is not None, "forward pass required for backward operation"
        warped_back_output = self.transform(data, -self.param)
        return warped_back_output

    def create_init_affine_transformation(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        rot_angle = (2*torch.rand(batch_size, 1,
                        dtype=torch.float32, device=self.device)-1) * self.rot_ratio*180
        return rot_angle

    def transform(self, data, angle=None):
        if angle is None:
            angle = self.param
        original_shape = data.shape[-2:]
        data = TF.resize(data, size=[original_shape[0] * 2, original_shape[1] * 2], interpolation=TF.InterpolationMode.BICUBIC)
        shape_before_pad = data.shape[-2:]
        py, px = self._get_affine_padding_size(data, angle, 1, [0,0])
        data = TF.pad(data, padding=[py, px], padding_mode='reflect')
        data = TF.rotate(data, angle.item(), interpolation=TF.InterpolationMode.BILINEAR)
        data = TF.center_crop(data, shape_before_pad)
        data = TF.resize(data, original_shape, interpolation=TF.InterpolationMode.BICUBIC)
        return data

    def _get_affine_padding_size(self, image, angle, scale, shear):
        """
        Calculates the necessary padding size before applying the
        general affine transformation. The output image size is determined based on the
        input image size and the affine transformation matrix.
        from MRAugment: https://github.com/z-fabian/MRAugment
        """
        h, w = image.shape[-2:]
        corners = [
            [-h/2, -w/2, 1.],
            [-h/2, w/2, 1.],
            [h/2, w/2, 1.],
            [h/2, -w/2, 1.]
        ]
        mx = torch.tensor(TF._get_inverse_affine_matrix([0.0, 0.0], -angle, [0, 0], scale, [-s for s in shear])).reshape(2,3)
        corners = torch.cat([torch.tensor(c).reshape(3,1) for c in corners], dim=1)
        tr_corners = torch.matmul(mx, corners)
        all_corners = torch.cat([tr_corners, corners[:2, :]], dim=1)
        bounding_box = all_corners.amax(dim=1) - all_corners.amin(dim=1)
        px = torch.clip(torch.floor((bounding_box[0] - h) / 2), min=0.0, max=h-1)
        py = torch.clip(torch.floor((bounding_box[1] - w) / 2),  min=0.0, max=w-1)
        return int(py.item()), int(px.item())

    def forward(self, data):
        if self.param is None:
            self.init_parameters()
        transformed_input = self.transform(data, None)
        self.diff = data - transformed_input
        return transformed_input

    def forward_kspace(self, kspace):
        image = orig_kspace_to_images(kspace)
        image = complex_to_channels(image)
        transformed_image = self.forward(image)
        transformed_image = channels_to_complex(transformed_image)
        kspace_again = image_to_orig_kspace(transformed_image)
        return kspace_again
