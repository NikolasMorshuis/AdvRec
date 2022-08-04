import torch
import torch.linalg as la


def unit_normalize(d, ord=2):
    if d.dtype == torch.complex64:
        d = torch.view_as_real(d)
    norm = torch.linalg.vector_norm(d, ord=ord, dim=[-3, -2, -1]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # We only renormalize the tensor if the norm is larger than 1
    norm = torch.where(norm < torch.tensor(1).float().cuda(), torch.tensor(1).float().cuda(), norm)
    d = d/(norm+1e-20)
    return d


class KSpaceNoise:
    """
    This class creates an adversarial attack on the k-space noise.
    You can adjust that only not measured k-space data can be changed.
    """

    def __init__(self,
                 config_dict=None,
                 use_gpu=True, batch_size=1,
                 power_iteration=False):
        if config_dict is None:
            config_dict = {
                'epsilon': 1e-4,
                'masked_data_only': False,
                'attack_mode': 2,
            }
        self.batch_size = batch_size
        self.masked_data_only = config_dict['masked_data_only']
        # self.epsilon is only used if self.kspace_norms are None after initialization
        self.epsilon = config_dict['epsilon']
        self.power_iteration = power_iteration
        self.was_initialized = False
        self.ord = config_dict['attack_mode'] # l1-attack (1), l2-attack (2), l_inf-attack ('inf')
        self.kspace_norms = None
        self.acq = None

        self.use_gpu = use_gpu
        if self.use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.param = None

    def init_parameters(self, data_size, kspace_norms=None):
        '''
        initialize transformation parameters
        return random transformaion parameters
        '''
        # with the kspace_norms it is easier to find the relative error
        if kspace_norms is not None:
            self.kspace_norms = kspace_norms.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        elif self.kspace_norms is not None:
            self.kspace_norms = self.kspace_norms.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Gaussian noise
        noise_real = torch.randn(
                *data_size, device=self.device, dtype=torch.float32)
        noise_imag = torch.randn(
                     *data_size, device=self.device, dtype=torch.float32)

        noise = torch.concat((noise_real.unsqueeze(-1), noise_imag.unsqueeze(-1)), dim=-1)

        if self.acq is not None:
            # We do not want to sample noise where we do not have measurements
            noise[:, :, :, :self.acq[0]] = 0
            noise[:, :, :, self.acq[1]:] = 0

        if self.kspace_norms is not None:
            noise = self.kspace_norms * unit_normalize(noise, self.ord)
        else:
            noise = self.epsilon * unit_normalize(noise, self.ord)
        noise = torch.view_as_complex(noise)
        self.param = noise
        self.was_initialized = True
        return noise

    def transform(self, data, noise):
        return data + noise

    def forward(self, data):
        if self.param is None:
            self.init_parameters(data.size(), kspace_norms=self.kspace_norms)
        transformed_data = self.transform(data, self.param)
        self.diff = data - transformed_data
        return transformed_data

    def optimize_parameters(self, step_size=None, mask=None, ord=None):
        if step_size is not None:
            self.step_size = step_size
        if ord is None:
            ord=self.ord

        # we can try to add the normalized grad to params
        grad = self.param.grad.detach()
        param = self.param/la.vector_norm(self.param)+step_size*grad/la.vector_norm(grad)
        self.param = param.detach()

        # Non-essential, just interesting if you want to check how you can perturb the data without changing the
        # reconstruction algorithm (only perturb the data that is not sampled)
        if mask is not None and self.masked_data_only:
            if mask.shape[-1] != self.param.shape[-1]:
                mask = torch.permute(mask, (0,1,3,2))
            self.param = self.param * ~mask
        if self.acq is not None:
            self.param[:, :, :, :self.acq[0]] = 0
            self.param[:, :, :, self.acq[1]:] = 0

        if self.kspace_norms is not None:
            self.param = self.kspace_norms * unit_normalize(torch.view_as_real(self.param), ord)
        else:
            self.param = self.epsilon * unit_normalize(torch.view_as_real(self.param), ord)
        self.param = torch.view_as_complex(self.param)
        return self.param

    def train(self):
        self.param = torch.nn.Parameter(self.param, requires_grad=True)
