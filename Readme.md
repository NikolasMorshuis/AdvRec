# Adversarial Robustness of MR Image Reconstruction under Realistic Perturbations

This repository contains the code for our MICCAI Workshop paper "Adversarial Robustness of MR Image Reconstruction under Realistic Perturbations".
With this code, you can find realistic perturbations (k-space noise and rotation-angles) that can significantly 
alter the annotated anomalies visible in the MRI-scan.

Note that in contrast to the paper, in the current version of the code we conduct the experiments with the original fastMRI-baseline models, that are 
automatically downloaded if you run the code. In contrast to the smaller versions used in the workshop paper, the versions used
in this code are trained on both training and validation set. Therefore, when evaluating on the validation set, the model 
has already seen the datapoints, which might affect the adversarial robustness. However, our experiments have shown that the model
is still susceptible to our perturbations and hence we keep this option for now because of the easy availablility of the fastMRI-models.

## How to use the code?
In order to use the code, you need the fastMRI image data of the multicoil-knee validation set. You can ask access to this dataset on the 
fastMRI website: https://fastmri.org/

Once you have downloaded the data, you can execute the code with the following command:

```
python adversarial.py
--data_path your/fastmri/datapath/raw/knee_multicoil
--mask_type random
--use_dataset_cache_file False
--accelerations 8
--center_fractions 0.04
--loss_name MSE
--relevant_only True
--used_model_array varnet
--used_transform noise
```