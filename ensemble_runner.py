# Step 1: Imports and configuaration
import SimpleITK as sitk # staple

import matplotlib.pyplot as plt
import numpy as np
import os, sys
import glob
import shutil
from tqdm import tqdm

import nibabel as nib

# Step 2: confgurations and setup

config = dict()

config["post_enhancing"] = True
 
# BraTS paths
config["brats_input_path"] = "./predict_trans/"
config["brats_output_path"] = "./predict_trans/axial_ensemble/"

# Test training case
input_path = config["brats_input_path"]+f"axial_fold0/trans_f0_ep_{epoch_nums[0]}_post"
filenames = os.listdir(input_path) # all modalities filenames
filenames.sort()

# ensembling settings

# models best epochs
epoch_nums = ["236", "168", "210", "170", "112"]

models_list = [f"axial_fold0/trans_f0_ep_{epoch_nums[0]}_post", 
               f"axial_fold1/trans_f1_ep_{epoch_nums[1]}_post_whole", 
               f"axial_fold2/trans_f2_ep_{epoch_nums[2]}_post_whole", 
               f"axial_fold3/trans_f3_ep_{epoch_nums[3]}_post", 
               f"axial_fold4/trans_f4_ep_{epoch_nums[4]}_post"]

overwrite = False

if not os.path.exists(config["brats_output_path"]):
    os.makedirs(config["brats_output_path"])


def postprocess_tumor_staple(seg_data, post_enhancing=False, threshold=200):
    # post-process the enhancing tumor region
    seg_data[seg_data == 5] = 4
    if post_enhancing:
        seg_enhancing = (seg_data == 4)
        if np.sum(seg_enhancing) < threshold:
            if np.sum(seg_enhancing) > 0:
                seg_data[seg_enhancing] = 1
                print("\tConverted {} voxels from label 4 to label 1!".format(np.sum(seg_enhancing)))
    return seg_data.astype(np.uint8)


# Step 3: ensembling
# run on the whole validation set
for i, full_ID in enumerate(tqdm(filenames)):
    #if i>0: break
    ID = full_ID.split(".nii.gz")[0] # case ID
    #print(ID)
    if not os.path.exists(os.path.join(config["brats_output_path"], '{}.nii.gz'.format(ID))) or overwrite:
        open(os.path.join(config["brats_output_path"], '{}.nii.gz'.format(ID)), 'a').close()
    else:
        continue

    seg_imgs = []
    #print("Ensembling...")
    for model_folder in models_list:
        sitk_image_m = sitk.ReadImage(os.path.join(config["brats_input_path"], model_folder, '{}.nii.gz'.format(ID)))
        sitk_image_m = sitk.Cast(sitk_image_m, sitk.sitkUInt8)
        seg_imgs.append(sitk_image_m)

    staple_filter = sitk.MultiLabelSTAPLEImageFilter() #sitk.STAPLEImageFilter()
    #staple_filter.SetLabelForUndecidedPixels(4)
    sitk_image_staple = staple_filter.Execute(seg_imgs)
    sitk.WriteImage(sitk_image_staple, os.path.join(config["brats_output_path"], '{}.nii.gz'.format(ID)))

    # post-process
    if config["post_enhancing"]:
        sitk_image = sitk.ReadImage(os.path.join(config["brats_output_path"], '{}.nii.gz'.format(ID)))
        sitk_data = sitk.GetArrayFromImage(sitk_image)
        sitk_data_post = postprocess_tumor_staple(sitk_data, post_enhancing=True)
        sitk_image_post = sitk.GetImageFromArray(sitk_data_post)
        sitk_image_post.CopyInformation(sitk_image)
        sitk.WriteImage(sitk_image_post, os.path.join(config["brats_output_path"], '{}.nii.gz'.format(ID)))



