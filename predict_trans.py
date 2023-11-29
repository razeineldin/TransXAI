import os
import numpy as np
import nibabel as nib
from glob import glob
from model_trans import transxai_model
from tensorflow.keras.models import load_model
from config_trans import cfg

from scipy import ndimage
from skimage import morphology

def read_brain(brain_dir, mode='train', x0=29, x1=221, y0=29, y1=221, z0=2, z1=146):

    """
    A function that reads and crops a brain modalities (nii.gz format)
    
    Parameters
    ----------
    brain_dir : string
        The path to a folder that contains MRI modalities of a specific brain
    mode : string
        'train' or 'validation' mode. The default is 'train'.
    x0, x1, y0, y1, z0, z1 : int
        The coordinates to crop 3D brain volume. For example, a brain volume with the 
        shape [x,y,z,modalites] is cropped [x0:x1, y0:y1, z0:z1, :] to have the shape
        [x1-x0, y1-y0, z1-z0, modalities]. One can calculate the x0,x1,... by calculating
        none zero pixels through dataset. Note that the final three shapes must be divisible
        by the network downscale rate.
        
    Returns
    -------
    all_modalities : array
        The cropped modalities (+ gt if mode='train')
    brain_affine : array
        The affine matrix of the input brain volume
    brain_name : str
        The name of the input brain volume

    """
    
    brain_dir = os.path.normpath(brain_dir)
    flair     = glob( os.path.join(brain_dir, '*_flair*.nii.gz'))
    t1        = glob( os.path.join(brain_dir, '*_t1*.nii.gz'))
    t1ce      = glob( os.path.join(brain_dir, '*_t1ce*.nii.gz'))
    t2        = glob( os.path.join(brain_dir, '*_t2*.nii.gz'))
    
    if mode=='train':
        gt             = glob( os.path.join(brain_dir, '*_seg*.nii.gz'))
        modalities_dir = [flair[0], t1[0], t1ce[0], t2[0], gt[0]]
        
    elif mode=='validation':
        modalities_dir = [flair[0], t1[0], t1ce[0], t2[0]]   
    
    all_modalities = []    
    for modality in modalities_dir:      
        nifti_file   = nib.load(modality)
        brain_numpy  = np.asarray(nifti_file.dataobj)    
        all_modalities.append(brain_numpy)
        
    # all modalities have the same affine, so we take one of them (the last one in this case),
    # affine is just saved for preparing the predicted nii.gz file in the future.       
    brain_affine   = nifti_file.affine
    all_modalities = np.array(all_modalities)
    all_modalities = np.rint(all_modalities).astype(np.int16)
    all_modalities = all_modalities[:, x0:x1, y0:y1, z0:z1]
    # to fit keras channel last model
    all_modalities = np.transpose(all_modalities) 
    # tumor grade + name
    brain_name     = os.path.basename(os.path.split(brain_dir)[0]) + '_' + os.path.basename(brain_dir) 

    return all_modalities, brain_affine, brain_name
    
    

def normalize_slice(slice):
    
    """
    Removes 1% of the top and bottom intensities and perform
    normalization on the input 2D slice.
    """
    
    b = np.percentile(slice, 99)
    t = np.percentile(slice, 1)
    slice = np.clip(slice, t, b)
    if np.std(slice)==0:
        return slice
    else:
        slice = (slice - np.mean(slice)) / np.std(slice)
        return slice
    

def normalize_volume(input_volume):
    
    """
    Perform a slice-based normalization on each modalities of input volume.
    """
    normalized_slices = np.zeros_like(input_volume).astype(np.float32)
    for slice_ix in range(4):
        normalized_slices[..., slice_ix] = normalize_slice(input_volume[..., slice_ix])

    return normalized_slices      


def remove_small_objects(img):
        binary = np.copy(img)
        binary[binary>0] = 1
        labels = morphology.label(binary)
        labels_num = [len(labels[labels==each]) for each in np.unique(labels)]
        rank = np.argsort(np.argsort(labels_num))
        index = list(rank).index(len(rank)-2)
        new_img = np.copy(img)
        new_img[labels!=index] = 0
        return new_img

def save_predicted_results(prediction, brain_affine, view, output_dir, post_process=False, whole_only=False, threshold=200, z_main=155, z0=2, z1=146, y_main=240, y0=29, y1=221, x_main=240, x0=29, x1=221):
    
    """
    Save the segmented results into a .nii.gz file, so that it can be uploaded to the BraTS server.
    Note that to correctly save the segmented brains, it is necessery to set x0, x1, ... correctly.
    
    Parameters
    ----------
    prediction : array
        The predictred brain.
    brain_affine : array
        The affine matrix of the predicted brain volume
    view : str
        'axial', 'sagittal' or 'coronal'. The 'view' is needed to reconstruct output axes.
    output_dir : str
        The path to save .nii.gz file.


    """
    
    prediction = np.argmax(prediction, axis=-1).astype(np.uint16)            
    prediction[prediction==3] = 4
    
    if view=="axial":
        prediction    = np.pad(prediction, ((z0, z_main-z1), (y0, y_main-y1), (x0, x_main-x1)), 'constant')
        prediction    = prediction.transpose(2,1,0)
    elif view=="sagital":
        prediction    = np.pad(prediction, ((x0, x_main-x1), (y0, y_main-y1), (z0 , z_main-z1)), 'constant')
    elif view=="coronal":
        prediction    = np.pad(prediction, ((y0, y_main-y1), (x0, x_main-x1), (z0 , z_main-z1)), 'constant')
        prediction    = prediction.transpose(1,0,2)


    if post_process:
        seg_enhancing = (prediction == 4)
        if np.sum(seg_enhancing) < threshold:
            prediction[seg_enhancing] = 1
            print("\tConverted {} voxels from label 4 to label 1!".format(np.sum(seg_enhancing)))


        if whole_only:
            # morphological opening and closing to remove small objects
            whole_data = (prediction>0).astype(np.uint8)

            # remove small objects in the segmentation
            whole_post = remove_small_objects(whole_data).astype(np.uint8)

            # apply the filter to edema
            edema_data = (prediction==2)
            prediction_copy = np.copy(prediction)
            prediction[whole_post == 0] = 0

            # make sure that labels 1 and 4 are kept
            prediction[prediction_copy == 1] = 1
            prediction[prediction_copy == 4] = 4

        else:
            # remove small objects in the segmentation
            prediction = remove_small_objects(prediction).astype(np.uint8)

    # save nifti images
    prediction_ni    = nib.Nifti1Image(prediction, brain_affine)
    prediction_ni.to_filename(output_dir+ '.nii.gz')


if __name__ == '__main__':
       
    val_data_dir       = './BRATS_2019/MICCAI_BraTS_2019_Data_Validation/*'
    view               = 'axial'
    epoch_num          = 236 # starts from 1
    fold               = 0
    saved_model_dir    = glob(f'./save_trans/{view}_fold{fold}/trans_f{fold}.{epoch_num:03d}*.hdf5')[0]  #model.hdf5'


    save_pred_dir      = f'./predict_trans/{view}_fold{fold}/trans_f{fold}_ep_{epoch_num:03d}'
    #save_pred_dir      = f'./predict_trans/train_HGG_model_ep_{epoch_num}'
    batch_size         = 32
    save_weights_only  = True
    overwrite          = False
    post_process       = True
    whole_only         = False #True
    save_pred_dir      = save_pred_dir+'_post' if post_process else save_pred_dir 
    save_pred_dir      = save_pred_dir+'_whole' if whole_only else save_pred_dir+'/'

    print("\n\n\n-----------------------------------------------------------------")
    print("TransXAI model prediction:...")
    print("\tview:", view)
    print("\tfold:", fold)
    print("\tbatch_size:", batch_size)
    print("\tepoch_num:", epoch_num)
    print("\tsaved_model_dir:", saved_model_dir)
    print("-----------------------------------------------------------------\n\n\n")


    if not os.path.isdir(save_pred_dir):
        os.makedirs(save_pred_dir)
       
    all_brains_dir = glob(val_data_dir)
    all_brains_dir.sort()
    
    if view == 'axial':
        view_axes = (0, 1, 2, 3)            
    elif view == 'sagittal': 
        view_axes = (2, 1, 0, 3)
    elif view == 'coronal':
        view_axes = (1, 2, 0, 3)            
    else:
        ValueError('unknown input view => {}'.format(view))
    
    if save_weights_only:
        # building the model
        model_input_shape = cfg['table_data_shape'][1:] + (4,)
        print("model_input_shape:", model_input_shape)
        model             = transxai_model(model_input_shape,
                                filter_num=[16, 32, 48, 64], n_labels=4, stack_num_down=2, stack_num_up=2,
                                embed_dim=768, num_mlp = 3072, num_heads=2, num_transformer=2, # my mode (up=down=2)
                                activation='ReLU', mlp_activation='GELU', output_activation='Softmax', 
                                batch_norm=True, pool=True, unpool='bilinear', name='transunet_tiny', learning_rate=cfg['lr'], saved_model_dir=saved_model_dir)


    else:
        model        = load_model(saved_model_dir, compile=False)
    
    for i, brain_dir in enumerate(all_brains_dir):    
        if os.path.isdir(brain_dir):
            output_dir                      = os.path.join(save_pred_dir, os.path.basename(brain_dir))
            #print(output_dir)
            if not os.path.exists(output_dir+".nii.gz") or overwrite:
                #print(os.path.basename(brain_dir))
                #if os.path.basename(brain_dir) != "BraTS19_CBICA_ATW_1": continue

                print(f"Predicting volume ID ({i+1}/{len(all_brains_dir)-2}): ", os.path.basename(brain_dir))
                open(output_dir+".nii.gz", 'a').close()
                all_modalities, brain_affine, _ = read_brain(brain_dir, mode='validation')
                all_modalities                  = all_modalities.transpose(view_axes)
                all_modalities                  = normalize_volume(all_modalities)
                prediction                      = model.predict(all_modalities, batch_size=batch_size, verbose=1)
                save_predicted_results(prediction, brain_affine, view, output_dir, post_process, whole_only)
            else:
                continue
            
            
