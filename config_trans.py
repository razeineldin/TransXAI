from glob import glob

# GPU handling
import tensorflow as tf
import os

# Tensorflow 2.XX\n",
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # '0,1'

gpus = tf.config.experimental.list_physical_devices('GPU')
#     print("Num GPUs Available: {len(gpus)}")
if gpus:
    if float(tf.__version__[:3]) >= 2.0:
        tf.random.set_seed(256) # tf 2.XX

        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs\n")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        tf.set_random_seed(256) #tf 1.XX
        from tensorflow.keras.backend.tensorflow_backend import set_session

        gpu_config = tf.ConfigProto(allow_soft_placement=True)
        gpu_config.gpu_options.allow_growth = True
        set_session(tf.Session(config=gpu_config))



"""

The required configurations for training phase ('prepare_Data.py', 'train.py').

"""

cfg = dict()



"""
The coordinates to crop brain volumes. For example, a brain volume with the 
One can set the x0,x1,... by calculating none zero pixels through dataset. 
Note that the final three shapes must be divisible by the network downscale rate.
"""
cfg['crop_coord']            =  {'x0':29, 'x1':221, #'x0':42, 'x1':194,
                                 'y0':29, 'y1':221,
                                 'z0':2,  'z1':146}



"""
The path to all brain volumes (ex: suppose we have a folder 'MICCAI_BraTS_2019_Data_Training'
that contains two HGG and LGG folders so:
data_dir='./MICCAI_BraTS_2019_Data_Training/*/*')
"""
cfg['data_dir']              = './BRATS_2019/MICCAI_BraTS_2019_Data_Training/*/*'



"""
The final data shapes of saved table file.
"""
cfg['table_data_shape']      =  (cfg["crop_coord"]['z1']-cfg["crop_coord"]['z0'],
                                 cfg["crop_coord"]['y1']-cfg["crop_coord"]['y0'], 
                                 cfg["crop_coord"]['x1']-cfg["crop_coord"]['x0'])



"""
BraTS datasets contain 4 channels: (FLAIR, T1, T1ce, T2)
"""
cfg['data_channels']         = 4



"""
'axial', 'sagittal' or 'coronal'. The 'view' has no effect in "prepare_data.py".
All 2D slices and the model will be prepared  with respect to 'view'.
"""
cfg['view']                  = 'axial' # axial, coronal, sagittal




"""
The path to save table file + k-fold files
"""
cfg['save_data_dir']         = f'./data_trans/' #_{cfg["view"]}/'
if not os.path.exists(cfg['save_data_dir']):
    os.makedirs(cfg['save_data_dir'])


"""
The path to save models + log files + tensorboards
"""
cfg['save_dir']        = f'./save_trans/' #_{cfg["view"]}/'
if not os.path.exists(cfg['save_dir']):
    os.makedirs(cfg['save_dir'])


"""
k-fold cross-validation
"""
cfg['k_fold']                = 5
cfg['fold']                  = 0 # 0,1,2,3,4


"""
The defualt path of saved table.
"""
cfg['hdf5_dir']              = f'./{cfg["save_data_dir"]}/data{cfg["fold"]}.hdf5'



"""
The path to brain indexes of specific fold (a numpy file that was saved in ./data/ by default)
"""
cfg['brains_idx_dir']        = f'./{cfg["save_data_dir"]}/fold{cfg["fold"]}_idx.npy'




"""
The batch size for training and validating the model
"""
cfg['batch_size']            = 16
cfg['val_batch_size']        = 32



"""
The augmentation parameters.
"""
cfg['hor_flip']              = True
cfg['ver_flip']              = True
cfg['rotation_range']        = 20
cfg['zoom_range']            = 0.2



"""
The leraning rate and the number of epochs for training the model
"""
cfg['epochs']                = 250
cfg['lr']                    = 0.008 #XX0.01 after epoc 121, val_batch=16



"""
If True, use process-based threading. "https://keras.io/models/model/"
"""
cfg['multiprocessing']       = False 



"""
Maximum number of processes to spin up when using process-based threading. 
If unspecified, workers will default to 1. If 0, will execute the generator 
on the main thread. "https://keras.io/models/model/"
"""
cfg['workers']               = 1



"""
The depth of the U-structure 
"""
cfg['levels']                = 3



"""
The number of channels of the first conv
"""
cfg['start_chs']             = 64



"""
If specified, before training, the model weights will be loaded from this path otherwise
the model will be trained from scratch.
"""
cfg['epoch_num']             = '0'
cfg['load_model_dir']        = None #glob(f"./save_trans/axial_fold{cfg["fold"]}/model.{cfg['epoch_num']}*.hdf5")[0]  #None
cfg['initial_epoch']         = int(cfg['epoch_num']) if cfg['load_model_dir'] else 0  # continue training

print("\n\n\n-----------------------------------------------------------------")
print("TransXAI model configuration:...")
print("\tview:", cfg['view'])
print("\tfold:", cfg['fold'])
print("\tbatch_size:", cfg['batch_size'])
print("\tlr:", cfg['lr'])
print("\tstart_chs:", cfg['start_chs'])
print("\tepoch_num:", cfg['epoch_num'])
print("\tinitial_epoch:", cfg['initial_epoch'])
print("\tload_model_dir:", cfg['load_model_dir'])
print("-----------------------------------------------------------------\n\n\n")
