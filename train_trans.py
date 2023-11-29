import os
import tables
import numpy as np
from config_trans import cfg
from model_trans import transxai_model
from data_generator import CustomDataGenerator
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard


def train_model(hdf5_dir, brains_idx_dir, view, batch_size=16, val_batch_size=32,
                lr=0.01, epochs=100, hor_flip=False, ver_flip=False, zoom_range=0.0, save_dir='./save/',
                start_chs=64, levels=3, multiprocessing=False, load_model_dir=None, initial_epoch=0):
    """

    The function that builds/loads UNet model, initializes the data generators for training and validation, and finally 
    trains the model.

    """
    # preparing generators
    hdf5_file        = tables.open_file(hdf5_dir, mode='r+')
    brain_idx        = np.load(brains_idx_dir)
    datagen_train    = CustomDataGenerator(hdf5_file, brain_idx, batch_size, view, 'train',
                                    hor_flip, ver_flip, zoom_range, shuffle=True)
    datagen_val      = CustomDataGenerator(hdf5_file, brain_idx, val_batch_size, view, 'validation', shuffle=False)
    
    # add callbacks    
    save_dir     = os.path.join(save_dir, '{}_{}'.format(view, os.path.basename(brains_idx_dir)[:5]))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    logger       = CSVLogger(os.path.join(save_dir, 'log.txt'))
    #checkpointer = ModelCheckpoint(filepath = os.path.join(save_dir, 'model.hdf5'), verbose=1, save_best_only=False) #True)
    checkpointer = ModelCheckpoint(filepath = os.path.join(save_dir, "trans_f"+str(cfg['fold'])+".{epoch:03d}-{val_generalized_dice:.2f}.hdf5") , monitor='val_generalized_dice', save_best_only=False, save_weights_only=False) #True)


    tensorboard  = TensorBoard(os.path.join(save_dir, 'tensorboard'))
    callbacks    = [logger, checkpointer, tensorboard]        
    
    # building the model
    model_input_shape = datagen_train.data_shape[1:]
    print("model_input_shape:", model_input_shape)

    model             = transxai_model(model_input_shape,
                                filter_num=[16, 32, 48, 64], n_labels=4, stack_num_down=2, stack_num_up=2,
                                embed_dim=768, num_mlp = 3072, num_heads=2, num_transformer=2, # my mode (up=down=2)
                                activation='ReLU', mlp_activation='GELU', output_activation='Softmax', 
                                batch_norm=True, pool=True, unpool='bilinear', name='transunet_tiny', learning_rate=lr, saved_model_dir=load_model_dir)

    # training the model
    model.fit_generator(datagen_train, epochs=epochs, initial_epoch=initial_epoch, use_multiprocessing=multiprocessing, 
                        callbacks=callbacks, validation_data = datagen_val)


   
if __name__ == '__main__':
    
    
    train_model(cfg['hdf5_dir'], cfg['brains_idx_dir'], cfg['view'], cfg['batch_size'], 
                cfg['val_batch_size'], cfg['lr'], cfg['epochs'], cfg['hor_flip'], cfg['ver_flip'], 
                cfg['zoom_range'], cfg['save_dir'], cfg['start_chs'], cfg['levels'], 
                cfg['multiprocessing'], cfg['load_model_dir'], cfg['initial_epoch'])
    
    
    
    
    
