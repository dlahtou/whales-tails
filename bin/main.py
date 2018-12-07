import numpy as np
import pandas as pd 
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import Add, Input, AveragePooling2D, Concatenate, Conv2D, MaxPool2D, BatchNormalization, Dropout, Dense, Flatten, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.backend import int_shape
import tensorflow as tf
from tqdm import tqdm

df = pd.read_csv("../input/train.csv")

image_target=(224,448)
batch_size=100
dropout=0.2

idg = ImageDataGenerator(rescale=1./255,
                        rotation_range=20,
                        width_shift_range=0.2,
                        shear_range=20,
                        zoom_range=0.2,
                        horizontal_flip=True)

train_generator = idg.flow_from_dataframe(df,
                        directory = '../input/train',
                        x_col='Image',
                        y_col='Id',
                        target_size=image_target,
                        batch_size=batch_size)

num_classes = train_generator.num_classes

def bn_relu_conv(in_, f_maps, dropout=0.2, stride=1):
    '''
    Builds an elementary bn_relu_conv block with optional stride for downsampling
    '''
    x = BatchNormalization() (in_)
    x = Conv2D(f_maps,
               kernel_size=(3,3),
               strides=stride,
               padding='same',
               activation='relu'
              ) (x)
    x = Dropout(dropout) (x)
    
    return x

def block(in_, f_maps, dropout=0.2, downsample=False):
    '''
    Builds a resnet unit block using two bn_relu_conv blocks
    '''
    x = bn_relu_conv(in_, f_maps, dropout, stride=2 if downsample else 1)
    x = bn_relu_conv(x, f_maps, dropout)
    
    return x

def res_add(in_, residual):
    '''
    Adds resnet units together for shortcut connections
    '''
    filters = int_shape(in_)[3]
    
    # convolve to match shape if required
    if int_shape(in_) != int_shape(residual):
        residual = Conv2D(filters,
                          kernel_size=(1,1),
                          strides=2
                         ) (residual)
    
    return Add() ([in_, residual])

arch = [(64, 2), (128, 4), (256, 5), (512, 3), (1024, 3)]

# arch = [(64, 2), (128, 2), (256, 2), (512, 2)]

def build_res(arch, image_target, num_classes):

    with tf.device('/gpu:0'):
        input_ = Input((image_target[0], image_target[1], 3))
        
        initial_conv = bn_relu_conv(input_, 64, stride=2)
        
        initial_pool = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same") (initial_conv)
        
        res = initial_pool
        
        # create blocks and skip connects according to specified architecture
        for step, (f_maps, num_blocks) in enumerate(arch):
            for i in range(num_blocks):
                if i == 0 and step != 0:
                    x = block(res, f_maps, downsample=True)
                else:
                    x = block(res, f_maps, downsample=False)
                
                res = res_add(x, res)
        
        out_pool = GlobalAveragePooling2D() (res)
        output = Dense(num_classes, activation='softmax') (out_pool)
        
        model = Model(inputs=input_, outputs=output)
        adam = Adam(lr=0.001)

        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model = build_res(arch, image_target, num_classes)

len_train=len(df)

stale = EarlyStopping(patience=3, verbose=1)
checkpoint_model = ModelCheckpoint(f'whale_model.h5', verbose=1, save_best_only=True)

model.fit_generator(train_generator,
                    epochs=10,
                    steps_per_epoch=len_train//batch_size*5,
                    callbacks=[stale, checkpoint_model])

test_df = pd.DataFrame({'Image':os.listdir('../input/test')})

idg_2 = ImageDataGenerator(rescale=1./255)

test_generator = idg_2.flow_from_dataframe(dataframe=test_df,
                                         x_col='Image',
                                         directory='../input/test',
                                         shuffle=False,
                                         target_size=image_target,
                                         batch_size=40,
                                         class_mode=None #no class labels because this is the test set
                                        )

predicts = model.predict_generator(test_generator,
                                   steps=199,
                                   verbose=1)

class_names = list(train_generator.class_indices.keys())

tqdm_preds = tqdm(predicts)
predict_strings = []
for i in tqdm_preds:
    top5 = sorted(zip(i, class_names), reverse=True)[:5]
    '''for z in range(5):
        if top5[z][0] < 0.05:
            top5.insert(z, ('', 'new_whale'))
            break'''
    top5 = [k for j,k in top5[:5]]
    out_string = ' '.join(top5)
    predict_strings.append(out_string)

test_df['Id'] = predict_strings

test_df.to_csv('submission.csv', index=False)