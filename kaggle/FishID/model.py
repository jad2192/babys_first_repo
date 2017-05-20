# Here is the actual model training and testing

import numpy as np
import warnings
import os
import PIL

from convnetskeras.customlayers import Softmax4D
from keras.layers import UpSampling2D

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers.core import Dropout
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import obtain_input_shape
from keras.engine.topology import get_source_inputs

from keras.preprocessing import image
from convnetskeras.imagenet_tool import synset_to_dfs_ids
import pandas as pd



def fish_image_batch_partitioner(batch_size, fp='/fp/train', no_fish=True):
  '''Function used to load files for mini-batches from disk to be used for training.
  
     no_fish: If false only get images which contain fish. Used to train Fish-or-nofish model.
     
     Outputs a list of lists where each element list is of length batch_size.'''
    
    bs = batch_size
    images = [image for image in os.listdir(fp) if image[-3:] in ['jpg', 'png']]
    if not no_fish:
        images = [image for image in os.listdir(fp) if image[-3:]=='jpg' and image[:3]!='NoF']
    
    for k in range(50):
        random.shuffle(images)
    
    im_part = []
    k = 0
    cur_part = []
    while images:
        if k > 0 and k%bs == 0:
            im_part.append(cur_part)
            cur_part = []
        
        im = images.pop()
        cur_part.append(im)
        k += 1
    
    return im_part

def fish_image_batch_generator(batch_partition, fp = '/fp/train/',
                              full=True, Train=True, no_fish=True, loc=False ):
    '''Loads and pre-processes images from the mini-batches and makes them ready to input into CNN model.
    
       Batch_partition: An element of the output of fish_image_batch_partitioner.
                        E.g. a list of length = batch_size
       
       full: False to output only whether fish or not.
       
       no_fish: False means only use 7 categories i.e leave out the NoF category.
       
       loc: Use deafault, this was for an  feature not used.
       '''
    
    bs = len(batch_partition)
    im_batch = np.zeros((bs,224*224*3))
    im_labels = np.zeros((bs,8))
    if not no_fish:
        im_labels = np.zeros((bs,7))
    im_loc = np.zeros((bs,8))
    fish_or_not = np.zeros((bs,2))
    label_dict = {'ALB':0, 'BET':1, 'DOL':2, 'LAG':3, 'SHA':4, 'YFT':5,
                  'OTH':6, 'NoF':7}
    if not no_fish:
        label_dict = {'ALB':0, 'BET':1, 'DOL':2, 'LAG':3, 'SHA':4, 'YFT':5,
                      'OTH':6}
        
    #loc_dict = {'ALB':alb_loc, 'BET':bet_loc, 'DOL':dol_loc, 'LAG':lag_loc, 'SHA':shk_loc, 'YFT':yft_loc,
    #              'OTH':oth_loc, 'NoF':7}
    k = 0
    for fish in batch_partition:
        cur_fish = PIL.Image.open(fp + fish)

        if cur_fish.size != (256,256):
            
            cur_fish = cur_fish.resize((256,256), Image.ANTIALIAS)
            cur_fish = cur_fish.crop((15,15,239,239))
            
            im_batch[k] = (np.asarray(cur_fish,
                                  dtype='float32')).reshape((3*224**2,)) / 255
            
        else:
            cur_fish = cur_fish.crop((15,15,239,239))
     
            im_batch[k] = (np.asarray(cur_fish,
                                  dtype='float32'))[:,:,::-1].reshape((3*224**2,)) / 255
            
        if Train and not loc:
            
            im_labels[k][label_dict[fish[:3]]] = np.float32(1)
            fish_or_not[k][int(fish[:3]!='NoF')] = np.float32(1)
        
        if Train and loc:
            
            spec = fish[:3]
            num = int(''.join([x for x in fish if x.isdigit()]))
            im_loc[k] = (loc_dict[spec])[num]         
            
        cur_fish.close()
        k += 1
    if full and Train and not loc:
        return np.asarray(im_batch, dtype='float32'), np.asarray(im_labels, dtype='float32')
    elif Train and not loc:
        return np.asarray(im_batch, dtype='float32'), np.asarray(fish_or_not, dtype='float32')
    elif Train and loc:
        return np.asarray(im_batch, dtype='float32'), np.asarray(im_loc, dtype='float32') 
    else:
        return np.asarray(im_batch, dtype='float32')
        

# Pre-trained VGG16 Model taken from here:
# https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py
# Weights: https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
 

def VGG16(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):
    """Instantiates the VGG16 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_dim_ordering(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Conv2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Conv2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Conv2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Conv2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = '/fp/to/weights'      # Need to put the directory to downloaded weights here.
        else:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_dim_ordering() == 'th':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model

vgg16 = VGG16(weights='imagenet', include_top=True)


layers = list(vgg16.layers)


# Creating the Fish-or-no-fish model

input_img1 = Input((224,224,3))

output1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(input_img1)
output1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1p')(output1)
output1 = BatchNormalization(axis=1)(output1)

output1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(output1)

output1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(output1)
output1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2p')(output1)
output1 = BatchNormalization(axis=1)(output1)

output1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool2')(output1)


# Block 2/3
output1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(output1)
output1 = BatchNormalization(axis=1)(output1)
output1 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(output1)
output1= Conv2D(64, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(output1)
output1 = BatchNormalization(axis=1)(output1)
output1= MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool2')(output1)

#Block 4/5
output1= Conv2D(64, 3, 3, activation='relu', border_mode='same', name='block2_conv3')(output1)
output1 = BatchNormalization(axis=1)(output1)
output1= Conv2D(64, 3, 3, activation='relu', border_mode='same', name='block2_conv4')(output1)
output1 = BatchNormalization(axis=1)(output1)
output1= MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool3')(output1)

# FC Block


output1 = Flatten(name='flatten')(output1)
output1= Dense(2048, activation='relu', name='fc1')(output1)
output1 = BatchNormalization()(output1)
output1 = Dropout(0.45)(output1)
output1 = Dense(1024, activation='relu', name='fc2')(output1)
output1 = BatchNormalization()(output1)
output1 = Dropout(0.45)(output1)
output1 = Dense(2, activation='softmax', name='predictions')(output1)

convnet_fish_or_no = Model(input=input_img1, output = output1, name='convnet_fish_or_no')


# Creating the VGG16-fish model

input_imgf = Input((224,224,3))

output = Conv2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1',
               weights=layers[1].get_weights(),trainable=False)(input_imgf)


output = Conv2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2',
               weights=layers[2].get_weights(),trainable=False )(output)

output = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(output)

# Block 2
output = Conv2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1',
               weights=layers[4].get_weights(),trainable=False)(output)

output= Conv2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2',
              weights=layers[5].get_weights(),trainable=False)(output)

output= MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(output)

# Block 3
output = Conv2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1',
               weights=layers[7].get_weights(),trainable=False)(output)

output= Conv2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2',
              weights=layers[8].get_weights(),trainable=False)(output)

output = Conv2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3',
               weights=layers[9].get_weights(),trainable=False)(output)

output = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(output)

# Block 4
output = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1',
               weights=layers[11].get_weights(),trainable=False)(output)

output = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2',
               weights=layers[12].get_weights(),trainable=False)(output)

output = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3',
               weights=layers[13].get_weights())(output)
output = BatchNormalization(axis=1)(output)

output = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(output)

# Block 5
output= Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1',
              weights=layers[15].get_weights())(output)
output = BatchNormalization(axis=1)(output)

output= Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2',
              weights=layers[16].get_weights())(output)
output = BatchNormalization(axis=1)(output)

output = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3',
               weights=layers[17].get_weights())(output)
output = BatchNormalization(axis=1)(output)

output = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(output)

# FC Block


output = Flatten(name='flatten')(output)
output= Dense(4096, activation='relu', name='fc1', weights=layers[20].get_weights())(output)
output = BatchNormalization()(output)
output = Dropout(0.75)(output)
output = Dense(4096, activation='relu', name='fc2', weights=layers[21].get_weights())(output)
output = BatchNormalization()(output)
output = Dropout(0.65)(output)
output = Dense(7, activation='softmax', name='predictions')(output)

vgg16_fish = Model(input=input_imgf, output=output)



def create_class_weight(labels_dict,mu=0.53):
    total = np.sum(np.array(list(labels_dict.values())))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = np.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

class_count_fish = {0:1719,1:200, 2:117, 3:66, 4:176, 5:734, 6:299}
class_count_FoN = {0:2500, 1:500}

class_wt_fish = create_class_weight(class_count_fish)
class_wt_FoN = create_class_weight(class_count_FoN)

from keras.optimizers import adam

sgd = adam(lr=0.00001)

convnet_fish_or_no.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

sgd2 = adam(lr=0.000007)

vgg16_fish.compile(optimizer=sgd2, loss='categorical_crossentropy', metrics=['accuracy'])


# Training Fish-or-no-fish
train_path = '/fp/train'

epoch = 1

while epoch <= 9:
    
    train_batch = fish_image_batch_partitioner(batch_size=32, fp=train_path)
    
    for part in train_batch:
        vb = int(train_batch.index(part) % 25 == 0)
        X, y = fish_image_batch_generator(part,fp=train_path + '/',full=False)
        X = X.reshape((32,224,224,3))
        convnet_fish_or_no.fit(X,y,batch_size=32,nb_epoch=1, verbose=vb, class_weight=class_wt_FoN)
        if vb == 1:
            print(epoch)
    epoch += 1


# Training VGG16
train_path = '/fp/train'

epoch = 1

while epoch <= 5:
    
    train_batch = fish_image_batch_partitioner(batch_size=32, fp=train_path, no_fish=False)
    
    for part in train_batch:
        vb = int(train_batch.index(part) % 25 == 0)
        X, y = fish_image_batch_generator(part,fp=train_path + '/', no_fish=False)
        X = X.reshape((32,224,224,3))
        vgg16_fish.fit(X,y,batch_size=32,nb_epoch=1, verbose=vb, class_weight=class_wt_fish)
        if vb == 1:
            print(epoch)
    epoch += 1
