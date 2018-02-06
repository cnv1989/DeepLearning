import keras
from keras import layers
from keras import initializers
from keras.datasets import cifar10
from keras.models import Model

def convolution_block(X_IN, filter_size, number_of_filters, stage, block, stride=2):
    name_fmt = 'convolution_block_layer_{}' + 'stage_{}_block_{}'.format(stage, block) + '_branch_{}'
    F1, F2, F3 = number_of_filters
    
    # 1x1 convolutions to change the number of filters and reduce the dimensions of the input
    X = layers.Conv2D(F1, (1, 1), 
                      strides=(stride, stride),
                      padding='same',
                      kernel_initializer=initializers.glorot_uniform(seed=0),
                      name=name_fmt.format('conv', 'a'))(X_IN)
    X = layers.BatchNormalization(axis=3, name=name_fmt.format('bn', 'a'))(X)
    X = layers.Activation('relu')(X)
    
    # filter_size x filter_size convolution to change the number of filters and change the dimensions
    X = layers.Conv2D(F2, (filter_size, filter_size), 
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer=initializers.glorot_uniform(seed=0),
                      name=name_fmt.format('conv', 'b'))(X)
    X = layers.BatchNormalization(axis=3, name=name_fmt.format('bn', 'b'))(X)
    X = layers.Activation('relu')(X)
    
    # 1 x 1 convolution to change the number of filters
    X = layers.Conv2D(F3, (1, 1), 
                      strides=(1, 1),
                      padding='valid',
                      kernel_initializer=initializers.glorot_uniform(seed=0),
                      name=name_fmt.format('conv', 'c'))(X)
    X = layers.BatchNormalization(axis=3, name=name_fmt.format('bn', 'c'))(X)
    
    X_shortcut = layers.Conv2D(F3, (1, 1), 
                               strides=(stride, stride), 
                               padding='valid', 
                               kernel_initializer=initializers.glorot_uniform(seed=0),
                               name=name_fmt.format('conv_short', '1'))(X_IN)
    X_shortcut = layers.BatchNormalization(axis=3, name=name_fmt.format('bn_short', '1'))(X_shortcut)
    
    X = layers.Add()([X, X_shortcut])
    return layers.Activation('relu')(X)

def identity_block(X_IN, filter_size, number_of_filters, stage, block):
    name_fmt = 'identity_block_layer_{}' + 'stage_{}_block_{}'.format(stage, block) + '_branch_{}'
    F1, F2, F3 = number_of_filters
    
    X = layers.Conv2D(F1, (1, 1), strides=(1, 1), padding='valid',
                      kernel_initializer=initializers.glorot_uniform(seed=0),
                      name=name_fmt.format('conv', 'a'))(X_IN)
    X = layers.BatchNormalization(axis=3, name=name_fmt.format('bn', 'a'))(X)
    X = layers.Activation('relu')(X)
    
    X = layers.Conv2D(F2, (filter_size, filter_size), strides=(1, 1), padding='same',
                      kernel_initializer=initializers.glorot_uniform(seed=0),
                      name=name_fmt.format('conv', 'b'))(X)
    X = layers.BatchNormalization(axis=3, name=name_fmt.format('bn', 'b'))(X)
    X = layers.Activation('relu')(X)
    
    X = layers.Conv2D(F3, (1, 1), strides=(1, 1), padding='valid',
                      kernel_initializer=initializers.glorot_uniform(seed=0),
                      name=name_fmt.format('conv', 'c'))(X)
    X = layers.BatchNormalization(axis=3, name=name_fmt.format('bn', 'c'))(X)
    
    X = layers.Add()([X, X_IN])
    return layers.Activation('relu')(X)
    
def residual_stage(X_IN, filter_size, number_of_filters, stage, number_of_blocks=1):
    
    X = convolution_block(X_IN, filter_size, number_of_filters, stage, 0)
    number_of_blocks -= 1
    
    for i in range(number_of_blocks):
        X = identity_block(X, filter_size, number_of_filters, stage, i + 1)
    
    return X

def Resnet50(input_shape, classes):
    X_input = layers.Input(input_shape)
    X = layers.ZeroPadding2D((3, 3))(X_input)
    X = layers.Conv2D(64, (7, 7), strides=(2, 2), 
                      padding='same', 
                      kernel_initializer=initializers.glorot_uniform(seed=0),
                      name='conv_stage_1')(X)
    X = layers.BatchNormalization(axis=3, name='bn_stage_1')(X)
    print(X)
    X = layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
    print(X)
    
    X = residual_stage(X, 3, [64, 64, 256], 2, number_of_blocks=3)
    print(X)
    X = residual_stage(X, 3, [128, 128, 512], 3, number_of_blocks=4)
    print(X)
    X = residual_stage(X, 3, [256, 256, 1024], 4, number_of_blocks=3)
    print(X)
    
    X = layers.AveragePooling2D()(X)
    print(X)
    
    X = layers.Flatten()(X)
    X = layers.Dense(classes, activation='softmax',
                     kernel_initializer=initializers.glorot_uniform(seed=0),
                     name='fully_connected_5')(X)
    return Model(inputs=X_input, outputs=X, name='Resnet50')
    
    
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    resnet50 = Resnet50((32, 32, 3), 10)
    resnet50.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    for i in range(5):
        resnet50.fit(x_train, y_train, epochs = 10, batch_size = 128)
        # serialize model to JSON
        model_json = resnet50.to_json()
        with open("resnet50.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        resnet50.save_weights("resnet50.h5")
        print("Saved model to disk")
