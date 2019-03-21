"""
VGG16 model
"""

from keras.layers import Input, Conv2D, MaxPool2D, ZeroPadding2D, Flatten, Dense, Dropout, BatchNormalization, \
    Activation
from keras.models import Model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def conv_block(layer, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', name=None):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               kernel_initializer="he_normal",
               name=name)(layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def VGG_16(input_shape=(224, 224, 3), num_classes=10, weight_path=None):
    # instantiate a Keras tensor
    input_layer = Input(shape=input_shape)
    # stage 1
    x = conv_block(input_layer, filters=64, kernel_size=(3, 3), name="conv1_1_64_3x3_1")
    x = conv_block(x, filters=64, kernel_size=(3, 3), name="conv1_2_64_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_1_2x2_2")(x)
    # stage 2
    x = conv_block(x, filters=128, kernel_size=(3, 3), name="conv2_1_128_3x3_1")
    x = conv_block(x, filters=128, kernel_size=(3, 3), name="conv2_2_128_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_2_2x2_2")(x)
    # stage 3
    x = conv_block(x, filters=256, kernel_size=(3, 3), name="conv3_1_256_3x3_1")
    x = conv_block(x, filters=256, kernel_size=(3, 3), name="conv3_2_256_3x3_1")
    x = conv_block(x, filters=256, kernel_size=(1, 1), name="conv3_3_256_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_3_2x2_2")(x)
    # stage 4
    x = conv_block(x, filters=512, kernel_size=(3, 3), name="conv4_1_512_3x3_1")
    x = conv_block(x, filters=512, kernel_size=(3, 3), name="conv4_2_512_3x3_1")
    x = conv_block(x, filters=512, kernel_size=(1, 1), name="conv4_3_512_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_4_2x2_2")(x)
    # stage 5
    x = conv_block(x, filters=512, kernel_size=(3, 3), name="conv5_1_512_3x3_1")
    x = conv_block(x, filters=512, kernel_size=(3, 3), name="conv5_2_512_3x3_1")
    x = conv_block(x, filters=512, kernel_size=(1, 1), name="conv5_3_512_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_5_2x2_2")(x)

    # FC layers
    # FC layer 1
    x = Flatten()(x)
    x = Dense(2048)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Activation("relu")(x)
    # FC layer 2
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Activation("relu")(x)
    # FC layer 3
    """我以为最后一层不需要dropout层"""
    x = Dense(num_classes)(x)
    x = BatchNormalization()(x)
    x = Activation("softmax")(x)

    if weight_path:
        x.load_weights(weight_path)
    model = Model(input_layer, x, name="VGG16_Net")
    model.summary()
    return model


if __name__ == "__main__":
    VGG_16()
