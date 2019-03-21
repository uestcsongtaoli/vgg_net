"""
AlexNet Keras implementation

"""

# Import necessary libs

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import math
from keras import optimizers
from vgg_model import VGG_16
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

input_shape = (224, 224, 3)
num_classes = 10

vgg16_net = VGG_16(input_shape=input_shape, num_classes=num_classes)
parallel_model = multi_gpu_model(vgg16_net, gpus=2)

epochs = 200
model_name = "VGG16-1"
train_dir = r'/home/lst/datasets/cifar-10-images_train/'
test_dir = r'/home/lst/datasets/cifar-10-images_test/'
batch_size = 64
target_weight_height = (224, 224)

adadelta = optimizers.adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
parallel_model.compile(loss=['categorical_crossentropy'],
                       optimizer=adadelta,
                       metrics=["accuracy"])
# callbacks
tensorboard = TensorBoard(log_dir=f'./logs/{model_name}', histogram_freq=0,
                          write_graph=True, write_images=False)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1)
mc = ModelCheckpoint(f"{model_name}.h5", monitor='vac_acc', mode="max", verbose=1, save_best_only=True)
cb_list = [tensorboard, early_stopping, mc]

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_weight_height,
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_weight_height,
    batch_size=batch_size,
    class_mode='categorical')

num_train_samples = train_generator.samples
num_val_samples = validation_generator.samples

parallel_model.fit_generator(train_generator,
                             validation_data=validation_generator,
                             steps_per_epoch=math.ceil(num_train_samples / batch_size),
                             validation_steps=math.ceil(num_val_samples / batch_size),
                             epochs=epochs,
                             callbacks=cb_list, )
