import modules
from PIL import Image
import numpy as np 
import tensorflow as tf

# Build model
def build_model():
    #reflect_pad = modules.ReflectPad2D(2, 2, 2, 2, dtype=tf.float32, name='reflect_pad')
    #conv = modules.Conv2D(kernel_size=[3, 3], output_channel_num=8, dtype=tf.float32, name='conv')
    conv = modules.Conv2D(filters=8, kernel_size=[3, 3], padding='SAME')
    batch_norm = modules.BatchNormalization(dtype=tf.float32, name='batch_norm')
    instance_norm = modules.InstanceNormalization(dtype=tf.float32, name='instance_norm')

    inputs = tf.keras.Input(shape=(256, 256, 3), dtype=tf.float32)   
    #x = reflect_pad(inputs)
    x = conv(inputs)
    x = batch_norm(x)
    x = instance_norm(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

# Load image file
file_name = "/home/feesh/projects/cycle_gan/dataset/vangogh2photo/testA/00001.jpg"
x = Image.open(file_name).convert('RGB')
x = np.asarray(x, dtype=np.float32)
print("Shape of x :",x.shape)

# Build model
model = build_model()

# Feed forward
x = tf.expand_dims(x, 0)
x = model(x)
print("Shape of x :", x.shape)

