from PIL import Image
import numpy as np 
import tensorflow as tf

# Load image file
file_name = "/home/feesh/projects/cycle_gan/dataset/vangogh2photo/testA/00001.jpg"
x = Image.open(file_name).convert('RGB')
x = np.asarray(x, dtype=np.float32)
print("Shape of x :",x.shape)

# Expand batch dimension
x = tf.expand_dims(x, 0)
print("Shape of 4d tensor x :", x.shape)

# Apply reflection padding
pad_size = 2
paddings = [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]
paddings = np.array(paddings, dtype=np.int32)
x = tf.pad(x, paddings, mode='REFLECT')
print("Shape of padded tensor x :", x.shape)


def create_variable(shape, initializer=tf.random.uniform):
    return tf.Variable(initializer(shape, dtype=tf.float32))

# Create filters for 2D Convolution
filter_height = 3
filter_width = 3
input_channel_num = x.shape[3]
output_channel_num = 8
filters = create_variable([filter_height, filter_width, input_channel_num, output_channel_num])

# Conv operation
x = tf.nn.conv2d(x, filters, strides=1, padding='VALID', data_format='NHWC')
print("Shape of x :", x.shape)