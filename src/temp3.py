import model
from PIL import Image
import numpy as np 
import tensorflow as tf

# Load image file
file_name = "/home/feesh/projects/cycle_gan/dataset/vangogh2photo/testA/00001.jpg"
x = Image.open(file_name).convert('RGB')
x = np.asarray(x, dtype=np.float32)
print("Shape of x :",x.shape)

# Build model
generator = model.ResnetGenerator(8, dtype=tf.float32, name='generator')


# Feed forward
x = tf.expand_dims(x, 0)
x = generator(x)
print("Shape of x :", x.shape)



for var in generator.trainable_variables:
    print(var.name)

