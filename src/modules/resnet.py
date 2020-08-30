import tensorflow as tf
from .convnet import Conv2D


class ResidualConv2D(tf.keras.layers.Layer):
    def __init__(self, 
                kernel_size, 
                strides=1, 
                pad_type='ZEROS',
                padding='SAME',
                initializer=tf.random.uniform, 
                **kwargs
                ):
        super(ResidualConv2D, self).__init__(**kwargs)
        
        self.kernel_size = kernel_size
        self.strides = strides
        self.pad_type = pad_type
        self.padding = padding
        self.initializer = initializer

    def build(self, input_shapes):
        """
        input_shapes : [batch_size, height, width, input_channel_num]
        """
        self.conv = Conv2D(
                filters=input_shapes[3],
                kernel_size=self.kernel_size, 
                strides=self.strides, 
                pad_type=self.pad_type,
                padding=self.padding,
                initializer=self.initializer, 
                dtype=self.dtype,
                name='conv'
                )
        self.built = True

    def call(self, x):
        return self.conv(x) + x