import tensorflow as tf
from .conv2d import Conv2D

class ResidualConv2D(tf.keras.layers.Layer):
    def __init__(self, 
                kernel_size, 
                strides=1, 
                pad_type='ZEROS',
                initializer=tf.random.uniform, 
                **kwargs
                ):
        # Check arguments
        assert(len(kernel_size) == 2)
        assert(strides > 0)
        assert(pad_type in ['ZEROS', 'REFLECT', 'SYMMETRIC'])
        
        super(ResidualConv2D, self).__init__(**kwargs)
        
        self.kernel_size = kernel_size
        self.strides = strides
        self.pad_type = pad_type
        self.initializer = initializer

    def build(self, input_shapes):
        # Get shape of inputs
        _, _, _, input_channel_num = input_shapes  # [batch, input height, input width, input channels]

        
        # Set convolution layer
        self.conv = Conv2D(
                filters=input_channel_num,
                kernel_size=self.kernel_size, 
                strides=self.strides, 
                padding='SAME',
                pad_type=self.pad_type,
                initializer=self.initializer, 
                dtype=self.dtype,
                name='conv'
                )

        self.built = True

    def call(self, x):
        return self.conv(x) + x
        
if __name__ == "__main__":
    x = tf.zeros(shape=[100, 28, 28, 3], dtype=tf.float32)
    resnet = ResidualConv2D([8, 8], dtype=tf.float32)
    x = resnet(x)
    print(x.shape)