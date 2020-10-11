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
        assert(pad_type in ['ZEROS', 'REFLECT', 'SYMETRIC'])
        
        super(ResidualConv2D, self).__init__(**kwargs)
        
        self.kernel_size = kernel_size
        self.strides = strides
        self.pad_type = pad_type
        self.initializer = initializer

    def build(self, input_shapes):
        # Get shape of inputs
        _, input_height, input_width, input_channel_num = input_shapes

        # Set padding size
        # IH = OH = (IH + (PH_left + PH_right) - KH) / SH + 1
        # PH_left + PH_right = (IH - 1) * SH - IH + KH
        pad_height = (input_height - 1) * self.strides + - input_height + self.kernel_size[0]
        if pad_height % 2 == 0:
            ph_left = int(pad_height / 2)
            ph_right = int(pad_height / 2)
        else:
            ph_left = int(pad_height / 2)
            ph_right = int(pad_height / 2) + 1
        pad_width = (input_width - 1) * self.strides - input_width + self.kernel_size[1]
        if pad_width % 2 == 0:
            pw_left = int(pad_width / 2)
            pw_right = int(pad_width / 2)
        else:
            pw_left = int(pad_width / 2)
            pw_right = int(pad_width / 2) + 1

        self.padding_size = [[0, 0], [ph_left, ph_right], [pw_left, pw_right], [0, 0]]
        
        # Set convolution layer
        self.conv = conv2d.Conv2D(
                filters=input_channel_num,
                kernel_size=self.kernel_size, 
                strides=self.strides, 
                initializer=self.initializer, 
                dtype=self.dtype,
                name='conv'
                )

        self.built = True

    def call(self, x):
        if self.pad_type == 'ZEROS':
            padded_x = tf.pad(x, self.padding_size, mode='CONSTANT', constant_values=0, name='padding')
        else:
            padded_x = tf.pad(x, self.padding_size, mode=self.pad_type, name='padding')
        
        return self.conv(padded_x) + x

if __name__ == "__main__":
    x = tf.zeros(shape=[100, 28, 28, 3], dtype=tf.float32)
    resnet = ResidualConv2D([8, 8], dtype=tf.float32)
    x = resnet(x)
    print(x.shape)