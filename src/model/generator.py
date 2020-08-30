import tensorflow as tf
from modules import Conv2D, ResidualConv2D, InstanceNormalization


class ResnetGenerator(tf.keras.layers.Layer):
    def __init__(self,
                num_filters,
                norm_layer=InstanceNormalization,
                resnet_block_num=6,
                pad_type='REFLECT',
                **kwargs):
        """ Construct a Resnet based generator

        Parameters:
            num_filters         int     Number of filters in the last conv. layer
            norm_layer                  normlization layer
            resnet_block_num    int     Number of Resnet blocks
            padding_type        str     Name of padding layer in conv layers [ REFLECT | ZEROS ]
        """
        # Check arguments
        assert(resnet_block_num >= 0)

        super(ResnetGenerator, self).__init__(**kwargs)

        # Set configurations
        self.num_filters = num_filters
        self.norm_layer = norm_layer
        self.resnet_block_num = resnet_block_num
        self.pad_type = pad_type

        # Build network for down-sampling
        self.down_sampling = tf.keras.Sequential(name='downsampling')
        self.down_sampling.add(
            Conv2D(
                filters=self.num_filters,
                kernel_size=[7, 7],
                strides=2,
                pad_type=self.pad_type,
                padding='SAME',
                dtype=self.dtype,
                name='conv_1'
            )
        )
        
        self.down_sampling.add(self.norm_layer(dtype=self.dtype, name='norm_1'))
        self.down_sampling.add(tf.keras.layers.ReLU(name='relu_1'))
        # conv-2
        self.down_sampling.add(
            Conv2D(
                filters=self.num_filters * 2,
                kernel_size=[3, 3],
                strides=2,
                pad_type=self.pad_type,
                padding='SAME',
                dtype=self.dtype,
                name='conv_2'
            )
        )
        self.down_sampling.add(tf.keras.layers.ReLU(name='relu_2'))
        # conv-3
        self.down_sampling.add(
            Conv2D(
                filters=self.num_filters * 4,
                kernel_size=[3, 3],
                strides=2,
                pad_type=self.pad_type,
                padding='SAME',
                dtype=self.dtype,
                name='conv_3'
            )
        )
        self.down_sampling.add(tf.keras.layers.ReLU(name='relu_3'))

        
        # Build network for residual convolutional blocks
        self.resnet = tf.keras.Sequential(name='resnet')
        for i in range(self.resnet_block_num):
            self.resnet.add(
                ResidualConv2D(
                    kernel_size=[3, 3],
                    strides=2,
                    pad_type=self.pad_type,
                    padding='SAME',
                    dtype=self.dtype,
                    name='resnet_' + str(i + 1)
                )
            )
            self.resnet.add(tf.keras.layers.ReLU(name='relu_'+str(i + 1)))


        # Build network for up-sampling
        self.upsampling = tf.keras.Sequential(name='upsampling')
        

    def call(self, x, training=True):
        x = self.down_sampling(x, training=True)
        x = self.resnet(x)
        return x
