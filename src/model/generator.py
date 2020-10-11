import tensorflow as tf
import modules

class ResnetGenerator(tf.keras.layers.Layer):
    def __init__(self,
                num_filters,
                norm_layer=modules.InstanceNorm2D,
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
            # reflect padding (3)
            modules.Conv2D(filters=self.num_filters, kernel_size=[7, 7], strides=2, padding='NONE')
        )
        self.down_sampling.add(modules.InstanceNorm2D())
        self.down_sampling.add(tf.keras.layers.ReLU())

        num_downsampling = 2
        for i in range(num_downsampling):
            mult = 2 ** i
            # zero padding (1)
            self.down_sampling.add(
                modules.Conv2D(self.num_filters * mult, kernel_size = [3, 3], strides=2, padding='NONE')
            )
            self.down_sampling.add(modules.InstanceNorm2D())
            self.down_sampling.add(tf.keras.layers.ReLU())        

    def call(self, x, training=True):
        x = self.down_sampling(x, training=True)
        return x


if __name__ == "__main__": 
    batch_size = 1
    input_height = 256
    input_width = 256
    input_channels = 3

    generator = ResnetGenerator(64)

    x = tf.zeros(shape=[batch_size, input_height, input_width, input_channels], dtype=tf.float32)
    x = generator(x)
    print(x.shape)