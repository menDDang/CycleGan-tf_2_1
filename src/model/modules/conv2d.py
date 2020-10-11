import tensorflow as tf


class Conv2D(tf.keras.layers.Layer):
    '''
        Note that this class does not apply padding
        So, user must use external padding layer
    '''
    def __init__(self, 
                filters, 
                kernel_size, 
                strides=1, 
                initializer=tf.random.uniform, 
                **kwargs
                ):
        # Check argument
        assert(filters > 0)
        assert(len(kernel_size) == 2)
        assert(strides >= 1)

        super(Conv2D, self).__init__(**kwargs)

        # Set attributes
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.initializer = initializer

    def build(self, input_shapes):
        """
        input_shapes : [batch_size, height, width, input_channel_num]
        """
        # Set kernel
        filter_shape = [self.kernel_size[0], self.kernel_size[1], input_shapes[3], self.filters]
        self.kernel = self.add_weight(
            shape=filter_shape,
            initializer=self.initializer,
            dtype=self.dtype,
            name='kernel'
        )

        # Set bias
        self.bias = self.add_weight(
            shape=[self.filters],
            initializer=tf.initializers.Zeros,
            dtype=self.dtype,
            name='bias'
        )

        # Set paddings
        self.padding = [[0, 0], [0, 0], [0, 0], [0, 0]]

        self.built = True

    def call(self, x):
        # Convolution
        x = tf.nn.conv2d(x, self.kernel, strides=self.strides, padding=self.padding, data_format='NHWC', name='conv_out')
        # Add bias
        x += self.bias

        return x


class Upsampling2D(tf.keras.layers.Layer):
    '''
        Note that this class apply 'VALID' padding
        So, user must do not use external padding layer
    '''
    def __init__(self, filters, kernel_size, output_shape, strides=1, initializer=tf.random.uniform, **kwargs):
        # Check arguments
        assert(filters > 0)
        assert(len(kernel_size) == 2)
        assert(len(output_shape) == 2)
        assert(strides >= 1)
        
        super(Upsampling2D, self).__init__(**kwargs)

        # Set attributes
        self.filters = filters
        self.kernel_size = kernel_size
        self.output_height = output_shape[0]
        self.output_width = output_shape[1]
        self.strides = strides
        self.initializer = initializer

    def build(self, input_shapes):
        # Get shape of inputs
        batch_size, _, _, input_channel_num = input_shapes  # [batch, height, width, channels]

        # Set kernel
        filter_shape = [self.kernel_size[0], self.kernel_size[1], self.filters, input_channel_num]
        self.kernel = self.add_weight(
            shape=filter_shape,
            initializer=self.initializer,
            dtype=self.dtype,
            name='kernel'
        )

        # Set bias
        self.bias = self.add_weight(
            shape=[self.filters],
            initializer=tf.initializers.Zeros,
            dtype=self.dtype,
            name='bias'
        )

        # Set output shape
        self._output_shape = [batch_size, self.output_height, self.output_width, self.filters]

        self.built = True

    def call(self, x):
        # Transposed convolution 2d
        x = tf.nn.conv2d_transpose(x, self.kernel, output_shape=self._output_shape, strides=self.strides, padding='VALID', 
            data_format='NHWC', name='conv_transpose_out')
            
        x += self.bias

        return x


if __name__ == "__main__":
    x = tf.zeros(shape=[100, 28, 28, 3], dtype=tf.float32)

    down_sampling = Conv2D(16, [8, 8], dtype=tf.float32)
    up_sampling = Upsampling2D(16, [8, 8], output_shape=[28, 28], dtype=tf.float32)

    x = down_sampling(x)
    print(x.shape)
    x = up_sampling(x)
    print(x.shape)