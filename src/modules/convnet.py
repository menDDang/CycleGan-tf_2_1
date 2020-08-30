import tensorflow as tf


class Conv2D(tf.keras.layers.Layer):
    def __init__(self, 
                filters, 
                kernel_size, 
                strides=1, 
                pad_type='ZEROS',
                padding='SAME',
                initializer=tf.random.uniform, 
                **kwargs
                ):
        # Check argument
        assert(filters > 0)
        assert(len(kernel_size) == 2)
        assert(strides >= 1)
        assert((pad_type == 'ZEROS' or pad_type == 'REFLECT' or pad_type == 'SYMETRIC'))
        assert((padding == 'SAME' or 'NONE'))

        super(Conv2D, self).__init__(**kwargs)

        # Conviguration
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pad_type = pad_type
        self.padding = padding
        self.initializer = initializer

    def build(self, input_shapes):
        """
        input_shapes : [batch_size, height, width, input_channel_num]
        """
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
            initializer=self.initializer,
            dtype=self.dtype,
            name='bias'
        )

        # Set paddings
        if self.padding == 'SAME':
            # IH = OH = (IH + 2*PH - KH) / SH + 1
            # So, we have PH = ((IH - 1) * SH + KH - IH) / 2 
            pad_h = int(((input_shapes[1] - 1) * self.strides + self.kernel_size[0] - input_shapes[1]) / 2) + 1
            pad_w = int(((input_shapes[2] - 1) * self.strides + self.kernel_size[1] - input_shapes[2]) / 2) + 1
            self.padding_size = [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]]

        self.built = True

    def call(self, x):
        # Apply padding
        if self.padding == 'SAME':
            if self.pad_type == 'ZEROS':
                x = tf.pad(x, self.padding_size, mode='CONSTANT', constant_values=0, name='padding')
            else:
                x = tf.pad(x, self.padding_size, mode=self.pad_type, name='padding')
        
        # Convolution
        x = tf.nn.conv2d(x, self.kernel, strides=self.strides, padding='VALID', data_format='NHWC', name='conv_out') + self.bias
        return x

