import tensorflow as tf
import math

class Conv2D(tf.keras.layers.Layer):
    """
        OH = floor{ (IH + 2 * PH - KH) / SH + 1 }
    """
    def __init__(self, 
                filters, 
                kernel_size, 
                strides=1, 
                padding='VALID',
                pad_type='ZEROS',
                initializer=tf.random.uniform, 
                **kwargs
                ):
        # Check argument
        assert(filters > 0)
        assert(len(kernel_size) == 2)
        assert(strides >= 1)
        assert(padding in ['VALID', 'SAME'])
        assert(pad_type in ['ZEROS', 'REFLECT', 'SYMMETRIC'])

        super(Conv2D, self).__init__(**kwargs)

        # Set attributes
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.pad_type = pad_type
        self.initializer = initializer

    def build(self, input_shapes):
        """
        input_shapes : [batch_size, height, width, input_channel_num]
        """
        # Get shape of inputs
        _, input_height, input_width, _ = input_shapes

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
        if self.padding == 'SAME':
            # OH := floor{ (IH + (PH_left + PH_right) - KH) / SH + 1 }
            # Sincce OH == IH, we have
            # PH_left + PH_right = (IH - 1) * SH - IH + KH
            pad_height = math.floor((input_height - 1) * self.strides + - input_height + self.kernel_size[0])
            if pad_height % 2 == 0:
                ph_left = int(pad_height / 2)
                ph_right = int(pad_height / 2)
            else:
                ph_left = int(pad_height / 2)
                ph_right = int(pad_height / 2) + 1
            pad_width = math.floor((input_width - 1) * self.strides - input_width + self.kernel_size[1])
            if pad_width % 2 == 0:
                pw_left = int(pad_width / 2)
                pw_right = int(pad_width / 2)
            else:
                pw_left = int(pad_width / 2)
                pw_right = int(pad_width / 2) + 1
            self.padding_size = [[0, 0], [ph_left, ph_right], [pw_left, pw_right], [0, 0]]
        
        else:
            # (IH + (PH_left + PH_right) - KH) % SH = 0
            # -> (IH + (PH_left + PH_right) - KH) = SH * k for some positive integer k
            # -> PH_left + PH_right = SH * k + KH - IH
            k = 1
            while self.strides * k + self.kernel_size[0] - input_height <= 0:
                k += 1
            ph = self.strides * k + self.kernel_size[0] - input_height
            if ph % 2 == 0:
                ph_left = int(ph / 2)
                ph_right = int(ph / 2)
            else:
                ph_left = int(ph / 2)
                ph_right = int(ph / 2) + 1
            
            k = 1
            while self.strides * k + self.kernel_size[1] - input_width <= 0:
                k += 1
            pw = self.strides * k + self.kernel_size[1] - input_width
            if pw % 2 == 0:
                pw_left = int(pw / 2)
                pw_right = int(pw / 2)
            else:
                pw_left = int(pw / 2)
                pw_right = int(pw / 2) + 1
            self.padding_size = [[0, 0], [ph_left, ph_right], [pw_left, pw_right], [0, 0]]

        self.built = True

    def call(self, x):
        # Pad
        if self.pad_type == 'ZEROS':
            x = tf.pad(x, self.padding_size, mode='CONSTANT', constant_values=0, name='padding')
        else:
            x = tf.pad(x, self.padding_size, mode=self.pad_type, name='padding')
       
        # Convolution
        x = tf.nn.conv2d(x, self.kernel, strides=self.strides, padding=[[0, 0], [0, 0], [0, 0], [0, 0]], data_format='NHWC', name='conv_out')
        
        # Add bias
        x += self.bias

        return x


class Upsampling2D(tf.keras.layers.Layer):
    '''
    Note that
        OH = (IH - 1) * SH - 2 * PH + KH
    So,
        PH_left + PH_right = OH - (IH - 1) * SH - KH
    '''
    def __init__(self, filters, kernel_size, output_shape, pad_type='ZEROS', strides=1, initializer=tf.random.uniform, **kwargs):
        # Check arguments
        assert(filters > 0)
        assert(len(kernel_size) == 2)
        assert(len(output_shape) == 2)
        assert(pad_type in ['ZEROS', 'REFLECT', 'SYMMETRIC'])
        assert(strides >= 1)
        
        super(Upsampling2D, self).__init__(**kwargs)

        # Set attributes
        self.filters = filters
        self.kernel_size = kernel_size
        self.output_height = output_shape[0]
        self.output_width = output_shape[1]
        self.pad_type = pad_type
        self.strides = strides
        self.initializer = initializer

    def build(self, input_shapes):
        # Get shape of inputs
        batch_size, input_height, input_width, input_channel_num = input_shapes  # [batch, height, width, channels]

        # Set padding size
        ph = self.output_height - (input_height - 1) * self.strides - self.kernel_size[0]
        ph_left = int(ph / 2)
        ph_right = int(ph - ph_left)
        pw = self.output_width - (input_width - 1) * self.strides - self.kernel_size[1]
        pw_left = int(pw / 2)
        pw_right = int(pw - pw_left)
        self.padding_size = [[0, 0], [ph_left, ph_right], [pw_left, pw_right], [0, 0]]
        self._output_shape = [batch_size, self.output_height, self.output_width, self.filters]

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

        self.built = True

    def call(self, x):
        # Padding
        if self.pad_type == 'ZEROS':
            x = tf.pad(x, self.padding_size, mode='CONSTANT', constant_values=0, name='padding')
        else:
            x = tf.pad(x, self.padding_size, mode=self.pad_type, name='padding')

        # Transposed convolution 2d
        x = tf.nn.conv2d_transpose(x, self.kernel, output_shape=self._output_shape, strides=self.strides, 
            padding='VALID', data_format='NHWC', name='conv_transpose_out')
            
        x += self.bias

        return x


if __name__ == "__main__":
    x = tf.zeros(shape=[100, 28, 28, 3], dtype=tf.float32)

    down_sampling = Conv2D(16, [8, 8], padding='SAME', dtype=tf.float32)
    up_sampling = Upsampling2D(16, [8, 8], output_shape=[35, 35], pad_type='ZEROS', dtype=tf.float32)

    x = down_sampling(x)
    print(x.shape)
    x = up_sampling(x)
    print(x.shape)