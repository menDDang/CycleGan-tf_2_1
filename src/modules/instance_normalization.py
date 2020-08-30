import tensorflow as tf


class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, momentum=0.9, epsilon=1e-7, use_bias=True, **kwargs):
        # Check arguments
        assert( 0 < momentum < 1)
        assert( 0 < epsilon < 1)

        super(InstanceNormalization, self).__init__(**kwargs)

        # Set configurations
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_bias = use_bias

    def build(self, input_shapes):
        """
        input_shapes : [batch_size, input_height, input_width, input_channel_num]
        """
        # Check arguments
        assert(len(input_shapes) == 4)  # Assume that use only 4 dimensional tensor in this layer

        # Set initial statistics
        statistics_shape = [input_shapes[3]]
        self.mean = tf.Variable(tf.zeros(shape=statistics_shape, dtype=tf.float32), name='mean')
        self.var = tf.Variable(tf.zeros(shape=statistics_shape, dtype=tf.float32), name='variance')

        # Set weights
        weights_shape = [input_shapes[1], input_shapes[2], input_shapes[3]]
        self.gamma = self.add_weight(shape=weights_shape, initializer=tf.ones_initializer, dtype=self.dtype, name='gamma')
        if self.use_bias:
            self.beta = self.add_weight(shape=weights_shape, initializer=tf.zeros_initializer, dtype=self.dtype, name='beta')

        self.built = True

    def call(self, x, training=True):
        if training:
            # Calculate mean & variance through channel dimension
            mean = tf.reduce_mean(x, axis=[1, 2])
            var = tf.reduce_mean((x - mean) ** 2, axis=[1, 2])

            # Update mean & variance
            self.mean = self.momentum * self.mean + (1 - self.momentum) * tf.reduce_mean(mean, axis=0)
            self.var = self.momentum * self.var + (1 - self.momentum) * tf.reduce_mean(var, axis=0)
        else:
            mean = self.mean
            var = self.var

        # Normalize x
        x_hat = (x - mean) / tf.sqrt(var + self.epsilon)

        # Scaling
        y = self.gamma * x_hat + self.beta
        
        return y