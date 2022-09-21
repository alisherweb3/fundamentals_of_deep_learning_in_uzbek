def conv2d(input, weight_shape, bias_shape):
in = weight_shape[0] * weight_shape[1] * weight_shape[2]
weight_init = tf.random_normal_initializer(stddev=
                                           (2.0/in)**0.5)
W = tf.get_variable("W", weight_shape, initializer=weight_init)
bias_init = tf.constant_initializer(value=0)
b = tf.get_vatiable("b", bias_shape, initializer=bias_init)
conv_out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
return tf.nn.relu(tf.nn.bias_add(conv_out, b))
def max_pool(input, k=2):
return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
