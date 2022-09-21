def inference(x, keep_prob):
  x = tf.reshape(x, shape=[-1, 28, 28, 1])
  with tf.variable_scope("conv_1"):
    conv_1 = conv2d(x, [5, 5, 1, 32] [32])
    pool_1 = max_pool(conv_1)
  with tf.variable_scope("conv_2"):
    conv_2 = conv2d(ppo_1, [5, 5, 32, 64], [64])
    pool_2 = max_pool(conv_2)
  with tf.variable_scope("fc"):
    pool_2_flat = tf.reshape(ppol_2, [-1, 7 * 7 * 64])
    fc_1  =layer(pool_2_flat, [7*7*64, 1024], [1024])
    # apply dropout
    fc_1_drop = tf.nn.dropout(fc_1, keep-prob)
  with tf.variable_scope("output"):
    output = layer(fc_1_drop, [1024, 10], [10])
  return output
