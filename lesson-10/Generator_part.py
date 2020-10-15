def generator(z, batch_size, z_dim):
    with tf.variable_scope('generator') as scope:
        # 首先我们定义第一层的filter数量
        g_dim = 64

        # 接下来我们需要定义各个层的filter尺寸，一般来说我们较为常见的是逐步增加的方式，也就是每一层和相邻层是1/2倍的关系，当然你也可以设置为别的数字。

        # 28是因为最终的图像尺寸是28。
        s = 28
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

        # 回顾一下CNN的最后一个卷基层和第一个全连接层的关系，是不是将特征图拍平变成向量的形式。这里在generator正好是相反的，要将向量转化成特征图的形式。这里使用reshape。
        h0 = tf.reshape(z, [batch_size, s16+1, s16+1, 25])
        h0 = tf.nn.relu(h0)
        # 到目前为止，特征图的尺寸为batch_size x 2 x 2 x 25。

        # 第一个反卷基层。将特征图的尺寸从2x2变成3x3，数量从25变成256。一定不要忘了加入batch norm层和激活函数。
        output1_shape = [batch_size, s8, s8, g_dim*4]
        W_conv1 = tf.get_variable('g_wconv1', [5, 5, output1_shape[-1], int(h0.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv1 = tf.get_variable('g_bconv1', [output1_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv1 = tf.nn.conv2d_transpose(h0, W_conv1, output_shape=output1_shape, strides=[1, 2, 2, 1], padding='SAME')
        H_conv1 = tf.contrib.layers.batch_norm(inputs = H_conv1, center=True, scale=True, is_training=True, scope="g_bn1")
        H_conv1 = tf.nn.relu(H_conv1)

        # 第二个反卷基层。将特征图的尺寸从3x3变成6x6，数量从256变成128。同样的，不要忘了加入batch norm层和激活函数。
        output2_shape = [batch_size, s4 - 1, s4 - 1, g_dim*2]
        W_conv2 = tf.get_variable('g_wconv2', [5, 5, output2_shape[-1], int(H_conv1.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv2 = tf.get_variable('g_bconv2', [output2_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv2 = tf.nn.conv2d_transpose(H_conv1, W_conv2, output_shape=output2_shape, strides=[1, 2, 2, 1], padding='SAME')
        H_conv2 = tf.contrib.layers.batch_norm(inputs = H_conv2, center=True, scale=True, is_training=True, scope="g_bn2")
        H_conv2 = tf.nn.relu(H_conv2)

        # 第三个反卷基层。将特征图的尺寸从6x6变成12x12，数量从128变成64。老规矩，batch norm层和激活函数加上。
        output3_shape = [batch_size, s2 - 2, s2 - 2, g_dim*1]
        W_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(H_conv2.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape, strides=[1, 2, 2, 1], padding='SAME')
        H_conv3 = tf.contrib.layers.batch_norm(inputs = H_conv3, center=True, scale=True, is_training=True, scope="g_bn3")
        H_conv3 = tf.nn.relu(H_conv3)

        # 最后一个反卷基层，实际上得到的结果就是generator利用噪声数据生成的图片（假数据）。
        output4_shape = [batch_size, s, s, 1] # 这里的1是因为我们使用的mnist数据是只有一个通道的灰度图像，所以要写1。
        W_conv4 = tf.get_variable('g_wconv4', [5, 5, output4_shape[-1], int(H_conv3.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv4 = tf.get_variable('g_bconv4', [output4_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv4 = tf.nn.conv2d_transpose(H_conv3, W_conv4, output_shape=output4_shape, strides=[1, 2, 2, 1], padding='VALID')
        H_conv4 = tf.nn.tanh(H_conv4)

    return H_conv4
