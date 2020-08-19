def conv(self, input, filters, kernel_size=3, stride=1, activation=tf.nn.relu):
"""卷积函数

Args:
    input: 卷积层的输入
    filters: 卷积核数
    kernel_size: 卷积核的尺寸，默认是 3x3 的卷积
    stride: 步长，默认为 1
    activation: 激活函数，默认是 relu

 Returns:
    返回经过卷积层的 feature map
"""
    conved = tf.keras.layers.Conv2D(filters=filters, 
        kernel_size=kernel_size, strides=[stride, stride], 
        padding='same', activation=None)(input)
    conved = tf.keras.layers.BatchNormalization()(conved)
    conved = activation(conved)
    
    return conved


def pooling(self, input, pool_size=2, stride=2):
"""pooling 层，用来下采样

Args:
    input: 卷积层的输入
    pool_size: pooling 的尺寸
    stride: 步长，默认为 2

 Returns:
    返回经过下采样的的 feature map
"""
    return tf.keras.layers.MaxPool2D(pool_size=(pool_size, pool_size), 
        strides=stride)(input)


def up_conv(self, input, filters, activation=tf.nn.relu):
"""上采样

Args:
    input: 卷积层的输入
    filters: 卷积核数
    activation: 激活函数，默认是 relu

 Returns:
    返回经过卷积层的 feature map
"""
    conved = tf.keras.layers.Conv2DTranspose(filters = num_outputs, kernel_size=2, 
        strides=(2, 2), padding = self.padding, activation = None)(input)
    conved = tf.keras.layers.BatchNormalization()(tmp)
    conved = activation(conved)
    return conved


def copy_and_crop(self, source, target):
"""concat 操作，目的是将前面学习到的内容可以更好的保留到后面

Args:
    input: 卷积层的输入
    filters: 卷积核数
    activation: 激活函数，默认是 relu

 Returns:
    返回经过 concat 的 feature map
"""
    conved = tf.concat([source, target], -1)
    return conved