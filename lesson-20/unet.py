import tensorflow as tf
from utils.layers import *

'''
我们定一个输入为(256, 256, 3)， 输出为(256, 256，1)的 U-Net。
最终输出的 feature map 中，1 为前景(人像)，0 为背景
'''
class U_Net():
    """
    定义 U-Net 的网络结构
    """

    def __init__(self, input_shape=(256, 256，3), output_shape=(256, 256)):
    """U-Net 的初始化方法

    Args:
        input_shape: 网络输入的尺寸，默认输入的尺寸是 256x256
        output_shape: 网络的输出尺寸，默认输出的尺寸是 256x256
    """
        # 使用 Adam 优化
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        # 预测的 mask
        self.predict = self.model(input_shape)
        # GT 的 mask
        self.labels = tf.placeholder(tf.float32, [None]+output_shape)
        # 计算 Loss
        self.loss = tf.reduce_mean(tf.square(self.labels - self.predict))
        self.training = self.optimizer.minimize(self.loss)


    def model(self, input_shape):
        # 使用 placeholder 定义一个输入的占位符，None 代表传入的 batch 的大小
        self.inputs = tf.placeholder(tf.float32, [None] + input_shape)
        # Encoder 过程开始，也就是 U 的左侧部分
        # 卷积层 conv1_1， 输入是训练图片图片，卷积核的数目为 64，由于我们在 conf 方法中定义的 padding 参数为 same
        # 所以我们的卷积层是不会改变 feature map 的宽与高的。这一点与论文中少有不一样。
        # conv1_1 会输出一个(256, 256, 64)的 feature map
        conv1_1 = conv(self.inputs, 64)
        # conv1_2 会输出一个(256, 256, 64)的 feature map
        conv1_2 = conv(conv1_1, 64)
        # 使用 pooling 完成一个下采样，
        # pooling_1 会输出一个(128, 128, 64)的 feature map
        pooling_1 = pooling(conv1_2)
        conv2_1 = conv(pooling_1, 128)
        conv2_2 = conv(conv2_1, 128)
        # pooling_2 会输出一个(64, 64, 128)的 feature map
        pooling_2 = pooling(conv2_2)
        conv3_1 = conv(pooling_2, 256)
        conv3_2 = conv(conv3_1, 256)
        # pooling_3 会输出一个(32, 32, 256)的 feature map
        pooling_3 = pooling(conv3_2)
        conv4_1 = conv(pooling_3, 512)
        conv4_2 = conv(conv4_1, 512)
        # pooling_4 会输出一个(16, 16, 512)的 feature map
        pooling_4 = conv(conv4_2)
        conv5_1 = conv(pooling_4, 1024)
        # conv5_2 会输出一个(16, 16, 1024)的 feature map
        conv5_2 = conv(conv5_1, 1024)
        # 到这里 Encoder 过程就结束了，接下来要开始 Decoder 过程了
        # conv_up_4_1 会将 feature map 的宽高由 16x16 扩大到 32x32，然后与 conv4_2 合并
        conv_up_4_1= up_conv(conv5_2, 512)
        conv_up_4_1= copy_and_crop(conv4_2, conv_up_4_1)
        conv_up_4_2= conv(conv_up_4_1, 512)
        conv_up_4_3= conv(conv_up_4_2, 512)
        # conv_up_3_1 会将 feature map 的宽高由 32x32 扩大到 64x64，然后与 conv3_2 合并
        conv_up_3_1= up_conv(conv_up_4_3, 256)
        conv_up_3_1= copy_and_crop(conv3_2, conv_up_3_1)
        conv_up_3_2= conv(conv_up_3_1, 256)
        conv_up_3_3= conv(conv_up_3_2, 256)
        # conv_up_2_1 会将 feature map 的宽高由 64x64 扩大到 128x128，然后与 conv2_2 合并
        conv_up_2_1= up_conv(conv_up_3_3, 128)
        conv_up_2_1= copy_and_crop(conv2_2, conv_up_2_1)
        conv_up_2_2= conv(conv_up_2_1, 128)
        conv_up_2_3= conv(conv_up_2_2, 128)
        # conv_up_1_1 会将 feature map 的宽高由 128x128 扩大到 256x256，然后与 conv1_2 合并
        conv_up_1_1= up_conv(conv_up_2_3, 64)
        conv_up_1_1= copy_and_crop(conv1_2, conv_up_1_1)
        conv_up_1_2= conv(conv_up_1_1, 64)
        conv_up_1_3= conv(conv_up_1_2, 64)
        # 最后用一个 1x1 的卷积 + sigmoid 激活函数获得最终的输出
        outputs = conv(conv_up_1_3, 1, kernel_size=1, activation=tf.keras.activations.sigmoid)
        return outputs