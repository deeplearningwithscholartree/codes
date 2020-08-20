class Loader():
    """数据加载
    """
    def __init__(self, dir_original, dir_segmented, init_size=(256, 256), one_hot=True):
    """初始化方法

    Args:
        dir_original: 原图目录
        dir_segmented: Mask 目录
        init_size: 输出尺寸大小，默认是 256x256
        one_hot: 是否进行 one-hot 编码
    """
        #调用 Loader 的 import_data 方法读入数据
        self._data = Loader.import_data(dir_original, dir_segmented, init_size, one_hot)


    def load_train_test(self, train_rate=0.85, shuffle=True):
        """
        讲数据按比例拆分为训练集合测试集
        Args:
            train_rate: 训练数据所占的比例
            shuffle: 是否 shuffle
        Returns:
            训练集、测试集
        """
        if train_rate < 0.0 or train_rate > 1.0:
            raise ValueError("train_rate must be from 0.0 to 1.0.")

        if shuffle:
            self._data.shuffle()

        train_size = int(self._data.images_original.shape[0] * train_rate)
        data_size = int(len(self._data.images_original))
        train_set = self._data.perm(0, train_size)
        test_set = self._data.perm(train_size, data_size)

        return train_set, test_set

    @staticmethod
    def import_data(dir_original, dir_segmented, init_size=None, one_hot=True):
        """
        生成训练数据
        Args:
            dir_original: 原图目录
            dir_segmented: Mask 目录
            init_size: 输出尺寸大小
            one_hot: 是否进行 one-hot 编码
        Returns:
            返回 Dataset 类型的数据
        """
        paths_original, paths_segmented = Loader.generate_paths(dir_original, dir_segmented)
        images_original, images_segmented = Loader.extract_images(paths_original, paths_segmented, init_size, one_hot)

        return DataSet(images_original, images_segmented)

    @staticmethod
    def generate_paths(dir_original, dir_segmented):
        """
        生成训练数据的路径
        Args:
            dir_original: 原图目录
            dir_segmented: Mask 目录
        Returns:
            原图的目录列表、Mask 的目录列表
        """
        paths_original = glob.glob(dir_original + "/*")
        paths_segmented = glob.glob(dir_segmented + "/*")
        if len(paths_original) == 0 or len(paths_segmented) == 0:
            raise FileNotFoundError("Could not load images.")
        filenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], paths_segmented))
        paths_original = list(map(lambda filename: dir_original + "/" + filename + ".jpg", filenames))

        return paths_original, paths_segmented

    @staticmethod
    def extract_images(paths_original, paths_segmented, init_size, one_hot):
        """
        从图片中抽取数据
        Args:
            paths_original: 原图图片目录列表
            paths_segmented: Mask 图片目录列表
            init_size: 图片大小
            one_hot: 是否 one-hot 编码
        Returns:
            原图的 numpy 数组，mask 的 numpy 数组
        """
        images_original, images_segmented = [], []


        # Load images from directory_path using generator
        print("Loading original images")
        for image in Loader.image_generator(paths_original, init_size, antialias=True):
            images_original.append(image)
            if len(images_original) % 200 == 0:
                print(".")
        print(" Completed")
        print("Loading segmented images")
        for image in Loader.image_generator(paths_segmented, init_size, normalization=False):
            images_segmented.append(image)
            if len(images_segmented) % 200 == 0:
                print(".")
        print(" Completed")
        assert len(images_original) == len(images_segmented)

        # Cast to ndarray
        images_original = np.asarray(images_original, dtype=np.float32)
        images_segmented = np.asarray(images_segmented, dtype=np.uint8)

        images_segmented = np.where((images_segmented != 15) & (images_segmented != 255), 0, images_segmented)
        images_segmented = np.where(images_segmented == 15, 1, images_segmented)
        images_segmented = np.where(images_segmented == 255, len(DataSet.CATEGORY)-1, images_segmented)

        if one_hot:
            print("Casting to one-hot encoding... ")
            identity = np.identity(len(DataSet.CATEGORY), dtype=np.uint8)
            images_segmented = identity[images_segmented]
            print("Done")
        else:
            pass

        return images_original, images_segmented


    @staticmethod
    def cast_to_index(ndarray):
        return np.argmax(ndarray, axis=2)

    @staticmethod
    def cast_to_onehot(ndarray):
        identity = np.identity(len(DataSet.CATEGORY), dtype=np.uint8)
        return identity[ndarray]

    @staticmethod
    def image_generator(file_paths, init_size=None, normalization=True, antialias=False):
        """
        使用 pillow 读取图片，并将图片 resize 到指定大小
        Args:
            file_paths: 文件路径
            init_size: resize 大小
            normalization: 是否正规化，默认为是
            antialias: 是否为 mask
        Returns:
            原图的 numpy 数组，mask 的 numpy 数组
        """
        for file_path in file_paths:
            if file_path.endswith(".png") or file_path.endswith(".jpg"):
                # open a image
                image = Image.open(file_path)
                # to square
                image = Loader.crop_to_square(image)
                # resize by init_size
                if init_size is not None and init_size != image.size:
                    if antialias:
                        image = image.resize(init_size, Image.ANTIALIAS)
                    else:
                        image = image.resize(init_size)
                # delete alpha channel
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                image = np.asarray(image)
                if normalization:
                    image = image / 255.0
                yield image