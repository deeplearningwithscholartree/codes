class DataSet(object):
    """数据集定义
    """
    CATEGORY = (
        "background",
        "person"
    )

    def __init__(self, images_original, images_segmented):
    """初始化方法

    Args:
        images_original: 原图数据
        images_segmented: Mask 数据
    """
        assert len(images_original) == len(images_segmented), "images and labels must have same length."
        self._images_original = images_original
        self._images_segmented = images_segmented

    @property
    def images_original(self):
        return self._images_original

    @property
    def images_segmented(self):
        return self._images_segmented

    @property
    def length(self):
        return len(self._images_original)

    @staticmethod
    def length_category():
        return len(DataSet.CATEGORY)

    def shuffle(self):
        idx = np.arange(self._images_original.shape[0])
        np.random.shuffle(idx)
        self._images_original, self._images_segmented = self._images_original[idx], self._images_segmented[idx]

    def transpose_by_color(self):
        self._images_original = self._images_original.transpose(0, 3, 1, 2)
        self._images_segmented = self._images_segmented.transpose(0, 3, 1, 2)

    def perm(self, start, end):
        end = min(end, len(self._images_original))
        return DataSet(self._images_original[start:end], self._images_segmented[start:end],
                       self._augmenter)

    def __call__(self, batch_size=20, shuffle=True):
        """
        `A generator which yields a batch. The batch is shuffled as default.
        用于迭代，每次返回一个 batch_size 的数据
        Args:
            batch_size: batch size
            shuffle: 是否 shuffle
        Yields:
            batch (ndarray[][][]): A batch data.
        """

        if batch_size < 1:
            raise ValueError("batch_size must be more than 1.")
        if shuffle:
            self.shuffle()

        for start in range(0, self.length, batch_size):
            batch = self.perm(start, start+batch_size)
            yield batch