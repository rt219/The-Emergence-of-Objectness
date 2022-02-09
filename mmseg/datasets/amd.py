import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class AMDDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('background', 'foreground')

    PALETTE = [[0, 0, 0], [128, 0, 0]]

    def __init__(self, split, **kwargs):
        super(AMDDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.jpg', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
