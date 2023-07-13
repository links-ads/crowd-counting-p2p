from typing import Union, Sequence
import numpy as np
import cv2
from albumentations.augmentations.geometric.resize import SmallestMaxSize, LongestMaxSize, F
from albumentations.core.transforms_interface import KeypointInternalType, DualTransform

class ConditionalSmallestMaxSize(SmallestMaxSize):
    def __init__(
        self,
        max_size: Union[int, Sequence[int]] = 128,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1,
    ):
        super(ConditionalSmallestMaxSize, self).__init__(max_size=max_size, interpolation=interpolation, always_apply=always_apply, p=p)
        self.max_size = max_size

    def apply(self, img: np.ndarray, max_size: int = 128, interpolation: int = cv2.INTER_LINEAR, **params) -> np.ndarray:
        if np.min(img.shape[:-1]) < max_size:
            return super(ConditionalSmallestMaxSize, self).apply(img, max_size, interpolation, **params)
        return img

    def apply_to_keypoint(self, keypoint: KeypointInternalType, max_size: int = 128, **params) -> KeypointInternalType:
        height = params["rows"]
        width = params["cols"]

        if min(height, width) < max_size:
            return super(ConditionalSmallestMaxSize, self).apply_to_keypoint(keypoint, max_size, **params)
        return keypoint
    
class ConditionalLongestMaxSize(LongestMaxSize):
    def __init__(
        self,
        max_size: Union[int, Sequence[int]] = 1920,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1,
    ):
        super(ConditionalLongestMaxSize, self).__init__(max_size=max_size, interpolation=interpolation, always_apply=always_apply, p=p)
        self.max_size = max_size

    def apply(self, img: np.ndarray, max_size: int = 128, interpolation: int = cv2.INTER_LINEAR, **params) -> np.ndarray:
        if np.max(img.shape[:-1]) > max_size:
            return super(ConditionalLongestMaxSize, self).apply(img, max_size, interpolation, **params)
        return img

    def apply_to_keypoint(self, keypoint: KeypointInternalType, max_size: int = 128, **params) -> KeypointInternalType:
        height = params["rows"]
        width = params["cols"]

        if max(height, width) > max_size:
            return super(ConditionalLongestMaxSize, self).apply_to_keypoint(keypoint, max_size, **params)
        return keypoint
    
class ResizeMultiple(DualTransform):
    """Resize the input to make it multiple of the given height and width.

    Args:
        height (int): desired height of the output.
        width (int): desired width of the output.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super(ResizeMultiple, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        new_height =  img.shape[0] // self.height * self.height
        new_width = img.shape[1] // self.width * self.width
        return F.resize(img, height=new_height, width=new_width, interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        height = params["rows"]
        width = params["cols"]
        new_height =  height // self.height * self.height
        new_width = width // self.width * self.width
        scale_x = new_width / width
        scale_y = new_height / height
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        return ("height", "width", "interpolation")