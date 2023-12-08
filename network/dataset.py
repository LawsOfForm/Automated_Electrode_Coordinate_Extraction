from skimage.transform import resize
from skimage.exposure import rescale_intensity
import numpy as np

class GreyToRGB(object):
    
    def __call__(self, volume_mask):
        volume, mask = volume_mask
        w, h, d =  volume.shape
        volume += np.abs(np.min(volume))
        volume_max = np.max(volume)
        if volume_max > 0:
            volume /= volume_max 
        ret = np.empty((w, h, d, 3), dtype=np.uint8)
        ret[:, :, :, 2] = ret[:, :, :, 1] = ret[:, :, :, 0] = volume * 255
        return ret, mask
    
class ChannelSwitchBST(object):
    
    def __call__(self, volume_mask):
        volume, mask = volume_mask
        volume = np.transpose(volume, (0, 1, 3, 2))
        return volume, mask
    
class FixChannelDimension(object):
    
    def __call__(self, volume_mask):
        volume, mask = volume_mask
        volume = np.transpose(volume, (0, 1, 3, 2))
        return volume, mask

class CropSample(object):
    def __call__(self, volume_mask):
        volume, mask = volume_mask
        
        def min_max_projection(axis: int):
            
            axis = [i for i in range(4) if i != axis]
            axis = tuple(axis)
            projection = np.max(volume, axis=axis)
            non_zero = np.nonzero(projection)
            return np.min(non_zero), np.max(non_zero) + 1 
        z_min, z_max = min_max_projection(0)
        y_min, y_max = min_max_projection(1)
        x_min, x_max = min_max_projection(2)
        
        return (
            volume[z_min:z_max, y_min:y_max, x_min:x_max],
            mask[z_min:z_max, y_min:y_max, x_min:x_max],
        )


class PadSample(object):
    def __call__(self, volume_mask):
        volume, mask = volume_mask
        a = volume.shape[1]
        b = volume.shape[2]

        if a == b:
            return volume, mask
        diff = (max(a, b) - min(a, b)) / 2.0
        padding_insert = (
            int(np.floor(diff)),
            int(np.ceil(diff)),
        )
        if a > b:
            padding = ((0, 0), (0, 0), padding_insert)
        else:
            padding = ((0, 0), padding_insert, (0, 0))
        mask = np.pad(mask, padding, mode="constant", constant_values=0)
        padding = padding + ((0, 0),)
        volume = np.pad(volume, padding, mode="constant", constant_values=0)
        return volume, mask


class ResizeSample(object):
    def __init__(self, size=256):
        self.size = size

    def __call__(self, volume_mask):
        volume, mask = volume_mask
        v_shape = volume.shape
        out_shape = (v_shape[0], self.size, self.size)
        mask = resize(
            mask,
            output_shape=out_shape,
            order=0,
            mode="constant",
            cval=0,
            anti_aliasing=False,
        )
        out_shape = out_shape + (v_shape[3],)
        volume = resize(
            volume,
            output_shape=out_shape,
            order=2,
            mode="constant",
            cval=0,
            anti_aliasing=False,
        )
        return volume, mask


class NormalizeVolume(object):
    def __call__(self, volume_mask):
        volume, mask = volume_mask
        p10 = np.percentile(volume, 10)
        p99 = np.percentile(volume, 99)
        volume = rescale_intensity(volume, in_range=(p10, p99))
        m = np.mean(volume, axis=(0, 1, 2))
        s = np.std(volume, axis=(0, 1, 2))
        volume = (volume - m) / s
        return volume, mask