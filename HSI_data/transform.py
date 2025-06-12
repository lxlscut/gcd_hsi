import numpy as np
import torch
from torchvision import transforms


class RandomCropMaskRotate:
    def __init__(self, mask_ratio, mask_value=0, mask_mode="full"):
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.mask_mode = mask_mode
        self.noise_std = 0.1


    def weighted_spectral_noise(self, data):
        weights = np.linspace(0.1, 0.5, data.shape[2])  # 每通道不同权重
        noise = np.random.normal(loc=0, scale=weights, size=data.shape)
        return data + noise

    def random_flip(self,data):
        """随机翻转操作"""
        flip_type = np.random.choice(["none", "horizontal", "vertical", "both"])  # 随机选择翻转类型
        if flip_type == "horizontal":
            flipped_data = np.flip(data, axis=2)  # 水平翻转 (H 维度)
        elif flip_type == "vertical":
            flipped_data = np.flip(data, axis=1)  # 垂直翻转 (W 维度)
        elif flip_type == "both":
            flipped_data = np.flip(np.flip(data, axis=1), axis=2)  # 同时水平和垂直翻转
        else:
            flipped_data = data  # 不翻转
        return flipped_data.copy()

    def random_rotate(self, data):
        """随机旋转操作"""
        k = np.random.choice([0, 1, 2, 3])  # 0: 0°, 1: 90°, 2: 180°, 3: 270°
        rotated_data = np.rot90(data, k=k, axes=(1, 2))  # 在 W 和 H 维度上旋转
        return rotated_data.copy()

    def random_noise(self, data):
        """
        为每个通道单独添加高斯噪声，标准差随机位于 [0.5*noise_std, 2*noise_std] 的范围内。
        """
        C, W, H = data.shape
        noisy_data = data.copy()

        lower_bound = 0.5 * self.noise_std
        upper_bound = 2.0 * self.noise_std

        for c in range(C):
            band_std = np.random.uniform(lower_bound, upper_bound)
            noise_c = np.random.normal(loc=0, scale=self.noise_std, size=(W, H))
            noisy_data[c] += noise_c

        return noisy_data

    def __call__(self, sample):
        """执行随机裁剪、遮挡、旋转和添加噪声"""
        sample = self.random_noise(sample)
        sample = self.random_rotate(sample)
        sample = self.random_flip(sample)
        return sample


def get_transform(transform_type='hyperspectral', mask_ratio=0.3, mask_value=0, mask_mode="full"):
    """
        transform_type: str 'hyperspectral'。
        crop_size: tuple
        mask_ratio: float
        mask_value: int or float
        mask_mode: str
        train_transform: transforms.Compose
        test_transform: transforms.Compose
    """
    if transform_type == 'hyperspectral':
        train_transform = transforms.Compose([
            transforms.Lambda(lambda x: RandomCropMaskRotate(
                mask_ratio=mask_ratio,
                mask_value=mask_value,
                mask_mode=mask_mode
            )(x)),
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32))  # 转为 Torch Tensor
        ])

        test_transform = transforms.Compose([
            transforms.Lambda(lambda x: RandomCropMaskRotate(
                mask_ratio=0.01,
                mask_value=mask_value,
                mask_mode=mask_mode
            )(x)),
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32))  # 转为 Torch Tensor
        ])

    else:
        raise NotImplementedError(f"Transform type '{transform_type}' is not implemented.")

    return train_transform, test_transform
