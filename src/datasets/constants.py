import torch
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2

from PIL import Image

# Known values for the ImageNet dataset mean and std
IMAGENET_MEAN_CAFFE = [0.406 * 255, 0.456 * 255, 0.485 * 255]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# transforms.Normalize() subtracts the mean value and divides by the standard deviation
# Passing std=[1.0, 1.0, 1.0] means we only perform the mean subtraction.
# (PyTorch VGG16 on ImageNet does actual normalization, so use this only when working with the Caffe model)
MEAN_SUBTRACTION_ONLY = [1.0, 1.0, 1.0]

# Specifiying only one dimension instead of two will preserve aspect ratio
# Only the shorter dimension will be resized to 256. The other will still be longer.
TRANSFORM_TO_SIZE_TRAIN = 256

TRANSFORM_TO_SIZE_TEST = 256

TRANSFORM_TO_SIZE_TEST_CAFFE = 224  # Works better for the Caffe model

CROP_SIZE = [224, 224]

TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(TRANSFORM_TO_SIZE_TRAIN),
        transforms.RandomCrop(CROP_SIZE),
        # Original paper randomly flipped training images but only in training phase
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

VAL_TEST_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(TRANSFORM_TO_SIZE_TEST),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

VAL_TEST_TRANSFORMS_CAFFE = transforms.Compose(
    [
        transforms.Resize(TRANSFORM_TO_SIZE_TEST_CAFFE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.Lambda(
            lambda img: Image.merge(
                "RGB", (img.getchannel(2), img.getchannel(1), img.getchannel(0))
            )
        ),
        transforms.PILToTensor(),
        transforms_v2.ToDtype(torch.float32, scale=False),
        transforms.Normalize(mean=IMAGENET_MEAN_CAFFE, std=[1, 1, 1]),
    ]
)
