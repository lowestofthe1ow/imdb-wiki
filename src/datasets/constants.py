import torchvision.transforms as transforms

# When working with the Caffe model set this to 255
SCALE_FACTOR = 1

# Known values for the ImageNet dataset mean and std
IMAGENET_MEAN = [channel * SCALE_FACTOR for channel in [0.485, 0.456, 0.406]]
IMAGENET_STD = [0.229, 0.224, 0.225]

# transforms.Normalize() subtracts the mean value and divides by the standard deviation
# Passing std=[1.0, 1.0, 1.0] means we only perform the mean subtraction.
# (This is what Caffe did)
MEAN_SUBTRACTION_ONLY = [1.0, 1.0, 1.0]

# Specifiying only one dimension instead of two will preserve aspect ratio
# Only the shorter dimension will be resized to 256. The other will still be longer.
TRANSFORM_TO_SIZE_TRAIN = 256

# For val/test I've found that resizing straight to 224 then center-cropping performs better
# .......sometimes
TRANSFORM_TO_SIZE_TEST = 224

CROP_SIZE = [224, 224]

TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(TRANSFORM_TO_SIZE_TRAIN),
        transforms.RandomCrop(CROP_SIZE),
        # Original paper randomly flipped training images but only in training phase
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=MEAN_SUBTRACTION_ONLY),
    ]
)

VAL_TEST_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(TRANSFORM_TO_SIZE_TEST),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=MEAN_SUBTRACTION_ONLY),
    ]
)
