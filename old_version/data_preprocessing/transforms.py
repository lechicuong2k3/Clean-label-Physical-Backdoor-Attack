from math import ceil
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
torchvision.disable_beta_transforms_warning()

def pytorch_pipeline():
    pipeline = v2.RandomApply([v2.RandomChoice([
            v2.RandomHorizontalFlip(p=1),  # Fliplr
            v2.RandomRotation(degrees= (-35, 35)),
            v2.RandomAffine(degrees=(-25, 25), translate=(0.15, 0.15), scale=(0.8, 1.2), shear=(-16, 16), fill=0),
            v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
            v2.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 2.0)),
            v2.RandomPerspective(distortion_scale=0.3, p=1.0),
            v2.RandomAdjustSharpness(1.5, p=1),
        ], p=[0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1]) ], p=0.8)
    return pipeline

data_transforms = {
    'train_augment':
    v2.Compose([
        v2.ToImageTensor(),
        pytorch_pipeline(),
        v2.ToDtype(torch.float32),
        # v2.Normalize(mean=[0.5570, 0.5435, 0.5305], std=[0.2727, 0.2702, 0.2787])
    ]),
    'train':
    v2.Compose([
        v2.ToImageTensor(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32),
        # v2.Normalize(mean=[0.5570, 0.5435, 0.5305], std=[0.2727, 0.2702, 0.2787])
    ]),
    'test':
    v2.Compose([
        v2.ToImageTensor(),
        v2.ToDtype(torch.float32),
        # v2.Normalize(mean=[0.5570, 0.5435, 0.5305], std=[0.2727, 0.2702, 0.2787])
    ]),
}

# from PIL import Image
# import imgaug as ia
# import imgaug.augmenters as iaa
# import imageio.v2 as imageio  
# from imgaug.augmentables.batches import UnnormalizedBatch

# sometimes = lambda aug: iaa.Sometimes(0.5, aug)
# less_frequent = lambda aug: iaa.Sometimes(0.25, aug)
# more_frequent = lambda aug: iaa.Sometimes(0.75, aug)

# def prepare_pipeline():
#     """
#     Either perform transformation (with high probability) or do nothing (low probability)
#     """
#     aug = iaa.SomeOf((1, 2), 
#         [
#             iaa.Identity(),
#             iaa.Sequential(
#             [ 
#                 iaa.Fliplr(p=0.5), 
#                 # crop images by -5% to 5% of their height/width
#                 less_frequent(iaa.CropAndPad(
#                     percent=(-0.05, 0.05),
#                     pad_mode=ia.ALL,
#                     pad_cval=(0, 255)
#                 )),
                
#                 more_frequent(iaa.Affine(
#                     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
#                     translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
#                     rotate=(-25, 25), # rotate by -25 to +25 degrees
#                     shear={"x": (-16, 16), "y": (-16, 16)},  # shear by -16 to +16 degrees
#                     order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
#                     cval=(0, 255), # if mode is constant, use a cval between 0 and 255
#                     mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
#                 )),
                
#                 sometimes(iaa.OneOf(
#                     [
#                     iaa.LinearContrast(alpha=(0.8, 1.2)), 
#                     iaa.LogContrast(gain=(0.8, 1.2))
#                     ]
#                 )),

#                 sometimes(iaa.AddToBrightness((-25, 25))),
#                 sometimes(iaa.AddToSaturation((-25, 35))),
                
#                 sometimes(iaa.SomeOf((0, 2),
#                     [
#                         sometimes(iaa.OneOf([
#                             iaa.GaussianBlur((0, 1)), # blur images with a sigma between 0 and 3.0
#                             iaa.AverageBlur(k=(1, 3)), # blur image using local means with kernel sizes between 2 and 7
#                             iaa.MedianBlur(k=(1, 3)), # blur image using local medians with kernel sizes between 2 and 7
#                         ])),
#                         iaa.Grayscale(alpha=(0, 0.3)),
#                         sometimes(iaa.Sharpen(alpha=(0.5, 1.0), lightness=(0.8, 1.2))), # sharpen images
#                         sometimes(iaa.PerspectiveTransform(scale=(0, 0.06)))
#                     ],
#                     random_order=True
#                 ))
#             ], random_order=True)
#         ], random_order=False)
#     return aug

# def augment_and_save():
#     """
#     Augment and save to hard disk
#     """
#     src_dir = "/vinserver_user/21thinh.dd/FedBackdoor/source/dataset/facial_recognition_rescale_split_augment"
#     aug = prepare_pipeline()
#     for dir in os.listdir(src_dir):
#         for trigger_type in os.listdir(os.path.join(src_dir, dir)):
#             trigger_dir = os.path.join(src_dir, dir, trigger_type)
#             cnt = len(os.listdir(trigger_dir))
#             if trigger_type == "clean_image": BATCH_SIZE = ceil(1000/cnt)
#             else: BATCH_SIZE = ceil(800/cnt)
#             batches = []
#             for image_name in os.listdir(os.path.join(src_dir, dir, trigger_type)):
#                 path = os.path.join(src_dir, dir, trigger_type, image_name)
#                 image = imageio.imread(path)
#                 image_batch = [np.copy(image) for _ in range(BATCH_SIZE)]
#                 batches.append(UnnormalizedBatch(images=image_batch))
            
                
#             batches_aug = list(aug.augment_batches(batches, background=True))
#             for i in range(len(batches_aug)):
#                 for image_aug in batches_aug[i].images_aug:
#                     cnt += 1
#                     save_path = f"{trigger_dir}/{cnt}.jpg"
#                     imageio.imwrite(save_path, image_aug)
#                     print(f"Augmented and saved image: {save_path}")
#     print("Augmentation finished.")

"""Reference code: https://github.com/aleju/imgaug/issues/406"""

# def loader(path: str) -> np.ndarray:
#     img = Image.open(path)  # Open the image using PIL
#     img = np.array(img)  # Convert PIL Image to NumPy array
#     return img

# def worker_init_fn(worker_id):
#     ia.seed(np.random.get_state()[1][0] + worker_id)

# def collate_fn(data):
#     # TODO: Implement your function
#     return tuple(zip(*data))