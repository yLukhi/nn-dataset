# Age-estimation optimised transform: 224×224, colour jitter, flip, blur, random erasing
# Designed to work on face crops from UTKFace (200-400px originals).
import torchvision.transforms as T


def transform(norm):
    return T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224, scale=(0.80, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.ToTensor(),
        T.Normalize(*norm),
        T.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
    ])
