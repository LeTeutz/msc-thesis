import torch
import cv2
import numpy as np
import random

def set_global_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_color_to_digit(img):
    """Apply random color to a digit"""
    if isinstance(img, torch.Tensor):
        color = torch.tensor([random.random(), random.random(), random.random()])
        # Reshape color to [3, 1, 1] for broadcasting
        color = color.view(3, 1, 1)
        return img * color
    return img

def apply_contour(img):
    """Extract contours from digit"""
    if isinstance(img, torch.Tensor):
        img = img.squeeze(0).numpy()
    img = (img * 255).astype(np.uint8)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(img)
    cv2.drawContours(result, contours, -1, (255,255,255), 1)
    result = torch.from_numpy(result.astype(np.float32) / 255.0)
    return result.repeat(3, 1, 1)

def apply_contrast(img, factor=2.0):
    """Enhance contrast of the image"""
    if isinstance(img, torch.Tensor):
        centered = img - 0.5
        contrasted = torch.clamp(centered * factor + 0.5, 0, 1)
        return contrasted
    return img

def apply_noise(img, noise_factor=0.1):
    """Add random noise to image"""
    if isinstance(img, torch.Tensor):
        noise = torch.randn_like(img) * noise_factor
        noisy_img = torch.clamp(img + noise, 0, 1)
        return noisy_img
    return img

def apply_blur(img, kernel_size=3):
    """Apply Gaussian blur"""
    if isinstance(img, torch.Tensor):
        np_img = img.squeeze(0).numpy()
        blurred = cv2.GaussianBlur(np_img, (kernel_size, kernel_size), 0)
        blurred_tensor = torch.from_numpy(blurred).float()
        return blurred_tensor.repeat(3, 1, 1)
    return img

def apply_inversion(img):
    """Invert image intensities"""
    if isinstance(img, torch.Tensor):
        return 1.0 - img
    return img

def compose_transforms(*transforms):
    """Compose multiple transforms into one"""
    def composed_transform(img):
        result = img
        for t in transforms:
            result = t(result)
        return result
    return composed_transform


TRANSFORMS_REGISTRY = {
    # Identity transform
    'identity': lambda x: x,
    
    # Single transformations
    'colorize': apply_color_to_digit,
    'contour': apply_contour,
    'contrast': apply_contrast,
    'noise': apply_noise,
    'blur': apply_blur,
    'invert': apply_inversion,
    
    # Pairs 
    'color_contour': compose_transforms(apply_contour, apply_color_to_digit),
    'color_contrast': compose_transforms(apply_color_to_digit, apply_contrast),
    'color_noise': compose_transforms(apply_color_to_digit, apply_noise),
    'contour_noise': compose_transforms(apply_contour, apply_noise),
    'contrast_noise': compose_transforms(apply_contrast, apply_noise),
    
    # Triples
    'color_contour_noise': compose_transforms(
        apply_contour, 
        apply_color_to_digit, 
        apply_noise
    ),
    'color_contrast_noise': compose_transforms(
        apply_color_to_digit, 
        apply_contrast, 
        apply_noise
    ),
    'contour_contrast_noise': compose_transforms(
        apply_contour, 
        apply_contrast, 
        apply_noise
    ),
    
    # Quadruples
    'color_contour_contrast_noise': compose_transforms(
        apply_contour,
        apply_color_to_digit,
        apply_contrast,
        apply_noise
    ),
    
    # All transformations
    'all_transforms': compose_transforms(
        apply_contour,
        apply_color_to_digit,
        apply_contrast,
        apply_noise,
        apply_blur
    ),
}