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
        color = color.view(3, 1, 1)
        return img * color
    return img


def apply_contour(img):
    """Extract contours from a digit and convert back to RGB."""
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        
        if img.ndim == 3 and img.shape[0] == 1:  
            img = img.squeeze(0) 
        elif img.ndim == 3 and img.shape[0] == 3: 
            img = np.transpose(img, (1, 2, 0))
        elif img.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected tensor shape: {img.shape}")
    
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    if img.ndim == 2:  
        gray = img
    elif img.ndim == 3 and img.shape[-1] == 1:  
        gray = img.squeeze(-1)  
    elif img.ndim == 3:  
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unexpected image shape after conversion: {img.shape}")
    
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = np.zeros_like(gray)
    cv2.drawContours(contour_image, contours, -1, 255, 1)  # Draw contours in white

    result_rgb = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2RGB)

    result_tensor = torch.from_numpy(result_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return result_tensor

def apply_contrast(img, factor=2.0):
    """Enhance contrast of the image"""
    if isinstance(img, torch.Tensor):
        centered = img - 0.5
        contrasted = torch.clamp(centered * factor + 0.5, 0, 1)
        return contrasted
    return img

def apply_noise(img, noise_factor=0.4):
    """Add random noise to image"""
    if isinstance(img, torch.Tensor):
        noise = torch.randn_like(img) * noise_factor
        noisy_img = torch.clamp(img + noise, 0, 1)
        return noisy_img
    return img

def apply_blur(img, kernel_size=3):
    """Apply Gaussian blur"""
    if isinstance(img, torch.Tensor):
        if img.ndimension() == 3 and img.shape[0] == 3:
            np_img = img.permute(1, 2, 0).numpy()  
            blurred = cv2.GaussianBlur(np_img, (kernel_size, kernel_size), 0)
            blurred_tensor = torch.from_numpy(blurred).permute(2, 0, 1).float()  
            return blurred_tensor

        elif img.ndimension() == 3 and img.shape[0] == 1:
            np_img = img.squeeze(0).numpy()  
            blurred = cv2.GaussianBlur(np_img, (kernel_size, kernel_size), 0)
            blurred_tensor = torch.from_numpy(blurred).float()
            return blurred_tensor.unsqueeze(0).repeat(3, 1, 1)  

        elif img.ndimension() == 2:
            blurred = cv2.GaussianBlur(img.numpy(), (kernel_size, kernel_size), 0)
            blurred_tensor = torch.from_numpy(blurred).float()
            return blurred_tensor.unsqueeze(0).repeat(3, 1, 1)  
        
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



def apply_gaussian_blur(img, kernel_size=3, sigma=0):
    """Apply Gaussian blur with configurable parameters"""
    if isinstance(img, torch.Tensor):
        if img.ndimension() == 3 and img.shape[0] == 3:
            np_img = img.permute(1, 2, 0).numpy()
            blurred = cv2.GaussianBlur(np_img, (kernel_size, kernel_size), sigma)
            return torch.from_numpy(blurred).permute(2, 0, 1).float()
        else:
            np_img = img.squeeze(0).numpy()
            blurred = cv2.GaussianBlur(np_img, (kernel_size, kernel_size), sigma)
            blurred_tensor = torch.from_numpy(blurred).float()
            return blurred_tensor.unsqueeze(0).repeat(3, 1, 1)
    return img

def apply_median_blur(img, kernel_size=3):
    """Apply median blur - good for salt&pepper noise"""
    if isinstance(img, torch.Tensor):
        if img.ndimension() == 3 and img.shape[0] == 3:
            np_img = img.permute(1, 2, 0).numpy()
            blurred = cv2.medianBlur(np_img.astype(np.uint8), kernel_size)
            return torch.from_numpy(blurred).permute(2, 0, 1).float()
            
        elif img.ndimension() == 3 and img.shape[0] == 1:
            np_img = img.squeeze(0).numpy()
            blurred = cv2.medianBlur(np_img.astype(np.uint8), kernel_size)
            blurred_tensor = torch.from_numpy(blurred).float()
            return blurred_tensor.unsqueeze(0).repeat(3, 1, 1)
            
        elif img.ndimension() == 2:
            np_img = img.numpy()
            blurred = cv2.medianBlur(np_img.astype(np.uint8), kernel_size)
            blurred_tensor = torch.from_numpy(blurred).float()
            return blurred_tensor.unsqueeze(0).repeat(3, 1, 1)
    return img

def apply_box_blur(img, kernel_size=3):
    """Apply box blur (simple averaging)"""
    if isinstance(img, torch.Tensor):
        if img.ndimension() == 3 and img.shape[0] == 3:
            np_img = img.permute(1, 2, 0).numpy()
            blurred = cv2.boxFilter(np_img.astype(np.uint8), -1, (kernel_size, kernel_size))
            return torch.from_numpy(blurred).permute(2, 0, 1).float()
            
        elif img.ndimension() == 3 and img.shape[0] == 1:
            np_img = img.squeeze(0).numpy()
            blurred = cv2.boxFilter(np_img.astype(np.uint8), -1, (kernel_size, kernel_size))
            blurred_tensor = torch.from_numpy(blurred).float()
            return blurred_tensor.unsqueeze(0).repeat(3, 1, 1)
            
        elif img.ndimension() == 2:
            np_img = img.numpy()
            blurred = cv2.boxFilter(np_img.astype(np.uint8), -1, (kernel_size, kernel_size))
            blurred_tensor = torch.from_numpy(blurred).float()
            return blurred_tensor.unsqueeze(0).repeat(3, 1, 1)
    return img

def apply_motion_blur(img, kernel_size=3):
    """Apply motion blur effect"""
    if isinstance(img, torch.Tensor):
        if img.ndimension() == 3 and img.shape[0] == 3:
            np_img = img.permute(1, 2, 0).numpy()
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            blurred = cv2.filter2D(np_img.astype(np.uint8), -1, kernel)
            return torch.from_numpy(blurred).permute(2, 0, 1).float()
            
        elif img.ndimension() == 3 and img.shape[0] == 1:
            np_img = img.squeeze(0).numpy()
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            blurred = cv2.filter2D(np_img.astype(np.uint8), -1, kernel)
            blurred_tensor = torch.from_numpy(blurred).float()
            return blurred_tensor.unsqueeze(0).repeat(3, 1, 1)
            
        elif img.ndimension() == 2:
            np_img = img.numpy()
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            blurred = cv2.filter2D(np_img.astype(np.uint8), -1, kernel)
            blurred_tensor = torch.from_numpy(blurred).float()
            return blurred_tensor.unsqueeze(0).repeat(3, 1, 1)
    return img

def apply_blur_and_invert(img, blur_func):
    """Apply any blur function followed by inversion"""
    blurred = blur_func(img)
    return 1.0 - blurred


BLUR_CONFIGS = {
    # Basic blurs
    'light_gaussian': lambda img: apply_gaussian_blur(img, kernel_size=3, sigma=1),
    'medium_gaussian': lambda img: apply_gaussian_blur(img, kernel_size=5, sigma=1.5),
    'heavy_gaussian': lambda img: apply_gaussian_blur(img, kernel_size=7, sigma=2),
    'very_heavy_gaussian': lambda img: apply_gaussian_blur(img, kernel_size=9, sigma=2.5),
    'median': lambda img: apply_median_blur(img, kernel_size=3),
    'box': lambda img: apply_box_blur(img, kernel_size=3),
    'motion': lambda img: apply_motion_blur(img, kernel_size=5),
    
    # Blur + inversion combinations
    'light_gaussian_inv': lambda img: apply_blur_and_invert(img, BLUR_CONFIGS['light_gaussian']),
    'medium_gaussian_inv': lambda img: apply_blur_and_invert(img, BLUR_CONFIGS['medium_gaussian']),
    'heavy_gaussian_inv': lambda img: apply_blur_and_invert(img, BLUR_CONFIGS['heavy_gaussian']),
    'median_inv': lambda img: apply_blur_and_invert(img, BLUR_CONFIGS['median']),
    'box_inv': lambda img: apply_blur_and_invert(img, BLUR_CONFIGS['box']),
    'motion_inv': lambda img: apply_blur_and_invert(img, BLUR_CONFIGS['motion']),
    
    # Double blur combinations
    'double_gaussian': lambda img: apply_gaussian_blur(apply_gaussian_blur(img, 3), 3),
    'gaussian_median': lambda img: apply_median_blur(apply_gaussian_blur(img, 3), 3),
    'double_gaussian_inv': lambda img: apply_blur_and_invert(img, BLUR_CONFIGS['double_gaussian'])
}

TRANSFORMS_REGISTRY = {
    'light_gaussian': lambda img: apply_gaussian_blur(img, kernel_size=3, sigma=1),
    'medium_gaussian': lambda img: apply_gaussian_blur(img, kernel_size=5, sigma=1.5),
    'heavy_gaussian': lambda img: apply_gaussian_blur(img, kernel_size=7, sigma=2),
    'very_heavy_gaussian': lambda img: apply_gaussian_blur(img, kernel_size=9, sigma=2.5),
    'median': lambda img: apply_median_blur(img, kernel_size=3),
    'box': lambda img: apply_box_blur(img, kernel_size=3),
    'motion': lambda img: apply_motion_blur(img, kernel_size=5),
    
    # Blur + inversion combinations
    'light_gaussian_inv': lambda img: apply_blur_and_invert(img, BLUR_CONFIGS['light_gaussian']),
    'medium_gaussian_inv': lambda img: apply_blur_and_invert(img, BLUR_CONFIGS['medium_gaussian']),
    'heavy_gaussian_inv': lambda img: apply_blur_and_invert(img, BLUR_CONFIGS['heavy_gaussian']),
    'median_inv': lambda img: apply_blur_and_invert(img, BLUR_CONFIGS['median']),
    'box_inv': lambda img: apply_blur_and_invert(img, BLUR_CONFIGS['box']),
    'motion_inv': lambda img: apply_blur_and_invert(img, BLUR_CONFIGS['motion']),
    
    # Double blur combinations
    'double_gaussian': lambda img: apply_gaussian_blur(apply_gaussian_blur(img, 3), 3),
    'gaussian_median': lambda img: apply_median_blur(apply_gaussian_blur(img, 3), 3),
    'double_gaussian_inv': lambda img: apply_blur_and_invert(img, BLUR_CONFIGS['double_gaussian']),


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
    'color_contour_contrast': compose_transforms(
        apply_contour,
        apply_color_to_digit,
        apply_contrast
    ),
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

# TRANSFORMS_REGISTRY.extend(BLUR_CONFIGS)