import os
import tensorflow as tf
import scipy.ndimage as ndi
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)

# IMGS_DIR = r'/mnt/d/Updated_Dataset/Images'
# MASKS_DIR = r'/mnt/d/Updated_Dataset/Masks'
# print(os.listdir(IMGS_DIR))
# print(f"Total Imgs: {len(os.listdir(IMGS_DIR))} | Total Masks: {len(os.listdir(MASKS_DIR))}")

# IMAGE_SIZE = (256, 256)  # Match U-Net input size

def load_image(image_path, IMAGE_SIZE):
    if not os.path.exists(image_path): return None
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    img = cv2.resize(img, IMAGE_SIZE)  # Resize
    img = img.astype(np.float32) / 255.0  # Normalize
    return np.expand_dims(img, axis=-1)


def load_mask(mask_path, IMAGE_SIZE):
    if not os.path.exists(mask_path): return None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return None

    mask = cv2.resize(mask, IMAGE_SIZE)  # Resize
    mask = (mask > 127).astype(np.uint8)  # Binarize
    return np.expand_dims(mask, axis=-1)


# Function to get all image-mask pairs recursively
def get_image_mask_pairs(image_dir, mask_dir):
    image_mask_pairs = []

    for patient_id in sorted(os.listdir(image_dir)):  
        
        patient_image_path = os.path.join(image_dir, patient_id)
        patient_mask_path = os.path.join(mask_dir, patient_id)

        if not os.path.isdir(patient_image_path):
            continue

        if not os.path.isdir(patient_image_path) or not os.path.isdir(patient_mask_path):
            continue

        # Get all images and masks in the patient folder
        image_files = sorted(os.listdir(patient_image_path))
        mask_files = sorted(os.listdir(patient_mask_path))

        # Ensure images and masks are paired correctly
        for img_file, mask_file in zip(image_files, mask_files):
            
            img_path = os.path.join(patient_image_path, img_file)
            mask_path = os.path.join(patient_mask_path, mask_file)

            if img_path.split("/")[-1][0] == ".": continue
            
            if img_path == "/kaggle/input/mdn-lymph-nodes/Images/Images/P1/._adb-p1-img-366.png":
                continue

            if os.path.exists(mask_path):  # Ensure mask exists
                image_mask_pairs.append((img_path, mask_path))

    return image_mask_pairs


class ElasticDeformationAugmentation:
    def __init__(self, alpha=34, sigma=5, random_state=None):
        """
        Initialize Elastic Deformation Augmentation.
        
        Args:
            alpha (float): Scaling factor for deformation magnitude.
                Higher values create more dramatic deformations.
            sigma (float): Standard deviation for Gaussian filter.
                Controls the smoothness of deformations.
            random_state (int, optional): Seed for reproducibility.
        """
        self.alpha = alpha
        self.sigma = sigma
        self.random_state = random_state or np.random.randint(0, 2**32 - 1)
        
    def _generate_grid_displacement(self, shape):
        """
        Generate smooth random displacements using Gaussian filtering.
        
        Args:
            shape (tuple): Shape of the image (height, width)
        
        Returns:
            tuple: Displacement fields for x and y directions
        """
        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        
        # Generate random displacement fields
        dx = np.random.rand(*shape) * 2 - 1
        dy = np.random.rand(*shape) * 2 - 1
        
        # Apply Gaussian filter to create smooth deformations
        # This creates spatially continuous, smooth deformations
        dx = ndi.gaussian_filter(dx, sigma=self.sigma) * self.alpha
        dy = ndi.gaussian_filter(dy, sigma=self.sigma) * self.alpha
        
        return dx, dy
    
    def _elastic_deform_2d(self, image, dx, dy):
        """
        Apply elastic deformation to a 2D image.
        
        Args:
            image (np.ndarray): Input 2D image
            dx (np.ndarray): Displacement in x-direction
            dy (np.ndarray): Displacement in y-direction
        
        Returns:
            np.ndarray: Deformed image
        """
        # Create coordinate grid
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        
        # Apply displacements to coordinates
        deformed_x = x + dx
        deformed_y = y + dy
        
        # Interpolate deformed image
        deformed_image = ndi.map_coordinates(
            image, 
            [deformed_y, deformed_x], 
            order=1,  # Bilinear interpolation
            mode='nearest'  # Handling image boundary
        )
        
        return deformed_image
    
    def augment_images(self, images, masks):
        """
        Apply elastic deformation to images and masks.
        
        Args:
            images (np.ndarray): Input images (N, H, W, C)
            masks (np.ndarray): Corresponding masks (N, H, W, C)
        
        Returns:
            tuple: Deformed images and masks
        """
        deformed_images = []
        deformed_masks = []
        
        for img, mask in zip(images, masks):
            # Squeeze to 2D if needed (assuming single-channel images)
            img_2d = img.squeeze()
            mask_2d = mask.squeeze()
            
            # Generate displacement fields
            dx, dy = self._generate_grid_displacement(img_2d.shape)
            
            # Apply deformations
            deformed_img = self._elastic_deform_2d(img_2d, dx, dy)
            deformed_mask = self._elastic_deform_2d(mask_2d, dx, dy)
            
            # Restore original shape
            deformed_images.append(deformed_img[..., np.newaxis])
            deformed_masks.append(deformed_mask[..., np.newaxis])
        
        return np.array(deformed_images), np.array(deformed_masks)
    
    def generate_augmented_dataset(self, images, masks, augmentation_mode='multiply', target_size=None):
        """
        Generate an augmented dataset with elastic deformations.
        
        Args:
            images (np.ndarray): Original images
            masks (np.ndarray): Original masks
            augmentation_mode (str): Augmentation strategy
                - 'multiply': Generate multiple augmented samples per original image
                - 'expand': Expand dataset to a specific total number of images
            target_size (int, optional): 
                - If mode is 'multiply': Number of augmented samples per original image
                - If mode is 'expand': Total desired number of images in the final dataset
        
        Returns:
            tuple: Augmented images and masks
        """
        # Validate inputs
        if augmentation_mode not in ['multiply', 'expand']:
            raise ValueError("augmentation_mode must be either 'multiply' or 'expand'")
        
        # Determine augmentation strategy
        if augmentation_mode == 'multiply':
            # Default to 2 if not specified
            target_size = target_size if target_size else 2
            
            # Generate multiple augmented samples per original image
            all_augmented_images = [images]
            all_augmented_masks = [masks]
            
            for _ in range(target_size):
                aug_images, aug_masks = self.augment_images(images, masks)
                all_augmented_images.append(aug_images)
                all_augmented_masks.append(aug_masks)
            
            return np.concatenate(all_augmented_images), np.concatenate(all_augmented_masks)
        
        else:  # 'expand' mode
            # Calculate how many additional images we need to generate
            current_size = len(images)
            
            if target_size is None or target_size <= current_size:
                return images, masks
            
            additional_needed = target_size - current_size
            
            # Track augmented images
            all_augmented_images = [images]
            all_augmented_masks = [masks]
            
            # Generate additional images
            while len(all_augmented_images[0]) < target_size:
                # Generate augmentations
                aug_images, aug_masks = self.augment_images(images, masks)
                
                # Add augmented images
                all_augmented_images.append(aug_images)
                all_augmented_masks.append(aug_masks)
            
            # Concatenate and slice to exact target size
            augmented_images = np.concatenate(all_augmented_images)[:target_size]
            augmented_masks = np.concatenate(all_augmented_masks)[:target_size]
            
            return augmented_images, augmented_masks
        
class DataGenerator:
    def __init__(self, IMGS_DIR:str, MASKS_DIR:str, IMAGE_SIZE:tuple, test_size:float = 0.15):
        self.IMAGE_SIZE = IMAGE_SIZE
        self.IMGS_DIR = IMGS_DIR
        self.MASKS_DIR = MASKS_DIR
        self.image_mask_list = get_image_mask_pairs(self.IMGS_DIR, self.MASKS_DIR)
        self.images = np.array([load_image(img_path, self.IMAGE_SIZE) for img_path, _ in self.image_mask_list])
        self.masks = np.array([load_mask(mask_path, self.IMAGE_SIZE) for _, mask_path in self.image_mask_list])
        self.masks = self.masks.astype("float32")

        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.images, self.masks, test_size=test_size, random_state=42)
        self.train_path, self.val_path = train_test_split(self.image_mask_list, test_size=test_size, random_state=42)

    def get_local_data(self) -> tuple:
        return (self.X_train, self.Y_train, self.X_val, self.Y_val)

    def get_augmented_data(self, alpha=20, sigma=3, mode='multiply', target=2):
        augmenter = ElasticDeformationAugmentation(
                alpha,  # Deformation magnitude
                sigma   # Smoothness of deformation
            )

        augmented_images, augmented_masks = augmenter.generate_augmented_dataset(
                self.X_train, self.Y_train,
                augmentation_mode=mode,
                target_size=target
            )
        
        return augmented_images, augmented_masks
