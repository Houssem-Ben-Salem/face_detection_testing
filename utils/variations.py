import cv2
import numpy as np
from typing import Tuple, Optional

class PoseVariation:
    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Rotate image by given angle
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image and rotation matrix
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Get new image dimensions
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust rotation matrix
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Perform rotation
        rotated_image = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (new_width, new_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        return rotated_image, rotation_matrix

    @staticmethod
    def apply_yaw(image: np.ndarray, angle: float) -> np.ndarray:
        """Apply yaw rotation (left-right head movement)"""
        rotated, _ = PoseVariation.rotate_image(image, angle)
        return rotated

    @staticmethod
    def apply_pitch(image: np.ndarray, angle: float) -> np.ndarray:
        """Apply pitch rotation (up-down head movement)"""
        # For pitch, we first rotate the image 90 degrees, apply the rotation,
        # then rotate back
        temp = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated, _ = PoseVariation.rotate_image(temp, angle)
        return cv2.rotate(rotated, cv2.ROTATE_90_COUNTERCLOCKWISE)

    @staticmethod
    def apply_roll(image: np.ndarray, angle: float) -> np.ndarray:
        """Apply roll rotation (tilting head side to side)"""
        rotated, _ = PoseVariation.rotate_image(image, angle)
        return rotated

class LightingVariation:
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image brightness
        
        Args:
            image: Input image
            factor: Brightness factor (0-2, 1 is original)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image contrast
        
        Args:
            image: Input image
            factor: Contrast factor (0-2, 1 is original)
        """
        mean = np.mean(image, axis=(0, 1))
        adjusted = factor * (image.astype(np.float32) - mean) + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    @staticmethod
    def adjust_color_temperature(image: np.ndarray, temperature: float) -> np.ndarray:
        """
        Adjust color temperature
        
        Args:
            image: Input image
            temperature: Color temperature in Kelvin (2000-7000)
        """
        # Normalize temperature to a scale of -1 to 1
        normalized_temp = (temperature - 4500) / 2500  # 4500K is neutral
        
        # Create color balance adjustments
        if normalized_temp > 0:
            # Warmer - increase red, decrease blue
            rb_factor = 1 + (0.2 * normalized_temp)
            bb_factor = 1 - (0.2 * normalized_temp)
        else:
            # Cooler - increase blue, decrease red
            rb_factor = 1 + (0.2 * normalized_temp)
            bb_factor = 1 - (0.2 * normalized_temp)
            
        # Apply color balance
        result = image.copy().astype(np.float32)
        result[:, :, 2] = np.clip(result[:, :, 2] * rb_factor, 0, 255)  # Red
        result[:, :, 0] = np.clip(result[:, :, 0] * bb_factor, 0, 255)  # Blue
        
        return result.astype(np.uint8)

def create_variation_functions():
    """Create dictionary of variation functions"""
    variations = {
        "None": lambda frame: frame,
        
        # Pose Variations
        "Yaw +30°": lambda frame: PoseVariation.apply_yaw(frame, 30),
        "Yaw -30°": lambda frame: PoseVariation.apply_yaw(frame, -30),
        "Pitch +20°": lambda frame: PoseVariation.apply_pitch(frame, 20),
        "Pitch -20°": lambda frame: PoseVariation.apply_pitch(frame, -20),
        "Roll +15°": lambda frame: PoseVariation.apply_roll(frame, 15),
        "Roll -15°": lambda frame: PoseVariation.apply_roll(frame, -15),
        
        # Lighting Variations
        "Low Light": lambda frame: LightingVariation.adjust_brightness(frame, 0.5),
        "High Light": lambda frame: LightingVariation.adjust_brightness(frame, 1.5),
        "Low Contrast": lambda frame: LightingVariation.adjust_contrast(frame, 0.7),
        "High Contrast": lambda frame: LightingVariation.adjust_contrast(frame, 1.3),
        "Warm Temperature": lambda frame: LightingVariation.adjust_color_temperature(frame, 6500),
        "Cool Temperature": lambda frame: LightingVariation.adjust_color_temperature(frame, 2500),
    }
    
    return variations