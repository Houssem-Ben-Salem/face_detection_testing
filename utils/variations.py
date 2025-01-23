import cv2
import numpy as np
from typing import Tuple, Optional

class PoseVariation:
    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
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
        rotated, _ = PoseVariation.rotate_image(image, angle)
        return rotated

    @staticmethod
    def apply_pitch(image: np.ndarray, angle: float) -> np.ndarray:
        temp = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated, _ = PoseVariation.rotate_image(temp, angle)
        return cv2.rotate(rotated, cv2.ROTATE_90_COUNTERCLOCKWISE)

    @staticmethod
    def apply_roll(image: np.ndarray, angle: float) -> np.ndarray:
        rotated, _ = PoseVariation.rotate_image(image, angle)
        return rotated

class LightingVariation:
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        mean = np.mean(image, axis=(0, 1))
        adjusted = factor * (image.astype(np.float32) - mean) + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    @staticmethod
    def adjust_color_temperature(image: np.ndarray, temperature: float) -> np.ndarray:
        normalized_temp = (temperature - 4500) / 2500
        
        if normalized_temp > 0:
            rb_factor = 1 + (0.2 * normalized_temp)
            bb_factor = 1 - (0.2 * normalized_temp)
        else:
            rb_factor = 1 + (0.2 * normalized_temp)
            bb_factor = 1 - (0.2 * normalized_temp)
            
        result = image.copy().astype(np.float32)
        result[:, :, 2] = np.clip(result[:, :, 2] * rb_factor, 0, 255)
        result[:, :, 0] = np.clip(result[:, :, 0] * bb_factor, 0, 255)
        
        return result.astype(np.uint8)

class PerturbationVariation:
    @staticmethod
    def add_gaussian_noise(image: np.ndarray, std_dev: float = 25) -> np.ndarray:
        noise = np.random.normal(0, std_dev, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image

    @staticmethod
    def add_snow(image: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        result = image.copy()
        height, width = result.shape[:2]
        num_snow = int((height * width) * intensity)
        
        snow_points = np.random.randint(0, width, (num_snow, 2))
        snow_points[:, 1] = np.random.randint(0, height, num_snow)
        
        for point in snow_points:
            x, y = point
            cv2.circle(result, (x, y), np.random.randint(1, 3), (255, 255, 255), -1)
            
        return result

def create_variation_functions():
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
        
        # Perturbation Variations
        "Noise": lambda frame: PerturbationVariation.add_gaussian_noise(frame),
        "Snow": lambda frame: PerturbationVariation.add_snow(frame),
        "Heavy Snow": lambda frame: PerturbationVariation.add_snow(frame, 0.2),
        "Light Noise": lambda frame: PerturbationVariation.add_gaussian_noise(frame, 15),
        "Heavy Noise": lambda frame: PerturbationVariation.add_gaussian_noise(frame, 35)
    }
    
    return variations