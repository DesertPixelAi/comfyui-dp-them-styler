import torch
from PIL import Image, ImageOps
import numpy as np
from typing import Tuple, Optional


class DP_Add_Logo_Banner:
    """
    A ComfyUI node that adds a logo banner to an image with customizable positioning and sizing
    
    Features:
    - Support for transparent PNG logos
    - Corner positioning (top-left, top-right, bottom-left, bottom-right)
    - Customizable padding from edges
    - Logo height control with aspect ratio preservation
    - Handles different image sizes automatically
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "logo": ("IMAGE",),
                "location": (["top-left", "top-right", "bottom-left", "bottom-right"], 
                           {"default": "top-right"}),
                "padding": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "display": "number"
                }),
                "logo_height": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),

                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "number"
                }),
            },
            "optional": {
                "logo_mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "add_logo_banner"
    CATEGORY = "DP/Image"
    
    def __init__(self):
        pass
    
    def tensor_to_pil(self, img_tensor):
        """Convert ComfyUI image tensor to PIL Image"""
        try:
            # ComfyUI images are in format [B, H, W, C] with values 0-1
            if torch.is_tensor(img_tensor):
                img_tensor = img_tensor.cpu()
                if len(img_tensor.shape) == 4:
                    img_tensor = img_tensor[0]  # Remove batch dimension
                
                # Convert to numpy and scale to 0-255
                image_np = (img_tensor.numpy() * 255).astype(np.uint8)
                
                # Convert to PIL Image (keep as RGB for ComfyUI compatibility)
                pil_image = Image.fromarray(image_np, mode='RGB')
                
                return pil_image
            else:
                raise ValueError("Input is not a tensor")
        except Exception as e:
            raise Exception(f"Error converting tensor to PIL: {str(e)}")
    
    def mask_to_pil(self, mask_tensor):
        """Convert ComfyUI mask tensor to PIL Image"""
        try:
            if torch.is_tensor(mask_tensor):
                mask_tensor = mask_tensor.cpu()
                if len(mask_tensor.shape) == 3:
                    mask_tensor = mask_tensor[0]  # Remove batch dimension
                elif len(mask_tensor.shape) == 4:
                    mask_tensor = mask_tensor[0, :, :, 0]  # Remove batch and channel dimensions
                
                # Convert to numpy and scale to 0-255
                mask_np = (mask_tensor.numpy() * 255).astype(np.uint8)
                
                # Convert to PIL Image in L (grayscale) mode
                pil_mask = Image.fromarray(mask_np, mode='L')
                
                return pil_mask
            else:
                raise ValueError("Input is not a tensor")
        except Exception as e:
            raise Exception(f"Error converting mask tensor to PIL: {str(e)}")
    
    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor format (always RGB)"""
        try:
            # Ensure RGB mode for ComfyUI compatibility
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            
            # Convert to tensor with batch dimension [B, H, W, C]
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
            
            return image_tensor
        except Exception as e:
            raise Exception(f"Error converting PIL to tensor: {str(e)}")
    
    def resize_logo_with_aspect_ratio(self, logo_pil, target_height):
        """Resize logo maintaining aspect ratio based on target height"""
        try:
            original_width, original_height = logo_pil.size
            
            # Calculate new width maintaining aspect ratio
            aspect_ratio = original_width / original_height
            new_width = int(target_height * aspect_ratio)
            
            # Resize the logo
            resized_logo = logo_pil.resize((new_width, target_height), Image.Resampling.LANCZOS)
            
            return resized_logo
        except Exception as e:
            raise Exception(f"Error resizing logo: {str(e)}")
    
    def calculate_position(self, image_size, logo_size, location, padding):
        """Calculate the position coordinates for logo placement"""
        img_width, img_height = image_size
        logo_width, logo_height = logo_size
        
        if location == "top-left":
            x = padding
            y = padding
        elif location == "top-right":
            x = img_width - logo_width - padding
            y = padding
        elif location == "bottom-left":
            x = padding
            y = img_height - logo_height - padding
        elif location == "bottom-right":
            x = img_width - logo_width - padding
            y = img_height - logo_height - padding
        else:
            # Default to top-right
            x = img_width - logo_width - padding
            y = padding
        
        # Ensure coordinates are within bounds
        x = max(0, min(x, img_width - logo_width))
        y = max(0, min(y, img_height - logo_height))
        
        return (x, y)
    

    
    def add_logo_banner(self, image, logo, location, padding, logo_height, opacity=1.0, logo_mask=None):
        """
        Add logo banner to image
        
        Args:
            image: Main image tensor
            logo: Logo image tensor
            location: Corner position for logo
            padding: Distance from edges in pixels
            logo_height: Target height for logo in pixels
            opacity: Logo opacity (0.0 to 1.0)
            logo_mask: Optional mask tensor for logo transparency
        """
        try:
            # Convert tensors to PIL Images
            main_image_pil = self.tensor_to_pil(image)
            logo_pil = self.tensor_to_pil(logo)
            
            # Handle mask if provided
            mask_pil = None
            if logo_mask is not None:
                mask_pil = self.mask_to_pil(logo_mask)
            
            # Resize logo to target height maintaining aspect ratio
            resized_logo = self.resize_logo_with_aspect_ratio(logo_pil, logo_height)
            
            # Resize mask to match resized logo if mask exists
            if mask_pil is not None:
                mask_pil = mask_pil.resize(resized_logo.size, Image.Resampling.LANCZOS)
                # Invert the mask (black becomes white, white becomes black)
                mask_pil = ImageOps.invert(mask_pil)
            
            # Calculate position for logo placement
            position = self.calculate_position(
                main_image_pil.size,
                resized_logo.size,
                location,
                padding
            )
            
            # Create a copy of the main image to work with
            result_image = main_image_pil.copy()
            
            # Apply opacity to mask if needed
            if mask_pil is not None and opacity < 1.0:
                # Apply opacity to the mask
                mask_pil = mask_pil.point(lambda p: int(p * opacity))
            
            # Paste the logo onto the main image using mask for transparency
            if mask_pil is not None:
                result_image.paste(resized_logo, position, mask_pil)
            else:
                # No mask - paste normally with opacity
                if opacity < 1.0:
                    # Create a mask from opacity
                    opacity_mask = Image.new('L', resized_logo.size, int(255 * opacity))
                    result_image.paste(resized_logo, position, opacity_mask)
                else:
                    result_image.paste(resized_logo, position)
            
            # Convert back to tensor format
            result_tensor = self.pil_to_tensor(result_image)
            
            return (result_tensor,)
            
        except Exception as e:
            raise Exception(f"Error adding logo banner: {str(e)}")
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always process when inputs change
        return float("NaN")
