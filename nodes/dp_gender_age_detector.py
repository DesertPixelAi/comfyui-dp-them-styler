import torch
from PIL import Image
import numpy as np
from typing import Tuple, Dict, Optional
import re

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Please install it: pip install transformers")


class DP_Gender_Age_Detector:
    """
    A ComfyUI node that classifies both gender and age from an input image using Hugging Face models
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gender_model": (["rizvandwiki/gender-classification-2", 
                                "prithivMLmods/Realistic-Gender-Classification",
                                "dima806/man_woman_face_image_detection",
                                "Manual gender model"],),
                "age_model": (["nateraw/vit-age-classifier",
                             "dima806/faces_age_detection", 
                             "Manual age model"],),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "age_format": (["range", "single_value", "category"],),
            },
            "optional": {
                "manual_gender_model": ("STRING", {"default": ""}),
                "manual_age_model": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("gender", "gender_confidence", "age", "age_confidence", "age_category", "debug_info")
    FUNCTION = "detect_gender_age"
    CATEGORY = "DP/Image"
    
    def __init__(self):
        self.gender_processor = None
        self.gender_model = None
        self.current_gender_model = None
        
        self.age_processor = None
        self.age_model = None
        self.current_age_model = None
        
        # Cache for age pipelines (some models work better with pipelines)
        self.age_pipeline = None
    
    def load_gender_model(self, model_name):
        """Load the gender detection model if not already loaded or if model changed"""
        if not TRANSFORMERS_AVAILABLE:
            raise Exception("Transformers library not available. Please install it: pip install transformers")
            
        if model_name != self.current_gender_model:
            try:
                print(f"Loading gender detection model: {model_name}")
                self.gender_processor = AutoImageProcessor.from_pretrained(model_name)
                self.gender_model = AutoModelForImageClassification.from_pretrained(model_name)
                self.current_gender_model = model_name
                
                # Move model to GPU if available
                if torch.cuda.is_available():
                    self.gender_model = self.gender_model.cuda()
                    print(f"Gender model loaded on GPU: {model_name}")
                else:
                    print(f"Gender model loaded on CPU: {model_name}")
                    
            except Exception as e:
                raise Exception(f"Failed to load gender model {model_name}: {str(e)}")
    
    def load_age_model(self, model_name):
        """Load the age detection model if not already loaded or if model changed"""
        if not TRANSFORMERS_AVAILABLE:
            raise Exception("Transformers library not available. Please install it: pip install transformers")
            
        if model_name != self.current_age_model:
            try:
                print(f"Loading age detection model: {model_name}")
                
                # Some models work better with pipeline
                if model_name == "nateraw/vit-age-classifier":
                    self.age_pipeline = pipeline("image-classification", model=model_name)
                    self.age_processor = None
                    self.age_model = None
                else:
                    self.age_processor = AutoImageProcessor.from_pretrained(model_name)
                    self.age_model = AutoModelForImageClassification.from_pretrained(model_name)
                    self.age_pipeline = None
                    
                    # Move model to GPU if available
                    if torch.cuda.is_available():
                        self.age_model = self.age_model.cuda()
                        print(f"Age model loaded on GPU: {model_name}")
                    else:
                        print(f"Age model loaded on CPU: {model_name}")
                
                self.current_age_model = model_name
                    
            except Exception as e:
                raise Exception(f"Failed to load age model {model_name}: {str(e)}")
    
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
                
                # Convert to PIL Image
                pil_image = Image.fromarray(image_np, mode='RGB')
                return pil_image
            else:
                raise ValueError("Input is not a tensor")
        except Exception as e:
            raise Exception(f"Error converting tensor to PIL: {str(e)}")
    
    def classify_gender(self, pil_image, model_name):
        """Classify gender from image"""
        try:
            # Process image
            inputs = self.gender_processor(images=pil_image, return_tensors="pt")
            
            # Move inputs to same device as model
            if torch.cuda.is_available() and next(self.gender_model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.gender_model(**inputs)
                logits = outputs.logits
                
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get the predicted class
            predicted_class_idx = probs.argmax(-1).item()
            confidence = probs.max().item()
            
            # Map to gender string
            gender = self.get_gender_mapping(model_name, predicted_class_idx)
            
            return gender, confidence
            
        except Exception as e:
            raise Exception(f"Error in gender classification: {str(e)}")
    
    def classify_age(self, pil_image, model_name, age_format):
        """Classify age from image"""
        try:
            if self.age_pipeline:
                # Use pipeline for certain models
                results = self.age_pipeline(pil_image)
                
                # Get the top prediction
                if results:
                    top_result = results[0]
                    label = top_result['label']
                    confidence = top_result['score']
                    
                    # Parse age from label
                    age = self.parse_age_label(label, model_name, age_format)
                    return age, confidence
                else:
                    return "unknown", 0.0
            else:
                # Use standard transformers approach
                inputs = self.age_processor(images=pil_image, return_tensors="pt")
                
                # Move inputs to same device as model
                if torch.cuda.is_available() and next(self.age_model.parameters()).is_cuda:
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.age_model(**inputs)
                    logits = outputs.logits
                    
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get the predicted class
                predicted_class_idx = probs.argmax(-1).item()
                confidence = probs.max().item()
                
                # Map to age string
                age = self.get_age_mapping(model_name, predicted_class_idx, age_format)
                
                return age, confidence
                
        except Exception as e:
            raise Exception(f"Error in age classification: {str(e)}")
    
    def parse_age_label(self, label, model_name, age_format):
        """Parse age from various label formats"""
        # Extract numbers from label
        numbers = re.findall(r'\d+', label)
        
        if age_format == "single_value":
            if len(numbers) >= 2:
                # If range, return midpoint
                return str(int((int(numbers[0]) + int(numbers[1])) / 2))
            elif len(numbers) == 1:
                return numbers[0]
            else:
                return label
        elif age_format == "range":
            if len(numbers) >= 2:
                return f"{numbers[0]}-{numbers[1]}"
            else:
                return label
        else:  # category
            # Return the label as-is for category format
            return label
    
    def get_gender_mapping(self, model_name, class_idx):
        """Map model output to gender string"""
        mappings = {
            "rizvandwiki/gender-classification-2": {
                0: "female",
                1: "male"
            },
            "prithivMLmods/Realistic-Gender-Classification": {
                0: "female", 
                1: "male"
            },
            "dima806/man_woman_face_image_detection": {
                0: "male",
                1: "female"
            }
        }
        
        if model_name in mappings:
            return mappings[model_name].get(class_idx, "unknown")
        else:
            # Default mapping
            if class_idx == 0:
                return "female"
            elif class_idx == 1:
                return "male"
            else:
                return "unknown"
    
    def get_age_mapping(self, model_name, class_idx, age_format):
        """Map model output to age string based on format preference"""
        # Model-specific mappings
        if model_name == "dima806/faces_age_detection":
            # This model typically has age ranges
            age_ranges = {
                0: "0-2",
                1: "3-9", 
                2: "10-19",
                3: "20-29",
                4: "30-39",
                5: "40-49",
                6: "50-59",
                7: "60-69",
                8: "70+"
            }
            
            if class_idx in age_ranges:
                range_str = age_ranges[class_idx]
                
                if age_format == "single_value":
                    # Return midpoint
                    parts = range_str.replace('+', '').split('-')
                    if len(parts) == 2:
                        return str(int((int(parts[0]) + int(parts[1])) / 2))
                    else:
                        return parts[0]
                elif age_format == "category":
                    # Return as age group
                    return f"age_{range_str}"
                else:  # range
                    return range_str
        
        # Default: return as string
        return str(class_idx)
    
    def age_to_category(self, age_number):
        """Convert age number to age category"""
        try:
            age_num = int(age_number)
            if 0 <= age_num <= 12:
                return "child"
            elif 13 <= age_num <= 19:
                return "teenager"
            elif 20 <= age_num <= 29:
                return "young adult"
            elif 30 <= age_num <= 49:
                return "adult"
            elif 50 <= age_num <= 64:
                return "middle aged"
            elif age_num >= 65:
                return "elderly"
            else:
                return "adult"  # fallback
        except:
            return "adult"  # fallback for invalid input
    
    def extract_age_category(self, age_string):
        """Extract numeric age from age string and convert to category"""
        try:
            # Handle different age string formats
            if age_string == "uncertain" or age_string == "unknown" or age_string == "error":
                return "unknown"
            
            # Extract numbers from the age string
            numbers = re.findall(r'\d+', str(age_string))
            
            if not numbers:
                return "unknown"
            
            # If we have a range (e.g., "20-29"), use the midpoint
            if len(numbers) >= 2:
                age_num = (int(numbers[0]) + int(numbers[1])) / 2
            else:
                age_num = int(numbers[0])
            
            # Convert to age category
            return self.age_to_category(age_num)
            
        except Exception as e:
            print(f"Error extracting age category from '{age_string}': {e}")
            return "unknown"
    
    def detect_gender_age(self, image, gender_model, age_model, confidence_threshold, 
                         age_format, manual_gender_model="", manual_age_model=""):
        """Main function to detect both gender and age from image"""
        try:
            # Handle model selection
            if gender_model == "Manual gender model" and manual_gender_model.strip():
                actual_gender_model = manual_gender_model.strip()
            else:
                actual_gender_model = gender_model
            
            if age_model == "Manual age model" and manual_age_model.strip():
                actual_age_model = manual_age_model.strip()
            else:
                actual_age_model = age_model
            
            # Validate models
            if not actual_gender_model or actual_gender_model == "Manual gender model":
                return ("error", 0.0, "error", 0.0, "unknown", "No valid gender model specified")
            
            if not actual_age_model or actual_age_model == "Manual age model":
                return ("error", 0.0, "error", 0.0, "unknown", "No valid age model specified")
            
            # Load models if needed
            self.load_gender_model(actual_gender_model)
            self.load_age_model(actual_age_model)
            
            # Convert ComfyUI image tensor to PIL Image
            pil_image = self.tensor_to_pil(image)
            
            # Classify gender
            gender, gender_confidence = self.classify_gender(pil_image, actual_gender_model)
            
            # Classify age
            age, age_confidence = self.classify_age(pil_image, actual_age_model, age_format)
            
            # Apply confidence thresholds
            if gender_confidence < confidence_threshold:
                gender = "uncertain"
            
            if age_confidence < confidence_threshold:
                age = "uncertain"
                age_category = "unknown"
            else:
                # Calculate age category from the detected age
                age_category = self.extract_age_category(age)
            
            # Create debug info
            debug_info = (f"Gender: {actual_gender_model} (conf: {gender_confidence:.3f}), "
                         f"Age: {actual_age_model} (conf: {age_confidence:.3f})")
            
            return (gender, gender_confidence, age, age_confidence, age_category, debug_info)
            
        except Exception as e:
            error_msg = f"Error in detection: {str(e)}"
            print(error_msg)
            return ("error", 0.0, "error", 0.0, "unknown", error_msg)