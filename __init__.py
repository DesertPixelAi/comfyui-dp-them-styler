from .nodes.dp_dynamic_random_styler import DP_Dynamic_Random_Styler
from .nodes.dp_gender_age_detector import DP_Gender_Age_Detector
from .nodes.dp_add_logo_banner import DP_Add_Logo_Banner
from .nodes.dp_advanced_sampler_modified import DP_Advanced_Sampler

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "DP_Dynamic_Random_Styler": DP_Dynamic_Random_Styler,
    "DP_Gender_Age_Detector": DP_Gender_Age_Detector,
    "DP_Add_Logo_Banner": DP_Add_Logo_Banner,
    "DP_Advanced_Sampler_Modified": DP_Advanced_Sampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DP_Dynamic_Random_Styler": "DP Dynamic Random Styler",
    "DP_Gender_Age_Detector": "DP Gender Age Detector",
    "DP_Add_Logo_Banner": "DP Add Logo Banner",
    "DP_Advanced_Sampler_Modified": "DP Advanced Sampler Modified"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
