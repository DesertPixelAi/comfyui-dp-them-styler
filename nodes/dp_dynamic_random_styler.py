import random
import os
import re

class DP_Dynamic_Random_Styler:
    """
    Dynamic Random Styler for ComfyUI
    
    Features:
    - Dynamically loads themes from data/ folder
    - Each theme has its own base_prompt.txt template with placeholders
    - Supports numeric age input (can be connected from DP_Gender_Age_Detector)
    - Automatically converts age to category: child, teenager, young adult, adult, middle aged, elderly
    - Global placeholders: {gender}, {age}, {age_category}
    - Theme-specific placeholders: Any .txt file in theme folder becomes a {placeholder}
    """
    def __init__(self):
        # Get the parent directory since we're now in nodes/ subfolder
        self.node_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.node_dir, "data")
        
        # Cache for themes and their data
        self.themes_cache = {}
        self.last_themes_check = 0
        
        # Initialize themes
        self.available_themes = self.get_available_themes()
        
        # Add gender list (works for all themes)
        self.gender = ["male", "female"]
        
        # Add age categories (works for all themes)
        self.age_categories = ["young adult", "adult", "middle aged", "elderly", "teenager", "child"]
        
        # Store last fixed selections per theme
        self.fixed_selections = {}
        self.fixed_global_selections = {}  # For gender (age is now direct input)
        self.last_seed = 0

    def get_available_themes(self):
        """Dynamically load available themes from data folder"""
        try:
            if not os.path.exists(self.data_dir):
                print(f"Data directory not found: {self.data_dir}")
                return ["default"]
            
            themes = []
            for item in os.listdir(self.data_dir):
                item_path = os.path.join(self.data_dir, item)
                if os.path.isdir(item_path):
                    themes.append(item)
            
            if not themes:
                print("No theme folders found in data directory")
                return ["default"]  # Fallback
            
            themes.sort()  # Sort alphabetically for consistent order
            print(f"Found themes: {themes}")
            return themes
            
        except Exception as e:
            print(f"Error loading themes: {e}")
            return ["default"]

    def refresh_themes(self):
        """Refresh the list of available themes"""
        current_time = os.path.getmtime(self.data_dir) if os.path.exists(self.data_dir) else 0
        if current_time > self.last_themes_check:
            self.available_themes = self.get_available_themes()
            self.last_themes_check = current_time
        return self.available_themes

    @classmethod
    def get_themes_static(cls):
        """Static method to get themes without creating instance"""
        try:
            # Get the correct path - go up one level from nodes/ folder
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            node_dir = os.path.dirname(current_file_dir)
            data_dir = os.path.join(node_dir, "data")
            
            if not os.path.exists(data_dir):
                print(f"Data directory not found: {data_dir}")
                return ["default"]
            
            themes = []
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path):
                    themes.append(item)
            
            if not themes:
                print("No theme folders found in data directory")
                return ["default"]
            
            themes.sort()
            print(f"Found themes: {themes}")
            return themes
            
        except Exception as e:
            print(f"Error loading themes: {e}")
            return ["default"]

    @classmethod
    def INPUT_TYPES(cls):
        themes = cls.get_themes_static()
        
        return {
            "required": {
                "theme": (themes, {"default": themes[0] if themes else "default"}),
                "generation_mode": (["fixed", "randomize"], {"default": "randomize"}),
                "gender_selection": (["male", "female", "random"], {"default": "random"}),
            },
            "optional": {
                "age": ("INT", {
                    "default": 25,
                    "min": 0,
                    "max": 120,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative", "theme_info", "selected_theme")
    FUNCTION = "generate"
    CATEGORY = "DP/text"

    def load_file(self, path):
        """Load text file and return clean lines"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                # Skip comments, empty lines, and lines starting with '/' or '#'
                return [line.strip() for line in f 
                       if line.strip() 
                       and not line.strip().startswith(('#', '/'))
                       and not line.strip().startswith('/*')]
        except Exception as e:
            print(f"Error loading file {path}: {e}")
            return []

    def load_theme_data(self, theme):
        """Load all data files for a specific theme"""
        if theme in self.themes_cache:
            return self.themes_cache[theme]
        
        theme_dir = os.path.join(self.data_dir, theme)
        if not os.path.exists(theme_dir):
            print(f"Theme directory not found: {theme_dir}")
            return {}
        
        theme_data = {}
        
        # Load all .txt files in the theme directory
        try:
            for filename in os.listdir(theme_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(theme_dir, filename)
                    file_key = filename[:-4]  # Remove .txt extension
                    theme_data[file_key] = self.load_file(file_path)
                    
        except Exception as e:
            print(f"Error loading theme data for {theme}: {e}")
        
        # Cache the theme data
        self.themes_cache[theme] = theme_data
        return theme_data

    def generate(self, theme, generation_mode, gender_selection, age=25):
        """
        Generate prompt based on selected theme and parameters
        
        Args:
            theme: Theme folder to use
            generation_mode: "fixed" or "randomize"
            gender_selection: "male", "female", or "random"
            age: Numeric age (can be connected from DP_Gender_Age_Detector)
        """
        
        # Set up randomization
        if generation_mode == "randomize":
            seed = random.randint(0, 0xffffffffffffffff)
            self.last_seed = seed
            # Clear fixed selections when switching to randomize
            self.fixed_selections.clear()
            self.fixed_global_selections.clear()  # Only affects gender now
        else:
            seed = self.last_seed

        rng = random.Random(seed)
        
        # Refresh themes list in case folders were added/removed
        current_themes = self.refresh_themes()
        if theme not in current_themes:
            theme = current_themes[0] if current_themes else "default"
        
        # Load theme data
        theme_data = self.load_theme_data(theme)
        
        if not theme_data:
            error_msg = f"Error: No data found for theme '{theme}'. Data dir: {self.data_dir}. Please check if the theme folder exists and contains .txt files."
            return (error_msg, "", error_msg, theme)
        
        # Load base prompt template
        if 'base_prompt' not in theme_data or not theme_data['base_prompt']:
            error_msg = f"Error: No base_prompt.txt found for theme '{theme}'. Please add a base_prompt.txt file with placeholder template."
            return (error_msg, "", error_msg, theme)
        
        base_template = theme_data['base_prompt'][0]  # First line is the template
        
        # Load negative prompt template (optional)
        negative_template = ""
        if 'negative' in theme_data and theme_data['negative']:
            negative_template = theme_data['negative'][0]  # First line is the template
        
        # Handle gender selection
        if gender_selection == "random":
            if generation_mode == "fixed":
                # For fixed mode, store and reuse global selections
                if 'gender' not in self.fixed_global_selections:
                    self.fixed_global_selections['gender'] = rng.choice(self.gender)
                selected_gender = self.fixed_global_selections['gender']
            else:
                selected_gender = rng.choice(self.gender)
        else:
            selected_gender = gender_selection
        
        # Handle age - convert number to category
        selected_age = str(age)  # Keep exact age number
        selected_age_category = self.age_to_category(age)  # Convert to category
        
        # Replace placeholders in the template
        final_prompt = base_template
        
        # Replace {gender}, {age} and {age_category} first
        final_prompt = final_prompt.replace('{gender}', selected_gender)
        final_prompt = final_prompt.replace('{age}', selected_age)
        final_prompt = final_prompt.replace('{age_category}', selected_age_category)
        
        # Process negative prompt template
        final_negative = ""
        if negative_template:
            final_negative = negative_template
            # Replace {gender}, {age} and {age_category} first
            final_negative = final_negative.replace('{gender}', selected_gender)
            final_negative = final_negative.replace('{age}', selected_age)
            final_negative = final_negative.replace('{age_category}', selected_age_category)
        
        # Replace all other placeholders
        for file_key, file_data in theme_data.items():
            if file_key in ['base_prompt', 'negative']:  # Skip the template files themselves
                continue
                
            placeholder = f'{{{file_key}}}'
            
            # Handle all placeholders the same way - single selection
            if generation_mode == "fixed":
                # For fixed mode, store and reuse selections
                if theme not in self.fixed_selections:
                    self.fixed_selections[theme] = {}
                if file_key not in self.fixed_selections[theme]:
                    self.fixed_selections[theme][file_key] = rng.choice(file_data)
                selected_value = self.fixed_selections[theme][file_key]
            else:
                selected_value = rng.choice(file_data)
            
            # Replace in positive prompt
            if placeholder in final_prompt and file_data:
                final_prompt = final_prompt.replace(placeholder, selected_value)
            
            # Replace in negative prompt
            if placeholder in final_negative and file_data:
                final_negative = final_negative.replace(placeholder, selected_value)
        
        # Clean up any remaining unreplaced placeholders
        import re
        unreplaced = re.findall(r'\{(\w+)\}', final_prompt)
        if unreplaced:
            print(f"Warning: Unreplaced placeholders in {theme}: {unreplaced}")
        
        if final_negative:
            unreplaced_negative = re.findall(r'\{(\w+)\}', final_negative)
            if unreplaced_negative:
                print(f"Warning: Unreplaced placeholders in {theme} negative: {unreplaced_negative}")
        
        # Clean the prompts (remove multiple spaces, etc.)
        final_prompt = self.clean_prompt(final_prompt)
        if final_negative:
            final_negative = self.clean_prompt(final_negative)
        
        # Generate preview info about the theme
        preview_info = self.generate_preview_info(theme, theme_data)
        
        return (final_prompt, final_negative, preview_info, theme)
    
    def generate_preview_info(self, theme, theme_data):
        """Generate preview information about the theme folder"""
        info_lines = []
        info_lines.append(f"=== THEME: {theme.upper()} ===")
        info_lines.append(f"Total files: {len(theme_data)}")
        info_lines.append("")
        
        # Show info for each file
        for file_key, file_data in sorted(theme_data.items()):
            if file_key == 'base_prompt':
                # Count placeholders in base prompt template
                template_text = file_data[0] if file_data else ''
                placeholder_pattern = r'{\w+}'
                placeholder_count = len(re.findall(placeholder_pattern, template_text))
                info_lines.append(f"📝 {file_key}: Template with {placeholder_count} placeholders")
            elif file_key == 'negative':
                # Count placeholders in negative prompt template
                template_text = file_data[0] if file_data else ''
                placeholder_pattern = r'{\w+}'
                placeholder_count = len(re.findall(placeholder_pattern, template_text))
                info_lines.append(f"❌ {file_key}: Template with {placeholder_count} placeholders")
            else:
                count = len(file_data)
                sample = file_data[0] if file_data else "N/A"
                # Truncate long samples
                if len(sample) > 60:
                    sample = sample[:57] + "..."
                info_lines.append(f"📂 {file_key}: {count} items | Sample: {sample}")
        
        info_lines.append("")
        info_lines.append(f"Data directory: {self.data_dir}")
        
        return "\n".join(info_lines)

    def clean_prompt(self, prompt):
        """Clean and format the prompt"""
        # Remove multiple spaces
        cleaned = ' '.join(prompt.split())
        
        # Remove spaces before punctuation
        cleaned = re.sub(r'\s+([,.!?])', r'\1', cleaned)
        
        # Ensure single space after punctuation
        cleaned = re.sub(r'([,.!?])\s*', r'\1 ', cleaned)
        
        # Remove any trailing/leading whitespace
        cleaned = cleaned.strip()
        
        return cleaned

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

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN") 

 