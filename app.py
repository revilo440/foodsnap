import modal
import json
import hashlib
import time
from typing import Dict, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
from collections import OrderedDict
import io

# Create Modal app
app = modal.App("foodsnap")

# Define secrets for Hugging Face access
secrets = [modal.Secret.from_name("huggingface-secret")]

# Create Modal Volume for caching
cache_volume = modal.Volume.from_name("foodsnap-cache", create_if_missing=True)
CACHE_DIR = "/cache"

# Modal Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "transformers==4.44.2",
        "accelerate==0.25.0",
        "pillow==10.1.0",
        "fastapi==0.104.1",
        "pydantic==2.5.2",
        "python-multipart==0.0.6",
        "numpy<2.0",
        "sentencepiece==0.1.99",
        "protobuf==4.25.1",
    )
)

# GPU configuration
GPU_CONFIG = "A100"

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    secrets=secrets,
    volumes={CACHE_DIR: cache_volume},
    scaledown_window=300,
    timeout=600,
    min_containers=1,
)
class FoodSnapPipeline:
    """Main pipeline for food image analysis using BLIP + Llama."""
    
    device: str = "cuda"
    blip_model = None
    blip_processor = None
    llm = None
    llm_tokenizer = None
    cache = None
    cache_stats = {"hits": 0, "misses": 0}
        
    @modal.enter()
    def setup(self):
        """Load models on container startup."""
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("Loading BLIP model...")
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float16
        ).to(self.device)
        self.blip_model.eval()
        print("Successfully loaded BLIP model")
        self.vision_model_name = "blip-base"
        
        print("Loading optimized LLM model...")
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        
        # Load Llama 3.2 with optimizations for speed
        import os
        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=os.environ.get("HF_TOKEN")
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=False,
                token=os.environ.get("HF_TOKEN"),
                trust_remote_code=True
            )
            
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                
            print(f"Successfully loaded {model_name} with float16 optimization")
            print(f"Model device: {self.llm.device}")
            print(f"Model dtype: {self.llm.dtype}")
            self.model_name = "llama-3.2-3b-instruct"
            
        except Exception as e:
            print(f"Failed to load Llama 3.2: {e}")
            print("Falling back to Flan-T5-large...")
            
            # Fallback to Flan-T5
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            fallback_model = "google/flan-t5-large"
            
            self.llm_tokenizer = T5Tokenizer.from_pretrained(fallback_model)
            self.llm = T5ForConditionalGeneration.from_pretrained(
                fallback_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print(f"Successfully loaded fallback model: {fallback_model}")
            self.model_name = "flan-t5-large"
        
        print("Loading cache...")
        self._load_cache()
        print("Models loaded successfully!")
        
    def _load_cache(self):
        """Load cache from Modal Volume."""
        cache_file = Path(CACHE_DIR) / "foodsnap_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    # Convert to OrderedDict for LRU behavior
                    self.cache = OrderedDict(cache_data.get("entries", {}))
                    self.cache_stats = cache_data.get("stats", {"hits": 0, "misses": 0})
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.cache = OrderedDict()
        else:
            self.cache = OrderedDict()
            
    def _save_cache(self):
        """Save cache to Modal Volume."""
        cache_file = Path(CACHE_DIR) / "foodsnap_cache.json"
        try:
            # Limit cache size to 100 entries (LRU)
            if len(self.cache) > 100:
                # Remove oldest entries
                for _ in range(len(self.cache) - 100):
                    self.cache.popitem(last=False)
            
            cache_data = {
                "entries": dict(self.cache),
                "stats": self.cache_stats,
                "updated_at": datetime.now().isoformat()
            }
            
            # Ensure directory exists
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
            
    def _get_cache_key(self, image_bytes: bytes) -> str:
        """Generate cache key from image bytes."""
        return hashlib.md5(image_bytes).hexdigest()
        
    def _should_cache(self, result: Dict) -> bool:
        """Determine if result should be cached based on quality."""
        # Cache if confidence is high and has meaningful content
        return (
            result.get("confidence", 0) >= 0.8 and
            len(result.get("ingredients", [])) > 0 and
            result.get("recipe", {}).get("instructions", [])
        )
        
    def generate_caption(self, image) -> str:
        """Generate caption using BLIP with single, clean prompt."""
        import torch
        
        # Use single prompt to avoid repetition issues
        inputs = self.blip_processor(
            image, 
            text="a photo of", 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            out = self.blip_model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=3,
                early_stopping=True,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.2
            )
        
        caption = self.blip_processor.decode(
            out[0], 
            skip_special_tokens=True
        )
        
        # Clean up caption - remove prompt and deduplicate words
        caption = caption.replace("a photo of", "").strip()
        
        # Simple deduplication to fix repetition issues
        words = caption.split()
        cleaned_words = []
        for word in words:
            # Don't add if it's a duplicate of the last word
            if not cleaned_words or word.lower() != cleaned_words[-1].lower():
                cleaned_words.append(word)
        
        return " ".join(cleaned_words)
        
    def extract_food_info_with_llm(self, caption: str) -> Dict:
        """Use LLM to extract structured information from caption."""
        import torch
        import re
        
        # Debug: Log which model path we're taking
        print(f"Using model: {self.model_name}")
        
        # Check which model we're using and adapt accordingly
        if self.model_name == "flan-t5-large":
            print("Taking Flan-T5 path")
            return self._extract_with_flan_t5(caption)
        else:
            print("Taking Llama optimized path")
            return self._extract_with_llama_optimized(caption)
    
    def _extract_with_flan_t5(self, caption: str) -> Dict:
        """Extract food info using Flan-T5 model."""
        import torch
        
        # T5 prompt - simpler and more direct
        prompt = f"""Given this food description: '{caption}'
        
Generate a JSON with these fields:
        - dish_name: the specific name of the dish
        - cuisine: the type of cuisine
        - confidence: a score from 0 to 1
        - description: a brief description
        - ingredients: list of ingredients with amounts
        - recipe: cooking instructions
        
Output only valid JSON:"""
        
        # Tokenize and generate
        inputs = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs.input_ids,
                max_new_tokens=400,
                temperature=0.3,
                do_sample=True,
                top_p=0.9
            )
        
        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n=== T5 RESPONSE ===\n{response[:300]}...\n=== END ===\n")
        
        # Try to parse response
        return self._parse_llm_response(response, caption)
    
    def _extract_with_llama_optimized(self, caption: str) -> Dict:
        """Extract food info using optimized Llama model."""
        import torch
        
        # More detailed, structured prompt for better results
        system_prompt = """You are a professional food analysis expert. Analyze food images and provide detailed, accurate information in JSON format. Be specific and realistic with all details."""
        
        user_prompt = f"""Based on this food description: "{caption}"

Generate a detailed analysis of THIS SPECIFIC DISH in valid JSON format:

{{
  "dish_name": "<actual name of the dish you identified from the description>",
  "cuisine": "<appropriate cuisine type based on the dish>",
  "confidence": <confidence score between 0.1-1.0>,
  "description": "<comprehensive 2-3 sentence description of THIS dish, including appearance, textures, and preparation style>",
  "ingredients": [
    {{"name": "<ingredient name>", "amount": "<quantity>", "unit": "<measurement unit>", "optional": <true/false>}}
  ],
  "recipe": {{
    "prep_time": "<time in minutes> minutes",
    "cook_time": "<time in minutes> minutes", 
    "servings": <number of servings>,
    "difficulty": "<easy/medium/hard>",
    "instructions": [
      "<detailed step 1 with specific temperatures, times, and techniques>",
      "<detailed step 2 with specific actions and measurements>",
      "<detailed step 3 explaining exactly what to do>",
      "<detailed step 4 with clear instructions>",
      "<detailed step 5 describing the process>",
      "<additional steps as needed for completeness>"
    ]
  }},
  "nutrition": {{
    "calories": <estimated calories per serving>,
    "protein": "<grams>g",
    "carbs": "<grams>g",
    "fat": "<grams>g",
    "fiber": "<grams>g",
    "sugar": "<grams>g",
    "sodium": "<milligrams>mg"
  }},
  "allergens": ["<relevant allergens for this specific dish>"],
  "tags": ["<relevant descriptive tags>"]
}}

CRITICAL REQUIREMENTS:
1. Base ALL information on the food described in: "{caption}"
2. DO NOT copy example dishes - identify what THIS food actually is
3. Provide 5-8 detailed, complete recipe instructions
4. Calculate realistic nutrition for THIS specific dish
5. Every step must be a complete, actionable instruction
6. All values must be specific, not placeholders

Return ONLY valid JSON:"""
        
        # Format for Llama chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        input_text = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize with proper padding and attention mask
        inputs = self.llm_tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,  # Reduced from 2048
            padding=True,
            return_attention_mask=True
        ).to(self.device)
        
        # Quality-focused generation parameters with fixed config
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1200,  # Restored full length for complete responses
                temperature=0.6,      # Higher temperature for more natural responses
                do_sample=True,       # Enable sampling for quality
                num_beams=1,         # Single beam for speed
                pad_token_id=self.llm_tokenizer.pad_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id,
                repetition_penalty=1.15  # Stronger penalty to avoid repetition
            )
        
        response = self.llm_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        print(f"\n=== QUALITY LLAMA RESPONSE ===\n{response[:500]}...\n=== END ===\n")
        
        return self._parse_llm_response(response, caption)
    
    def _extract_with_llama(self, caption: str) -> Dict:
        """Extract food info using Llama model (legacy method)."""
        import torch
        
        # More focused prompt for better JSON generation
        system_prompt = """You are a food analysis expert. Analyze the given food description and provide structured information.
        
        CRITICAL: You must respond with ONLY valid JSON, no other text before or after.
        Start your response directly with { and end with }
        Do not include any explanations or comments outside the JSON."""
        
        # Simplified prompt with clear example
        user_prompt = f"""Food description: "{caption}"
        
        Generate a JSON response with this exact structure:
        {{
            "dish_name": "[specific dish name based on the description]",
            "cuisine": "[cuisine type]",
            "confidence": [0.0 to 1.0 as decimal],
            "description": "[one sentence description]",
            "ingredients": [
                {{"name": "[ingredient]", "amount": "[number]", "unit": "[unit]", "optional": false}}
            ],
            "recipe": {{
                "prep_time": "[X] minutes",
                "cook_time": "[X] minutes",
                "servings": [number],
                "difficulty": "[easy/medium/hard]",
                "instructions": [
                    "[detailed instruction without numbering]",
                    "[another detailed instruction without numbering]"
                ]
            }},
            "nutrition": {{
                "calories": [number],
                "protein": "[X]g",
                "carbs": "[X]g",
                "fat": "[X]g"
            }},
            "allergens": ["[allergen1]"],
            "tags": ["[tag1]", "[tag2]"]
        }}
        
        Important: Replace all [placeholders] with actual values based on the food description.
        Be specific with dish names and include realistic ingredients and instructions."""
        
        # Format for Llama chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Apply chat template
        input_text = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.llm_tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate response with lower temperature for more consistent JSON
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs.input_ids,
                max_new_tokens=800,
                temperature=0.3,  # Lower temperature for more deterministic output
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.pad_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.llm_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Log raw response for debugging
        print(f"\n=== LLAMA RESPONSE ===\n{response[:500]}...\n=== END ===\n")
        
        return self._parse_llm_response(response, caption)
    
    def _parse_llm_response(self, response: str, caption: str) -> Dict:
        """Parse LLM response and extract JSON."""
        import re
        import json
        
        # Extract and parse JSON from response
        try:
            # Clean response - remove any markdown code blocks
            cleaned = response.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            
            # Find JSON boundaries more robustly
            json_start = cleaned.find('{')
            json_end = cleaned.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned[json_start:json_end]
                
                # Try to fix common JSON issues
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                
                parsed = json.loads(json_str)
                print(f"Successfully parsed JSON with dish_name: {parsed.get('dish_name', 'N/A')}")
                
                # Validate and enhance the response
                return self._validate_and_enhance_response(parsed, caption)
            else:
                print(f"No valid JSON found in response. Response starts with: {cleaned[:100]}")
                # Try to extract key information even if JSON parsing fails
                return self._construct_from_text_response(cleaned, caption)
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Attempted to parse: {json_str[:200] if 'json_str' in locals() else 'N/A'}")
            # Try text extraction as fallback
            return self._construct_from_text_response(response, caption)
        except Exception as e:
            print(f"Unexpected error parsing LLM response: {e}")
            return self._get_fallback_response(caption)
    
    def _validate_and_enhance_response(self, parsed: Dict, caption: str) -> Dict:
        """Validate and enhance the parsed response from T5."""
        # Ensure required fields exist with defaults
        validated = {
            "dish_name": parsed.get("dish_name", "Food Dish"),
            "cuisine": parsed.get("cuisine", "unknown"),
            "confidence": float(parsed.get("confidence", 0.7)),
            "description": parsed.get("description", caption[:200]),
            "ingredients": parsed.get("ingredients", []),
            "recipe": parsed.get("recipe", {}),
            "nutrition": parsed.get("nutrition", {}),
            "allergens": parsed.get("allergens", []),
            "tags": parsed.get("tags", [])
        }
        
        # Validate ingredients format
        if not isinstance(validated["ingredients"], list):
            validated["ingredients"] = []
        
        # Ensure each ingredient has required fields
        valid_ingredients = []
        for ing in validated["ingredients"]:
            if isinstance(ing, dict):
                valid_ingredients.append({
                    "name": ing.get("name", "ingredient"),
                    "amount": str(ing.get("amount", "1")),
                    "unit": ing.get("unit", "piece"),
                    "optional": bool(ing.get("optional", False))
                })
            elif isinstance(ing, str):
                # Handle simple string ingredients
                valid_ingredients.append({
                    "name": ing,
                    "amount": "1",
                    "unit": "piece",
                    "optional": False
                })
        validated["ingredients"] = valid_ingredients
        
        # Validate recipe format
        if not isinstance(validated["recipe"], dict):
            validated["recipe"] = {}
        
        recipe_defaults = {
            "prep_time": "15 minutes",
            "cook_time": "20 minutes", 
            "servings": 4,
            "difficulty": "medium",
            "instructions": []
        }
        for key, default in recipe_defaults.items():
            if key not in validated["recipe"]:
                validated["recipe"][key] = default
        
        # Ensure instructions is a list
        if not isinstance(validated["recipe"]["instructions"], list):
            if isinstance(validated["recipe"]["instructions"], str):
                validated["recipe"]["instructions"] = [validated["recipe"]["instructions"]]
            else:
                validated["recipe"]["instructions"] = ["Instructions not available"]
        
        return validated
    
    def _construct_from_text_response(self, response: str, caption: str) -> Dict:
        """Construct a response when JSON parsing fails by extracting from text."""
        import re
        
        # Try to extract dish name using regex patterns
        dish_name = "Unknown Dish"
        dish_patterns = [
            r'"dish_name"\s*:\s*"([^"]+)"',
            r'dish_name[":]*\s*([A-Za-z\s]+)',
            r'\b(pasta|pizza|burger|salad|soup|sandwich|rice|noodles|chicken|beef|fish|pork)\b'
        ]
        
        for pattern in dish_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                dish_name = match.group(1).strip()
                if len(dish_name) > 2:  # Valid name found
                    break
        
        # If still generic, try to extract from caption
        if dish_name == "Unknown Dish" and caption:
            # Look for food words in caption
            food_words = re.findall(r'\b(\w+(?:pasta|pizza|burger|salad|soup|sandwich|rice|noodles|chicken|beef|fish|pork|curry|steak|taco|sushi|ramen|pho|pad thai))\b', caption, re.IGNORECASE)
            if food_words:
                dish_name = food_words[0].title()
        
        # Extract cuisine if possible
        cuisine = "International"
        cuisine_patterns = [
            r'"cuisine"\s*:\s*"([^"]+)"',
            r'\b(italian|chinese|japanese|thai|indian|mexican|french|american|vietnamese|korean)\b'
        ]
        for pattern in cuisine_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                cuisine = match.group(1).strip().title()
                break
        
        # Generate a reasonable confidence based on how much we extracted
        confidence = 0.6 if dish_name != "Unknown Dish" else 0.4
        
        # Build response with extracted info
        return {
            "dish_name": dish_name,
            "cuisine": cuisine,
            "confidence": confidence,
            "description": f"A {cuisine.lower()} dish featuring {dish_name.lower()}",
            "ingredients": [
                {"name": "Main ingredient", "amount": "200", "unit": "g", "optional": False},
                {"name": "Seasoning", "amount": "to taste", "unit": "", "optional": False}
            ],
            "recipe": {
                "prep_time": "20 minutes",
                "cook_time": "30 minutes",
                "servings": 4,
                "difficulty": "medium",
                "instructions": [
                    "Prepare all ingredients",
                    "Cook according to traditional method",
                    "Season to taste and serve hot"
                ]
            },
            "nutrition": {
                "calories": 350,
                "protein": "15g",
                "carbs": "45g", 
                "fat": "12g",
                "fiber": "3g",
                "sugar": "5g",
                "sodium": "600mg"
            },
            "allergens": [],
            "tags": [cuisine.lower(), "main course"]
        }
            
    def _get_fallback_response(self, caption: str) -> Dict:
        """Generate fallback response when LLM fails completely."""
        # Try to extract something meaningful from caption
        import re
        
        # Look for food-related words in caption
        food_words = re.findall(r'\b(\w+(?:pasta|pizza|burger|salad|soup|sandwich|rice|noodles|chicken|beef|fish|vegetable|fruit|dessert|cake|bread))\b', caption, re.IGNORECASE)
        dish_name = food_words[0].title() if food_words else "Unidentified Dish"
        
        return {
            "dish_name": dish_name,
            "cuisine": "International",
            "confidence": 0.3,
            "description": caption[:200] if caption else "Unable to fully analyze this food image",
            "ingredients": [
                {"name": "Primary ingredient", "amount": "varies", "unit": "", "optional": False}
            ],
            "recipe": {
                "prep_time": "30 minutes",
                "cook_time": "30 minutes",
                "servings": 4,
                "difficulty": "medium",
                "instructions": [
                    "Detailed recipe analysis unavailable",
                    "Please try uploading a clearer image"
                ]
            },
            "nutrition": {
                "calories": 300,
                "protein": "10g",
                "carbs": "40g",
                "fat": "10g",
                "fiber": "2g",
                "sugar": "5g",
                "sodium": "500mg"
            },
            "allergens": ["May contain common allergens"],
            "tags": ["needs-review"],
            "note": "Analysis confidence is low - results may be inaccurate"
        }
        
    @modal.method()
    def analyze(self, image_bytes: bytes) -> Dict:
        """Main analysis method for food images."""
        from PIL import Image
        import torch
        
        start_time = time.time()
        timings = {}
        
        # Check cache
        cache_key = self._get_cache_key(image_bytes)
        if self.cache and cache_key in self.cache:
            # Move to end (LRU)
            self.cache.move_to_end(cache_key)
            self.cache_stats["hits"] += 1
            cached_result = self.cache[cache_key].copy()
            cached_result["cached"] = True
            cached_result["cache_stats"] = self.cache_stats.copy()
            return cached_result
            
        self.cache_stats["misses"] += 1
        
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Stage 1: Caption generation
            caption_start = time.time()
            caption = self.generate_caption(image)
            timings["caption_ms"] = int((time.time() - caption_start) * 1000)
            
            # Clear GPU memory after caption generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Stage 2: LLM processing
            llm_start = time.time()
            food_info = self.extract_food_info_with_llm(caption)
            timings["llm_ms"] = int((time.time() - llm_start) * 1000)
            
            # Stage 3: Response formatting
            format_start = time.time()
            result = self._format_response(food_info, caption, timings)
            timings["format_ms"] = int((time.time() - format_start) * 1000)
            
            # Add total timing
            timings["total_ms"] = int((time.time() - start_time) * 1000)
            result["timings_ms"] = timings
            
            # Cache if high quality
            if self._should_cache(result):
                self.cache[cache_key] = result.copy()
                self._save_cache()
                
            result["cached"] = False
            result["cache_stats"] = self.cache_stats.copy()
            
            return result
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "error": str(e),
                "dish_name": "Error",
                "confidence": 0.0,
                "description": "Failed to analyze image",
                "ingredients": [],
                "recipe": {},
                "nutrition": {},
                "allergens": [],
                "tags": [],
                "cached": False,
                "cache_stats": self.cache_stats.copy(),
                "timings_ms": {"total_ms": int((time.time() - start_time) * 1000)}
            }
            
    def _format_response(self, food_info: Dict, caption: str, timings: Dict) -> Dict:
        """Format and validate the response."""
        # Ensure all required fields exist
        response = {
            "dish_name": food_info.get("dish_name", "Unknown Dish"),
            "cuisine": food_info.get("cuisine", "unknown"),
            "confidence": float(food_info.get("confidence", 0.5)),
            "description": food_info.get("description", caption[:200]),
            "ingredients": food_info.get("ingredients", []),
            "recipe": food_info.get("recipe", {}),
            "nutrition": food_info.get("nutrition", {}),
            "allergens": food_info.get("allergens", []),
            "tags": food_info.get("tags", []),
            "caption": caption,
            "pipeline": "caption-to-recipe",
            "models": {
                "vision": "blip-base",
                "llm": getattr(self, 'model_name', 'unknown')
            },
            "additional_info": {
                "dish_type": food_info.get("dish_type", "unknown"),
                "cooking_method": food_info.get("cooking_method", "unknown")
            }
        }
        
        # Validate and clean ingredients
        valid_ingredients = []
        for ing in response["ingredients"]:
            if isinstance(ing, dict) and "name" in ing:
                valid_ingredients.append({
                    "name": ing.get("name", ""),
                    "amount": ing.get("amount", ""),
                    "unit": ing.get("unit", ""),
                    "optional": ing.get("optional", False)
                })
        response["ingredients"] = valid_ingredients
        
        # Ensure recipe has required fields
        if not response["recipe"]:
            response["recipe"] = {}
        response["recipe"].setdefault("prep_time", "unknown")
        response["recipe"].setdefault("cook_time", "unknown")
        response["recipe"].setdefault("servings", 1)
        response["recipe"].setdefault("difficulty", "medium")
        response["recipe"].setdefault("instructions", [])
        
        # Ensure nutrition has required fields
        if not response["nutrition"]:
            response["nutrition"] = {}
        nutrition_defaults = {
            "calories": 0,
            "protein": "0g",
            "carbs": "0g",
            "fat": "0g",
            "fiber": "0g",
            "sugar": "0g",
            "sodium": "0mg"
        }
        for key, default in nutrition_defaults.items():
            response["nutrition"].setdefault(key, default)
            
        return response

# FastAPI endpoints
@app.function(
    image=image,
    secrets=secrets,
    volumes={CACHE_DIR: cache_volume},
    min_containers=1,
    scaledown_window=300,
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    """FastAPI app following Step 3 working pattern."""
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI()
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.post("/analyze")
    async def analyze(file: UploadFile = File(...)):
        """Main API endpoint for food analysis."""
        try:
            # Read file contents
            contents = await file.read()
            
            # Get or create pipeline instance
            pipeline = FoodSnapPipeline()
            
            # Analyze image
            result = pipeline.analyze.remote(contents)
            
            return JSONResponse(content=result)
            
        except Exception as e:
            return JSONResponse(
                content={"error": f"Failed to analyze image: {str(e)}"},
                status_code=500
            )
    
    return app



@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def cache_stats():
    """Get cache statistics."""
    from fastapi.responses import JSONResponse
    
    try:
        cache_file = Path(CACHE_DIR) / "foodsnap_cache.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                stats = cache_data.get("stats", {})
                stats["entries"] = len(cache_data.get("entries", {}))
                stats["updated_at"] = cache_data.get("updated_at", "unknown")
                return JSONResponse(content=stats)
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
        
    return JSONResponse(content={"entries": 0, "hits": 0, "misses": 0})

@app.function(image=image, volumes={CACHE_DIR: cache_volume})
@modal.fastapi_endpoint(method="POST")
def cache_cleanup():
    """Clear the cache."""
    from fastapi.responses import JSONResponse
    
    try:
        cache_file = Path(CACHE_DIR) / "foodsnap_cache.json"
        if cache_file.exists():
            cache_file.unlink()
        return JSONResponse(content={"status": "Cache cleared successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
