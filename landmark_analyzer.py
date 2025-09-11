
# ----------------------------------------------------------------------------------------------------
# --------------------------------------- OPTIMIZED OPENAI VERSION WITH TIMING ---------------------------------------
# ----------------------------------------------------------------------------------------------------

import os
import time
from contextlib import contextmanager
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
from functools import lru_cache
from PIL import Image
import io
from geo import get_location_from_coords

# Load environment variables once
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Global model instance (reused across calls)
_llm_instance = None

# Timer utilities for performance tracking
class Timer:
    """Context manager and utility class for timing operations."""
    
    def __init__(self, operation_name="Operation", verbose=True):
        self.operation_name = operation_name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        if self.verbose:
            print(f"⏱️  Starting: {self.operation_name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        if self.verbose:
            print(f"✅ Completed: {self.operation_name} - {self.duration:.3f}s")
    
    def get_duration(self):
        """Get the duration of the last timed operation."""
        return self.duration

class PerformanceTracker:
    """Track and report performance metrics for landmark processing."""
    
    def __init__(self):
        self.metrics = {}
        self.total_start_time = None
    
    def start_total_timer(self):
        """Start the overall timing."""
        self.total_start_time = time.perf_counter()
    
    def add_metric(self, name, duration):
        """Add a performance metric."""
        self.metrics[name] = duration
    
    def print_summary(self, show_details=True):
        """Print a comprehensive performance summary."""
        if self.total_start_time:
            total_time = time.perf_counter() - self.total_start_time
            self.metrics['Total Processing Time'] = total_time
        
        print("\n" + "="*60)
        print("🚀 PERFORMANCE SUMMARY")
        print("="*60)
        
        if show_details and len(self.metrics) > 1:
            # Show individual timings
            for metric_name, duration in self.metrics.items():
                if metric_name != 'Total Processing Time':
                    percentage = (duration / self.metrics.get('Total Processing Time', 1)) * 100
                    print(f"  {metric_name:<25}: {duration:>8.3f}s ({percentage:>5.1f}%)")
            print("-" * 60)
        
        # Show total
        total = self.metrics.get('Total Processing Time', 0)
        print(f"  {'TOTAL TIME':<25}: {total:>8.3f}s")
        
        # Performance insights
        if show_details and len(self.metrics) > 3:
            self._print_insights()
        
        print("="*60)
    
    def _print_insights(self):
        """Print performance insights and recommendations."""
        print("\n💡 PERFORMANCE INSIGHTS:")
        
        api_time = self.metrics.get('API Request', 0)
        image_time = self.metrics.get('Image Processing', 0)
        geo_time = self.metrics.get('Geocoding', 0)
        total_time = self.metrics.get('Total Processing Time', 1)
        
        # Identify bottlenecks
        if api_time > total_time * 0.6:
            print("  • API request is the main bottleneck (>60% of time)")
            print("    → Consider using a faster model or reducing prompt length")
        
        if image_time > total_time * 0.2:
            print("  • Image processing is slow (>20% of time)")
            print("    → Try reducing image size or quality further")
        
        if geo_time > total_time * 0.3:
            print("  • Geocoding is taking significant time (>30%)")
            print("    → Consider caching coordinates or using a faster geocoding service")
        
        # Speed recommendations
        if total_time < 3:
            print("  • ⚡ Excellent performance! (<3s total)")
        elif total_time < 6:
            print("  • ✅ Good performance (3-6s total)")
        elif total_time < 10:
            print("  • ⚠️  Moderate performance (6-10s total)")
        else:
            print("  • 🐌 Slow performance (>10s total) - optimization needed")

def get_llm_instance(temperature=0.3):
    """Get or create a singleton LLM instance to avoid re-initialization."""
    global _llm_instance
    
    if _llm_instance is None or _llm_instance.temperature != temperature:
        _llm_instance = ChatOpenAI(
            model="gpt-4o",  # Using faster gpt-4o instead of gpt-4.1
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY,
            max_retries=1,  # Reduce retries for faster failure
            request_timeout=30  # Set timeout to avoid hanging
        )
    return _llm_instance

def get_optimized_base64_image(image_path, max_size=(1024, 1024), quality=85, verbose=True):
    """
    Encodes and compresses image to Base64 string for faster upload.
    Reduces image size while maintaining quality for landmark recognition.
    """
    start_time = time.perf_counter()
    
    try:
        with Image.open(image_path) as img:
            original_size = img.size
            
            # Convert to RGB if necessary (removes alpha channel, reduces size)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Resize if image is larger than max_size (maintains aspect ratio)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Compress to reduce file size
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            buffer.seek(0)
            
            # Calculate compression stats
            original_bytes = os.path.getsize(image_path)
            compressed_bytes = len(buffer.getvalue())
            compression_ratio = (1 - compressed_bytes/original_bytes) * 100
            
            duration = time.perf_counter() - start_time
            if verbose:
                print(f"📸 Image optimized: {original_size} → {img.size}, "
                      f"size reduced by {compression_ratio:.1f}% ({duration:.3f}s)")
            
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
    except Exception as e:
        # Fallback to original method if optimization fails
        duration = time.perf_counter() - start_time
        if verbose:
            print(f"⚠️  Image optimization failed ({duration:.3f}s), using original: {e}")
        
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

# Cached translations (avoid dictionary lookups)
@lru_cache(maxsize=3)
def get_headings(language):
    """Cached function to get language-specific headings."""
    headings_translation = {
        "English": {
            "location": "Location", "year_completed": "Year Completed", "materials": "Materials",
            "architectural_style": "Architectural Style", "historical_overview": "Historical Overview",
            "cultural_impact": "Cultural Impact", "famous_for": "Famous For", "key_features": "Key Features",
            "primary_function": "Primary Function", "overview_and_significance": "Overview & Significance",
            "visitor_experience": "Visitor Experience", "known_for": "Known For"
        },
        "Chinese": {
            "location": "位置", "year_completed": "建成年份", "materials": "材料", "architectural_style": "建筑风格",
            "historical_overview": "历史概述", "cultural_impact": "文化影响", "famous_for": "著名原因",
            "key_features": "主要特色", "primary_function": "主要功能", "overview_and_significance": "概述与意义",
            "visitor_experience": "游客体验", "known_for": "著名原因"
        },
        "Traditional Chinese": {
            "location": "位置", "year_completed": "建成年份", "materials": "材料", "architectural_style": "建築風格",
            "historical_overview": "歷史概述", "cultural_impact": "文化影響", "famous_for": "著名原因",
            "key_features": "主要特色", "primary_function": "主要功能", "overview_and_significance": "概述與意義",
            "visitor_experience": "遊客體驗", "known_for": "著名原因"
        }
    }
    return headings_translation.get(language)

@lru_cache(maxsize=3)
def get_language_prompt(language):
    """Cached function to get language-specific prompts."""
    language_prompts = {
        "English": "Please respond in English.",
        "Chinese": "请用中文回答。",
        "Traditional Chinese": "請用繁體中文回答。"
    }
    return language_prompts.get(language)

# Streamlined prompt generation
def build_prompt(address, language, headings):
    """Build the analysis prompt more efficiently."""
    lang_prompt = get_language_prompt(language)
    
    # Shortened, more focused prompt for faster processing
    return f"""{lang_prompt}

Analyze this landmark image from {address}. Classify as Historical Landmark or General Place and use the appropriate template:

HISTORICAL LANDMARK TEMPLATE:
**[Name]**
{headings["location"]}: [City, Country]
{headings["year_completed"]}: [Year/Era]
{headings["architectural_style"]}: [Style]
{headings["historical_overview"]}: [2-3 sentences on history]
{headings["cultural_impact"]}: [2-3 sentences on significance]
{headings["famous_for"]}: [3 key points]

GENERAL PLACE TEMPLATE:
**[Name]**
{headings["location"]}: [City, Country]
{headings["primary_function"]}: [Function]
{headings["overview_and_significance"]}: [2-3 sentences]
{headings["visitor_experience"]}: [2-3 sentences]
{headings["known_for"]}: [3 key points]

Be concise. Omit unknown details."""

def process_landmark_with_timing(image_path, latitude, longitude, language="English", temperature=0.3, verbose=True):
    """
    Optimized landmark analysis with comprehensive timing and performance tracking.
    Returns: (result, metrics_dict)
    """
    # Initialize performance tracker
    tracker = PerformanceTracker()
    tracker.start_total_timer()
    
    try:
        if verbose:
            print(f"\n🏛️  Processing landmark: {os.path.basename(image_path)}")
            print(f"📍 Coordinates: ({latitude}, {longitude})")
            print(f"🌐 Language: {language}")
        
        # Step 1: Validate inputs
        with Timer("Input Validation", verbose) as validation_timer:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file '{image_path}' not found.")
            
            headings = get_headings(language)
            if not headings:
                raise ValueError(f"Unsupported language: {language}")
        
        tracker.add_metric("Input Validation", validation_timer.get_duration())

        # Step 2: Geocoding
        with Timer("Geocoding", verbose) as geo_timer:
            address = get_location_from_coords(latitude, longitude)
            if address in ["Location not found", "Geocoding error"]:
                if verbose:
                    print(f"⚠️  Warning: Could not fetch address - {address}")
                address = f"coordinates {latitude}, {longitude}"  # Fallback
        
        tracker.add_metric("Geocoding", geo_timer.get_duration())

        # Step 3: Image processing
        with Timer("Image Processing", verbose) as img_timer:
            base64_image = get_optimized_base64_image(image_path, verbose=verbose)
        
        tracker.add_metric("Image Processing", img_timer.get_duration())
        
        # Step 4: Prompt building
        with Timer("Prompt Building", verbose) as prompt_timer:
            prompt = build_prompt(address, language, headings)
        
        tracker.add_metric("Prompt Building", prompt_timer.get_duration())

        # Step 5: Model initialization
        with Timer("Model Setup", verbose) as model_timer:
            llm = get_llm_instance(temperature)
        
        tracker.add_metric("Model Setup", model_timer.get_duration())

        # Step 6: API request
        with Timer("API Request", verbose) as api_timer:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            )
            response = llm.invoke([message])
        
        tracker.add_metric("API Request", api_timer.get_duration())
        
        # Print performance summary
        if verbose:
            tracker.print_summary(show_details=True)
        
        return response.content, tracker.metrics

    except Exception as e:
        if verbose:
            print(f"❌ Error processing landmark: {str(e)}")
            tracker.print_summary(show_details=False)
        return None, tracker.metrics

# Original function signature for backward compatibility
def process_landmark(image_path, latitude, longitude, language="English", temperature=0.3):
    """
    Original function signature - returns only the result for backward compatibility.
    """
    try:
        result, _ = process_landmark_with_timing(
            image_path, latitude, longitude, language, temperature, verbose=False
        )
        return result
    except Exception as e:
        print(f"Error processing landmark: {str(e)}")
        return None

# Batch processing function for multiple landmarks
def process_landmarks_batch(landmark_data_list, language="English", temperature=0.3, verbose=True):
    """
    Process multiple landmarks efficiently with comprehensive timing.
    landmark_data_list: List of tuples (image_path, latitude, longitude)
    """
    batch_start_time = time.perf_counter()
    results = []
    all_metrics = {}
    
    if verbose:
        print(f"\n🚀 Starting batch processing of {len(landmark_data_list)} landmarks...")
    
    # Initialize model once for the entire batch
    with Timer("Batch Model Initialization", verbose) as init_timer:
        llm = get_llm_instance(temperature)
    
    for i, (image_path, lat, lng) in enumerate(landmark_data_list, 1):
        if verbose:
            print(f"\n{'='*20} LANDMARK {i}/{len(landmark_data_list)} {'='*20}")
        
        result, metrics = process_landmark_with_timing(image_path, lat, lng, language, temperature, verbose)
        results.append(result)
        all_metrics[f"landmark_{i}"] = metrics
    
    # Batch summary
    total_batch_time = time.perf_counter() - batch_start_time
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"📊 BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"  Total landmarks processed: {len(landmark_data_list)}")
        print(f"  Total batch time: {total_batch_time:.3f}s")
        print(f"  Average time per landmark: {total_batch_time/len(landmark_data_list):.3f}s")
        print(f"  Successful processes: {sum(1 for r in results if r is not None)}")
        print(f"{'='*60}")
    
    return results, all_metrics

# Utility function for quick timing of any operation
def time_operation(func, *args, operation_name=None, **kwargs):
    """Time any function call and return result with timing info."""
    name = operation_name or func.__name__
    start_time = time.perf_counter()
    
    try:
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start_time
        print(f"⏱️  {name}: {duration:.3f}s")
        return result, duration
    except Exception as e:
        duration = time.perf_counter() - start_time
        print(f"❌ {name} failed after {duration:.3f}s: {e}")
        return None, duration

if __name__ == "__main__":
    # Example usage with comprehensive timing
    print("🏛️  LANDMARK PROCESSING WITH PERFORMANCE TRACKING")
    print("="*60)
    
    # Configuration
    image_path = "images/img.jpg"  # Replace with your image path
    latitude = 21.8182
    longitude = 90.1398
    language = "English"
    
    print(f"📝 Configuration:")
    print(f"   Image: {image_path}")
    print(f"   Coordinates: ({latitude}, {longitude})")
    print(f"   Language: {language}")
    
    # Method 1: With detailed timing (recommended)
    print(f"\n🔄 Processing with detailed timing...")
    try:
        result, metrics = process_landmark_with_timing(
            image_path, latitude, longitude, 
            language=language, temperature=0, verbose=True
        )
        
        if result:
            print("\n📝 ANALYSIS RESULT:")
            print("-" * 40)
            print(result)
        else:
            print("❌ Failed to process landmark.")
            
    except Exception as e:
        print(f"❌ Error with timing version: {e}")
        
        # Method 2: Fallback to simple version
        print(f"\n🔄 Trying simple version...")
        result = process_landmark(
            image_path, latitude, longitude, 
            language=language, temperature=0
        )
        
        if result:
            print("\n📝 ANALYSIS RESULT (Simple Mode):")
            print("-" * 40)
            print(result)
        else:
            print("❌ Both methods failed.")
    
    # Performance comparison example
    print(f"\n⚡ Available functions:")
    print(f"   1. process_landmark() - Simple version (backward compatible)")
    print(f"   2. process_landmark_with_timing() - Full timing version")
    print(f"   3. process_landmarks_batch() - Batch processing with timing")
    print(f"   4. time_operation() - Time any custom function")













# # ----------------------------------------------------------------------------------------------------
# # --------------------------------------- OPENAI VERSION BELOW ---------------------------------------
# # ----------------------------------------------------------------------------------------------------

# import os
# from dotenv import load_dotenv
# from langchain_community.chat_models import ChatOpenAI  # Corrected import
# from langchain_core.messages import HumanMessage
# import base64
# from geo import get_location_from_coords

# # Load environment variables
# load_dotenv()
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY not found in environment variables.")

# # Function to encode the image to a base64 string
# def get_base64_encoded_image(image_path):
#     """Encodes a local image file to a Base64 string."""
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

# # Language-specific templates for headings
# HEADINGS_TRANSLATION = {
#     "English": {
#         "location": "Location", "year_completed": "Year Completed", "materials": "Materials",
#         "architectural_style": "Architectural Style", "historical_overview": "Historical Overview",
#         "cultural_impact": "Cultural Impact", "famous_for": "Famous For", "key_features": "Key Features",
#         "primary_function": "Primary Function", "overview_and_significance": "Overview & Significance",
#         "visitor_experience": "Visitor Experience", "known_for": "Known For"
#     },
#     "Chinese": {
#         "location": "位置", "year_completed": "建成年份", "materials": "材料", "architectural_style": "建筑风格",
#         "historical_overview": "历史概述", "cultural_impact": "文化影响", "famous_for": "著名原因",
#         "key_features": "主要特色", "primary_function": "主要功能", "overview_and_significance": "概述与意义",
#         "visitor_experience": "游客体验", "known_for": "著名原因"
#     },
#     "Traditional Chinese": {
#         "location": "位置", "year_completed": "建成年份", "materials": "材料", "architectural_style": "建築風格",
#         "historical_overview": "歷史概述", "cultural_impact": "文化影響", "famous_for": "著名原因",
#         "key_features": "主要特色", "primary_function": "主要功能", "overview_and_significance": "概述與意義",
#         "visitor_experience": "遊客體驗", "known_for": "著名原因"
#     }
# }

# # Function to fetch language prompt
# def get_language_prompt(language):
#     """Returns the appropriate prompt based on language."""
#     language_prompts = {
#         "English": "Please respond in English.",
#         "Chinese": "请用中文回答。",
#         "Traditional Chinese": "請用繁體中文回答。"
#     }
#     return language_prompts.get(language, None)

# # Function to process the landmark using LangChain
# def process_landmark(image_path, latitude, longitude, language="English", temperature=0.3):
#     """
#     Analyzes a landmark image using LangChain and the OpenAI API.
#     Additionally, fetches the address using latitude and longitude and dynamically adds it to the prompt.
#     """
#     try:
#         # Get the location address using latitude and longitude
#         address = get_location_from_coords(latitude, longitude)
#         if address in ["Location not found", "Geocoding error"]:
#             print(f"Failed to fetch address. {address}")
#             return None

#         # Validate image file existence
#         if not os.path.exists(image_path):
#             raise FileNotFoundError(f"Error: The file '{image_path}' does not exist.")

#         # Encode the image to a Base64 string
#         base64_image = get_base64_encoded_image(image_path)

#         # Validate language
#         headings = HEADINGS_TRANSLATION.get(language)
#         if not headings:
#             raise ValueError(f"Unsupported language: {language}")

#         # Fetch language-specific prompt
#         language_prompt = get_language_prompt(language)
#         if not language_prompt:
#             raise ValueError(f"Unsupported language: {language}")

#         # Construct the dynamic prompt
#         unified_prompt = f"""
# {language_prompt}

# Given image is from {address}, a popular tourist destination.

# You are an expert travel guide and architectural analyst. Your task is to analyze the provided image and generate a detailed, engaging, and factually accurate report about the prominent place or structure shown.

# CRITICAL PROCESSING INSTRUCTIONS:

# First, analyze the image to determine if the subject is a Historical Landmark (e.g., ancient temple, medieval castle, monument with deep historical significance) or a General Place of Interest (e.g., modern skyscraper, iconic bridge, famous commercial building, natural wonder, public square).

# Based on your classification, output the report using only one of the two exact templates below.

# TEMPLATE A: For a Historical Landmark
# {headings["location"]} [Full Official Landmark Name]

# {headings["location"]}: [City, Country]

# {headings["year_completed"]}: [Year or Era] (Omit if not known or not applicable)

# {headings["materials"]}: [Primary construction materials] (Omit if not discernible)

# {headings["architectural_style"]}: [Predominant architectural style] (Omit if not classified)

# {headings["historical_overview"]}:
# [A concise paragraph detailing its origin, key historical events, and significant figures involved (e.g., architects, rulers). Focus on its historical narrative.]

# {headings["cultural_impact"]}:
# [A single paragraph explaining its symbolic meaning, its influence on national/regional identity, and its role in culture, arts, or collective memory.]

# {headings["famous_for"]}:
# [1. Primary reason for global fame]

# [2. Secondary distinct reason]

# [3. Tertiary distinct reason]

# TEMPLATE B: For a General Place of Interest
# {headings["location"]} [Full Official Name of the Place/Structure]

# {headings["location"]}: [City, Country]

# {headings["year_completed"]}: [Year] (Omit if not known)

# {headings["key_features"]}: [Notable materials, engineering marvels, or design elements] (Omit if not discernible)

# {headings["primary_function"]}: [E.g., Observation Tower, Transportation Hub, Commercial Center, Public Park] (Omit if not applicable)

# {headings["overview_and_significance"]}:
# [A concise paragraph describing what it is, its primary purpose, and why it is significant to the city or field (e.g., engineering, urban planning, commerce).]

# [{headings["visitor_experience"]}]:
# [A single paragraph highlighting what visitors can see and do there, the atmosphere, and any unique experiential aspects.]

# [{headings["known_for"]}]:
# [1. Primary claim to fame]

# [2. Secondary distinct feature or fact]

# [3. Tertiary distinct feature or fact]

# GLOBAL OUTPUT RULES (Apply to both templates):

# Zero Repetition: Do not repeat any fact, figure, or description across different sections.

# Conciseness: Use clear, efficient, and engaging language. Avoid fluff and redundancy.

# Deduction: Base your analysis on visual cues from the image and your encyclopedic knowledge. Omit any numbered line (e.g., Year Completed, Materials) if the information cannot be reasonably inferred or is not applicable.

# Structure: Maintain the exact spacing, bolding, and section ordering as shown in the chosen template. Begin the response immediately with the landmark/place name.
# """

#         # Initialize the LangChain OpenAI model
#         llm = ChatOpenAI(
#             model="gpt-4.1",
#             temperature=temperature,
#             openai_api_key=OPENAI_API_KEY
#         )

#         # Create the message with text and image data
#         message = HumanMessage(
#             content=[
#                 {"type": "text", "text": unified_prompt},
#                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
#             ]
#         )

#         # Invoke the model and return the result
#         response = llm.invoke([message])
#         return response.content

#     except Exception as e:
#         print(f"Failed to process landmark: {str(e)}")
#         return None

# # Example usage
# if __name__ == "__main__":
#     image_path = "images/download.webp"  # Replace with your image path
#     latitude = 48.8534 # Replace with actual latitude
#     longitude = 2.3488  # Replace with actual longitude
#     language = "Chinese"  # Example language selection
    
#     result = process_landmark(image_path, latitude, longitude, language=language, temperature=0)
    
#     if result:
#         print("\n--- Final Result ---")
#         print(result)









# ----------------------------------------------------------------------------------------------------
# --------------------------------------- GEMINI VERSION BELOW ---------------------------------------
# ----------------------------------------------------------------------------------------------------


# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import HumanMessage
# import base64
# from geo import get_location_from_coords

# # Load environment variables
# load_dotenv()
# GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY not found in environment variables.")

# # Function to encode the image to a base64 string
# def get_base64_encoded_image(image_path):
#     """Encodes a local image file to a Base64 string."""
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

# # Language-specific templates for headings
# HEADINGS_TRANSLATION = {
#     "English": {
#         "location": "Location", "year_completed": "Year Completed", "materials": "Materials",
#         "architectural_style": "Architectural Style", "historical_overview": "Historical Overview",
#         "cultural_impact": "Cultural Impact", "famous_for": "Famous For", "key_features": "Key Features",
#         "primary_function": "Primary Function", "overview_and_significance": "Overview & Significance",
#         "visitor_experience": "Visitor Experience", "known_for": "Known For"
#     },
#     "Chinese": {
#         "location": "位置", "year_completed": "建成年份", "materials": "材料", "architectural_style": "建筑风格",
#         "historical_overview": "历史概述", "cultural_impact": "文化影响", "famous_for": "著名原因",
#         "key_features": "主要特色", "primary_function": "主要功能", "overview_and_significance": "概述与意义",
#         "visitor_experience": "游客体验", "known_for": "著名原因"
#     },
#     "Traditional Chinese": {
#         "location": "位置", "year_completed": "建成年份", "materials": "材料", "architectural_style": "建築風格",
#         "historical_overview": "歷史概述", "cultural_impact": "文化影響", "famous_for": "著名原因",
#         "key_features": "主要特色", "primary_function": "主要功能", "overview_and_significance": "概述與意義",
#         "visitor_experience": "遊客體驗", "known_for": "著名原因"
#     }
# }

# # Function to fetch language prompt
# def get_language_prompt(language):
#     """Returns the appropriate prompt based on language."""
#     language_prompts = {
#         "English": "Please respond in English.",
#         "Chinese": "请用中文回答。",
#         "Traditional Chinese": "請用繁體中文回答。"
#     }
#     return language_prompts.get(language, None)

# # Function to process the landmark using LangChain
# def process_landmark(image_path, latitude, longitude, language="English", temperature=0.3):
#     """
#     Analyzes a landmark image using LangChain and the Gemini API.
#     Additionally, fetches the address using latitude and longitude and dynamically adds it to the prompt.
#     """
#     try:
#         # Get the location address using latitude and longitude
#         address = get_location_from_coords(latitude, longitude)
#         if address in ["Location not found", "Geocoding error"]:
#             print(f"Failed to fetch address. {address}")
#             return None

#         # Validate image file existence
#         if not os.path.exists(image_path):
#             raise FileNotFoundError(f"Error: The file '{image_path}' does not exist.")

#         # Encode the image to a Base64 string
#         base64_image = get_base64_encoded_image(image_path)
#         image_data_uri = f"data:image/jpeg;base64,{base64_image}"

#         # Validate language
#         headings = HEADINGS_TRANSLATION.get(language)
#         if not headings:
#             raise ValueError(f"Unsupported language: {language}")

#         # Fetch language-specific prompt
#         language_prompt = get_language_prompt(language)
#         if not language_prompt:
#             raise ValueError(f"Unsupported language: {language}")

#         # Construct the dynamic prompt
#         unified_prompt = f"""
# {language_prompt}

# Given image is from {address}, a popular tourist destination.

# You are an expert travel guide and architectural analyst. Your task is to analyze the provided image and generate a detailed, engaging, and factually accurate report about the prominent place or structure shown.

# CRITICAL PROCESSING INSTRUCTIONS:

# First, analyze the image to determine if the subject is a Historical Landmark (e.g., ancient temple, medieval castle, monument with deep historical significance) or a General Place of Interest (e.g., modern skyscraper, iconic bridge, famous commercial building, natural wonder, public square).

# Based on your classification, output the report using only one of the two exact templates below.

# TEMPLATE A: For a Historical Landmark
# [{headings["location"]}] [Full Official Landmark Name]

# [{headings["location"]}]: [City, Country]

# [{headings["year_completed"]}]: [Year or Era] (Omit if not known or not applicable)

# [{headings["materials"]}]: [Primary construction materials] (Omit if not discernible)

# [{headings["architectural_style"]}]: [Predominant architectural style] (Omit if not classified)

# [{headings["historical_overview"]}]:
# [A concise paragraph detailing its origin, key historical events, and significant figures involved (e.g., architects, rulers). Focus on its historical narrative.]

# [{headings["cultural_impact"]}]:
# [A single paragraph explaining its symbolic meaning, its influence on national/regional identity, and its role in culture, arts, or collective memory.]

# [{headings["famous_for"]}]:
# [1. Primary reason for global fame]

# [2. Secondary distinct reason]

# [3. Tertiary distinct reason]

# TEMPLATE B: For a General Place of Interest
# [{headings["location"]}] [Full Official Name of the Place/Structure]

# [{headings["location"]}]: [City, Country]

# [{headings["year_completed"]}]: [Year] (Omit if not known)

# [{headings["key_features"]}]: [Notable materials, engineering marvels, or design elements] (Omit if not discernible)

# [{headings["primary_function"]}]: [E.g., Observation Tower, Transportation Hub, Commercial Center, Public Park] (Omit if not applicable)

# [{headings["overview_and_significance"]}]:
# [A concise paragraph describing what it is, its primary purpose, and why it is significant to the city or field (e.g., engineering, urban planning, commerce).]

# [{headings["visitor_experience"]}]:
# [A single paragraph highlighting what visitors can see and do there, the atmosphere, and any unique experiential aspects.]

# [{headings["known_for"]}]:
# [1. Primary claim to fame]

# [2. Secondary distinct feature or fact]

# [3. Tertiary distinct feature or fact]

# GLOBAL OUTPUT RULES (Apply to both templates):

# Zero Repetition: Do not repeat any fact, figure, or description across different sections.

# Conciseness: Use clear, efficient, and engaging language. Avoid fluff and redundancy.

# Deduction: Base your analysis on visual cues from the image and your encyclopedic knowledge. Omit any numbered line (e.g., Year Completed, Materials) if the information cannot be reasonably inferred or is not applicable.

# Structure: Maintain the exact spacing, bolding, and section ordering as shown in the chosen template. Begin the response immediately with the landmark/place name.
# """

#         # Initialize the LangChain AI model
#         llm = ChatGoogleGenerativeAI(
#             model="gemini-2.5-flash",
#             temperature=0,
#             google_api_key=GEMINI_API_KEY
#         )

#         # Create the message with text and image data URI
#         message = HumanMessage(
#             content=[
#                 {"type": "text", "text": unified_prompt},
#                 {"type": "image_url", "image_url": image_data_uri}
#             ]
#         )

#         # Invoke the model and return the result
#         response = llm.invoke([message])
#         return response.content

#     except Exception as e:
#         print(f"Failed to process landmark: {str(e)}")
#         return None

# # Example usage
# if __name__ == "__main__":
#     image_path = "images/bcd.jpg"  # Replace with your image path
#     latitude = 24.7460  # Replace with actual latitude
#     longitude = 90.4179  # Replace with actual longitude
#     language = "Chinese"  # Example language selection
    
#     result = process_landmark(image_path, latitude, longitude, language=language, temperature=0)
    
#     if result:
#         print("\n--- Final Result ---")
#         print(result)

















