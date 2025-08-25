# ----------------------------------------------------------------------------------------------------
# --------------------------------------- OPENAI VERSION BELOW ---------------------------------------
# ----------------------------------------------------------------------------------------------------

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
from geo import get_location_from_coords

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Function to encode the image to a base64 string
def get_base64_encoded_image(image_path):
    """Encodes a local image file to a Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Language-specific templates for headings
HEADINGS_TRANSLATION = {
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

# Function to fetch language prompt
def get_language_prompt(language):
    """Returns the appropriate prompt based on language."""
    language_prompts = {
        "English": "Please respond in English.",
        "Chinese": "请用中文回答。",
        "Traditional Chinese": "請用繁體中文回答。"
    }
    return language_prompts.get(language, None)

# Function to process the landmark using LangChain
def process_landmark(image_path, latitude, longitude, language="English", temperature=0.3):
    """
    Analyzes a landmark image using LangChain and the OpenAI API.
    Additionally, fetches the address using latitude and longitude and dynamically adds it to the prompt.
    """
    try:
        # Get the location address using latitude and longitude
        address = get_location_from_coords(latitude, longitude)
        if address in ["Location not found", "Geocoding error"]:
            print(f"Failed to fetch address. {address}")
            return None

        # Validate image file existence
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Error: The file '{image_path}' does not exist.")

        # Encode the image to a Base64 string
        base64_image = get_base64_encoded_image(image_path)

        # Validate language
        headings = HEADINGS_TRANSLATION.get(language)
        if not headings:
            raise ValueError(f"Unsupported language: {language}")

        # Fetch language-specific prompt
        language_prompt = get_language_prompt(language)
        if not language_prompt:
            raise ValueError(f"Unsupported language: {language}")

        # Construct the dynamic prompt
        unified_prompt = f"""
{language_prompt}

Given image is from {address}, a popular tourist destination.

You are an expert travel guide and architectural analyst. Your task is to analyze the provided image and generate a detailed, engaging, and factually accurate report about the prominent place or structure shown.

CRITICAL PROCESSING INSTRUCTIONS:

First, analyze the image to determine if the subject is a Historical Landmark (e.g., ancient temple, medieval castle, monument with deep historical significance) or a General Place of Interest (e.g., modern skyscraper, iconic bridge, famous commercial building, natural wonder, public square).

Based on your classification, output the report using only one of the two exact templates below.

TEMPLATE A: For a Historical Landmark
{headings["location"]} [Full Official Landmark Name]

{headings["location"]}: [City, Country]

{headings["year_completed"]}: [Year or Era] (Omit if not known or not applicable)

{headings["materials"]}: [Primary construction materials] (Omit if not discernible)

{headings["architectural_style"]}: [Predominant architectural style] (Omit if not classified)

{headings["historical_overview"]}:
[A concise paragraph detailing its origin, key historical events, and significant figures involved (e.g., architects, rulers). Focus on its historical narrative.]

{headings["cultural_impact"]}:
[A single paragraph explaining its symbolic meaning, its influence on national/regional identity, and its role in culture, arts, or collective memory.]

{headings["famous_for"]}:
[1. Primary reason for global fame]

[2. Secondary distinct reason]

[3. Tertiary distinct reason]

TEMPLATE B: For a General Place of Interest
{headings["location"]} [Full Official Name of the Place/Structure]

{headings["location"]}: [City, Country]

{headings["year_completed"]}: [Year] (Omit if not known)

{headings["key_features"]}: [Notable materials, engineering marvels, or design elements] (Omit if not discernible)

{headings["primary_function"]}: [E.g., Observation Tower, Transportation Hub, Commercial Center, Public Park] (Omit if not applicable)

{headings["overview_and_significance"]}:
[A concise paragraph describing what it is, its primary purpose, and why it is significant to the city or field (e.g., engineering, urban planning, commerce).]

[{headings["visitor_experience"]}]:
[A single paragraph highlighting what visitors can see and do there, the atmosphere, and any unique experiential aspects.]

[{headings["known_for"]}]:
[1. Primary claim to fame]

[2. Secondary distinct feature or fact]

[3. Tertiary distinct feature or fact]

GLOBAL OUTPUT RULES (Apply to both templates):

Zero Repetition: Do not repeat any fact, figure, or description across different sections.

Conciseness: Use clear, efficient, and engaging language. Avoid fluff and redundancy.

Deduction: Base your analysis on visual cues from the image and your encyclopedic knowledge. Omit any numbered line (e.g., Year Completed, Materials) if the information cannot be reasonably inferred or is not applicable.

Structure: Maintain the exact spacing, bolding, and section ordering as shown in the chosen template. Begin the response immediately with the landmark/place name.
"""

        # Initialize the LangChain OpenAI model
        llm = ChatOpenAI(
            model="gpt-4.1",
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY
        )

        # Create the message with text and image data
        message = HumanMessage(
            content=[
                {"type": "text", "text": unified_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        # Invoke the model and return the result
        response = llm.invoke([message])
        return response.content

    except Exception as e:
        print(f"Failed to process landmark: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    image_path = "images/download.webp"  # Replace with your image path
    latitude = 48.8534 # Replace with actual latitude
    longitude = 2.3488  # Replace with actual longitude
    language = "Chinese"  # Example language selection
    
    result = process_landmark(image_path, latitude, longitude, language=language, temperature=0)
    
    if result:
        print("\n--- Final Result ---")
        print(result)









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





