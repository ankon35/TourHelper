
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import base64
from geo import get_location_from_coords  # Importing the geolocation function from geo.py

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Function to encode the image to a base64 string
def get_base64_encoded_image(image_path):
    """Encodes a local image file to a Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to process the landmark using LangChain
def process_landmark(image_path, latitude, longitude, temperature=0.3):
    """
    Analyzes a landmark image using LangChain and the Gemini API.
    Additionally, fetches the address using latitude and longitude and dynamically adds it to the prompt.

    Args:
        image_path (str): The file path to the image.
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        temperature (float): The sampling temperature for the model.

    Returns:
        str: The structured response from the LLM.
    """
    try:
        # Get the location address using latitude and longitude
        address = get_location_from_coords(latitude, longitude)
        # print(address)  # Print the address for debugging

        # Check if the address was found
        if address == "Location not found" or address.startswith("Geocoding error"):
            print(f"Failed to fetch address. {address}")
            return None

        # Check if the image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Error: The file '{image_path}' does not exist.")

        # Encode the image to a Base64 string
        base64_image = get_base64_encoded_image(image_path)
        
        # Construct the data URI for the Base64 image
        image_data_uri = f"data:image/jpeg;base64,{base64_image}"

        # Dynamic prompt with the address inserted
        unified_prompt = f"""
Given image is from {address}, a popular tourist destination.

You are an expert travel guide and architectural analyst. Your task is to analyze the provided image and generate a detailed, engaging, and factually accurate report about the prominent place or structure shown.

CRITICAL PROCESSING INSTRUCTIONS:

First, analyze the image to determine if the subject is a Historical Landmark (e.g., ancient temple, medieval castle, monument with deep historical significance) or a General Place of Interest (e.g., modern skyscraper, iconic bridge, famous commercial building, natural wonder, public square).

Based on your classification, output the report using only one of the two exact templates below.

TEMPLATE A: For a Historical Landmark
[Full Official Landmark Name]

Location: [City, Country]

Year Completed: [Year or Era] (Omit if not known or not applicable)

Materials: [Primary construction materials] (Omit if not discernible)

Architectural Style: [Predominant architectural style] (Omit if not classified)

Historical Overview:
[A concise paragraph detailing its origin, key historical events, and significant figures involved (e.g., architects, rulers). Focus on its historical narrative.]

Cultural Impact:
[A single paragraph explaining its symbolic meaning, its influence on national/regional identity, and its role in culture, arts, or collective memory.]

Famous For:

[1. Primary reason for global fame]

[2. Secondary distinct reason]

[3. Tertiary distinct reason]

TEMPLATE B: For a General Place of Interest
[Full Official Name of the Place/Structure]

Location: [City, Country]

Established: [Year] (Omit if not known)

Key Features: [Notable materials, engineering marvels, or design elements] (Omit if not discernible)

Primary Function: [E.g., Observation Tower, Transportation Hub, Commercial Center, Public Park] (Omit if not applicable)

Overview & Significance:
[A concise paragraph describing what it is, its primary purpose, and why it is significant to the city or field (e.g., engineering, urban planning, commerce).]

Visitor Experience:
[A single paragraph highlighting what visitors can see and do there, the atmosphere, and any unique experiential aspects.]

Known For:

[1. Primary claim to fame]

[2. Secondary distinct feature or fact]

[3. Tertiary distinct feature or fact]

GLOBAL OUTPUT RULES (Apply to both templates):

Zero Repetition: Do not repeat any fact, figure, or description across different sections.

Conciseness: Use clear, efficient, and engaging language. Avoid fluff and redundancy.

Deduction: Base your analysis on visual cues from the image and your encyclopedic knowledge. Omit any numbered line (e.g., Year Completed, Materials) if the information cannot be reasonably inferred or is not applicable.

Structure: Maintain the exact spacing, bolding, and section ordering as shown in the chosen template. Begin the response immediately with the landmark/place name.


"""

        # Initialize the ChatGoogleGenerativeAI model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature= 0.3,
            google_api_key=GEMINI_API_KEY
        )

        # Create a HumanMessage with both text and the Base64 image URI
        message = HumanMessage(
            content=[
                {"type": "text", "text": unified_prompt},
                {"type": "image_url", "image_url": image_data_uri},
            ]
        )

        # Invoke the model with the multimodal message
        response = llm.invoke([message])
        
        # The response is a BaseMessage, so we access its content attribute
        result = response.content

        return result

    except Exception as e:
        print(f"Failed to process landmark: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    image_path = "images/bcd.jpg"  # Replace with your image path
    latitude =  24.7460 # Replace with actual latitude
    longitude = 90.4179 # Replace with actual longitude
    
    result = process_landmark(image_path, latitude, longitude, temperature=0.3)
    
    if result:
        print("\n--- Final Result ---")
        print(result)
