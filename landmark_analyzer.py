
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')

# Configure the API
genai.configure(api_key=gemini_api_key)

# Unified prompt template that handles both historical and non-historical landmarks
unified_prompt = """
Act as a, you are an expert Tour Guide. Analyze the landmark in this image and provide a detailed response in the following exact format without any repetition:

[Landmark Name]

1. Location: [City, Country]
2. Year Completed: [Year]
3. Materials: [Primary materials used]
4. Architectural Style: [Style description]

Historical Overview:
[Concise paragraph about key historical events and figures, no repetition]

Cultural Impact:
[Single well-structured paragraph about cultural significance]

Famous For:
[Bullet points of key reasons for fame, no repetition]

- Provide all information exactly once
- Maintain this exact structure and spacing
- Do not repeat any facts or sections
- Use clear, concise language without redundancy
- Begin immediately with the landmark name
"""

def process_landmark(image_path):
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Initialize the Gemini Vision model
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        
        # Single API call with unified prompt
        response = model.generate_content([unified_prompt, img])
        return response.text

    except Exception as e:
        return f"An error occurred: {e}"

# Example usage
if __name__ == "__main__":
    image_path = "images/download.webp"
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
    else:
        result = process_landmark(image_path)
        print(result)



