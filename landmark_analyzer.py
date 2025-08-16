import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')

# Configure the API
genai.configure(api_key=gemini_api_key)

# Prompt templates
# These prompts have been updated to instruct the model on the final output format.
# They are designed to produce a structured, readable response without using Markdown.
historical_prompt = """
You are an expert AI that will be analyzing historical landmarks.
Provide a detailed analysis in a clear, well-structured, and readable format. Do not use a preamble or any Markdown. Do not repeat information. Use 1, 2, 3, for better readable where needed. use proper spacing and line breaks.

Provide the name of the landmark.
Provide the location, year, materials, and architectural style.
Historical Overview: Describe the key events, notable figures, wars, or battles associated with the landmark.
Cultural Impact: Explain its influence on society and art.
Famous For: List the key reasons why the landmark is famous.

you can add more points where needed.
"""

non_historical_prompt = """
You are an expert AI that will be analyzing non-historical landmarks.
Provide a detailed analysis in a clear, well-structured, and readable format. Do not use a preamble or any Markdown. Do not repeat information. Use 1, 2, 3, for better readable where needed. use proper spacing and line breaks.

Provide the name of the landmark.
Provide the location, size, age, and features.
Type of Attraction: Describe the landmark's purpose (e.g., museum, skyscraper).
Famous For: List the key reasons why the landmark is famous.

you can add more points where needed.
"""

# A new function to determine the landmark type and then get the full response.
# This makes the process more efficient by handling both steps in one go.
def process_landmark(image_path):
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Initialize the Gemini Vision model
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        
        # Determine the landmark type and select the appropriate prompt.
        # This initial step is still necessary to choose between the historical and non-historical prompts.
        # The prompt for this step is kept simple to ensure a clear answer.
        preliminary_prompt = "Is the landmark in this image historical or non-historical? Answer with only 'historical' or 'non-historical'."
        
        # Generate a preliminary response to determine the type
        response = model.generate_content([preliminary_prompt, img])
        landmark_type = response.text.strip().lower()

        # Select the appropriate detailed prompt based on the type
        if 'historical' in landmark_type:
            final_prompt = historical_prompt
        else:
            final_prompt = non_historical_prompt

        # Generate the final structured output using the detailed prompt and image
        # The prompt itself contains all the formatting instructions.
        final_response = model.generate_content([final_prompt, img])
        return final_response.text

    except Exception as e:
        return f"An error occurred: {e}"

# Example usage
if __name__ == "__main__":
    # Ensure you have a test image in the same directory.
    # For this example, replace "download.webp" with your image file name.
    image_path = "img.jpg"
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
    else:
        result = process_landmark(image_path)
        print(result)

