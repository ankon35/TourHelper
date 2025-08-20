# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from typing import List
# from landmark_analyzer import process_landmark  # Import from your existing file
# import os
# from PIL import Image
# import io
# from geo import get_location_from_coords  # Ensure this import works correctly

# app = FastAPI(
#     title="Landmark Analysis API",
#     description="API for analyzing landmarks using Gemini AI",
#     version="1.0.0"
# )

# class LandmarkResponse(BaseModel):
#     landmark_name: str
#     location: str
#     year_completed: str
#     materials: str
#     architectural_style: str
#     historical_overview: str
#     cultural_impact: str
#     famous_for: List[str]

# class ErrorResponse(BaseModel):
#     detail: str

# def parse_response(response_text: str) -> LandmarkResponse:
#     """Parse the raw response text into structured data"""
#     sections = [section.strip() for section in response_text.split("\n\n") if section.strip()]
    
#     if len(sections) < 4:
#         raise ValueError("Invalid response format from AI model")
    
#     # Extract landmark name (first line)
#     landmark_name = sections[0].strip("[]")
    
#     # Parse details section
#     details = sections[1].split("\n")
#     details_dict = {}
#     for line in details:
#         if ":" in line:
#             key, value = line.split(":", 1)
#             details_dict[key.strip()] = value.strip()
    
#     # Get other sections
#     historical_overview = sections[2].replace("Historical Overview:", "").strip()
#     cultural_impact = sections[3].replace("Cultural Impact:", "").strip()
    
#     # Get famous for points (handle cases where it might be missing)
#     famous_for = []
#     if len(sections) > 4:
#         famous_for = [point.strip("- ").strip() for point in sections[4].split("\n") if point.strip()]
    
#     return LandmarkResponse(
#         landmark_name=landmark_name,
#         location=details_dict.get("1. Location", "Unknown"),
#         year_completed=details_dict.get("2. Year Completed", "Unknown"),
#         materials=details_dict.get("3. Materials", "Unknown"),
#         architectural_style=details_dict.get("4. Architectural Style", "Unknown"),
#         historical_overview=historical_overview,
#         cultural_impact=cultural_impact,
#         famous_for=famous_for
#     )

# @app.post("/analyze-landmark/", response_model=LandmarkResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
# async def analyze_landmark(image: UploadFile = File(...), latitude: float = 0.0, longitude: float = 0.0):
#     """Analyze a landmark image and return structured information"""
#     try:
#         # Validate image file
#         if not image.content_type.startswith('image/'):
#             raise HTTPException(status_code=400, detail="File must be an image")
        
#         # Read image content
#         contents = await image.read()
#         img = Image.open(io.BytesIO(contents))
        
#         # Save temporarily (or process directly from memory if possible)
#         temp_path = "temp_upload.jpg"
#         img.save(temp_path)
        
#         try:
#             # Process the image along with geolocation
#             raw_response = process_landmark(temp_path, latitude, longitude)
            
#             # Parse the response
#             response = parse_response(raw_response)
            
#             return response
            
#         finally:
#             # Clean up temporary file
#             if os.path.exists(temp_path):
#                 os.remove(temp_path)
                
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
