from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

def get_location_from_coords(latitude, longitude):
    """
    Convert latitude and longitude to human-readable address
    """
    try:
        # Initialize the geocoder
        geolocator = Nominatim(user_agent="location_finder_app")
        
        # Reverse geocode the coordinates
        location = geolocator.reverse((latitude, longitude), exactly_one=True)
        
        if location:
            return location.address
        else:
            return "Location not found"
            
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        return f"Geocoding error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
latitude = 24.09581710 
longitude = 90.41251810

address = get_location_from_coords(latitude, longitude)


