"""
Module to fetch and cache location data from Google Maps API.
"""

import os
import json
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import googlemaps
from pathlib import Path

class MapsDataFetcher:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Maps data fetcher.
        
        Args:
            api_key: Google Maps API key. If None, will look for GOOGLE_MAPS_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('GOOGLE_MAPS_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Google Maps API key not found. Either pass it to the constructor "
                "or set GOOGLE_MAPS_API_KEY environment variable."
            )
        
        self.gmaps = googlemaps.Client(key=self.api_key)
        self.cache_dir = Path('cache/maps_data')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, query: str) -> Path:
        """Get cache file path for a query."""
        # Use query as filename, replace invalid chars with underscore
        filename = "".join(c if c.isalnum() else "_" for c in query)
        return self.cache_dir / f"{filename}.json"
    
    def _is_cache_valid(self, cache_path: Path, max_age_days: int = 30) -> bool:
        """Check if cached data is still valid."""
        if not cache_path.exists():
            return False
        
        # Check if file is older than max_age_days
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime
        return age < timedelta(days=max_age_days)
    
    def _load_cache(self, cache_path: Path) -> Dict:
        """Load data from cache."""
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    def _save_cache(self, cache_path: Path, data: Dict):
        """Save data to cache."""
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def search_places(self, query: str, location: Optional[str] = None,
                     place_type: Optional[str] = None) -> List[Dict]:
        """
        Search for places using Google Maps Places API.
        
        Args:
            query: Search query (e.g., "tourist attractions in Mumbai")
            location: Location to center search around (e.g., "Mumbai, India")
            place_type: Type of place to search for (e.g., "tourist_attraction")
            
        Returns:
            List of place details
        """
        cache_path = self._get_cache_path(f"{query}_{location}_{place_type}")
        
        # Try to load from cache first
        if self._is_cache_valid(cache_path):
            return self._load_cache(cache_path)
        
        # If location is provided, geocode it
        location_bias = None
        if location:
            geocode_result = self.gmaps.geocode(location)
            if geocode_result:
                location_bias = {
                    'lat': geocode_result[0]['geometry']['location']['lat'],
                    'lng': geocode_result[0]['geometry']['location']['lng']
                }
        
        # Search for places
        places_result = self.gmaps.places(
            query,
            location=location_bias,
            type=place_type
        )
        
        # Get detailed information for each place
        detailed_places = []
        for place in places_result.get('results', []):
            try:
                # Get place details
                details = self.gmaps.place(
                    place['place_id'],
                    fields=['name', 'formatted_address', 'type', 'rating',
                           'user_ratings_total', 'opening_hours']
                )
                detailed_places.append(details['result'])
                
                # Respect API rate limits
                time.sleep(0.1)
            except Exception as e:
                print(f"Error fetching details for {place['name']}: {e}")
        
        # Cache the results
        self._save_cache(cache_path, detailed_places)
        
        return detailed_places
    
    def get_tourist_attractions(self, city: str) -> List[Dict]:
        """Get tourist attractions in a city."""
        return self.search_places(
            f"tourist attractions in {city}",
            location=f"{city}, India",
            place_type="tourist_attraction"
        )
    
    def get_landmarks(self, city: str) -> List[Dict]:
        """Get landmarks in a city."""
        return self.search_places(
            f"landmarks in {city}",
            location=f"{city}, India",
            place_type="landmark"
        )
    
    def get_temples(self, city: str) -> List[Dict]:
        """Get temples in a city."""
        return self.search_places(
            f"temples in {city}",
            location=f"{city}, India",
            place_type="hindu_temple"
        )
    
    def build_location_database(self, cities: List[str]) -> Dict:
        """
        Build a comprehensive database of locations and attractions.
        
        Args:
            cities: List of Indian cities to fetch data for
            
        Returns:
            Dictionary containing categorized location data
        """
        database = {
            'cities': {},
            'attractions': {},
            'temples': {},
            'landmarks': {}
        }
        
        for city in cities:
            print(f"Fetching data for {city}...")
            
            # Get city details
            city_details = self.gmaps.geocode(f"{city}, India")
            if city_details:
                database['cities'][city] = city_details[0]
            
            # Get attractions
            attractions = self.get_tourist_attractions(city)
            if attractions:
                database['attractions'][city] = attractions
            
            # Get temples
            temples = self.get_temples(city)
            if temples:
                database['temples'][city] = temples
            
            # Get landmarks
            landmarks = self.get_landmarks(city)
            if landmarks:
                database['landmarks'][city] = landmarks
            
            # Respect API rate limits
            time.sleep(1)
        
        return database

def main():
    """Example usage of the MapsDataFetcher."""
    # Cities organized by type and region
    cities = {
        'metro': [
            'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad',
            'Ahmedabad', 'Pune'
        ],
        'tourist': [
            'Jaipur', 'Agra', 'Varanasi', 'Goa', 'Udaipur', 'Rishikesh',
            'Amritsar', 'Mysore', 'Hampi'
        ],
        'hill_stations': [
            'Shimla', 'Manali', 'Darjeeling', 'Ooty', 'Munnar', 'Mussoorie',
            'Gangtok', 'Kodaikanal'
        ],
        'spiritual': [
            'Haridwar', 'Tirupati', 'Madurai', 'Puri', 'Pushkar', 'Bodh Gaya',
            'Ajmer', 'Dwarka'
        ],
        'beach': [
            'Kochi', 'Pondicherry', 'Kovalam', 'Varkala', 'Gokarna',
            'Port Blair', 'Diu', 'Mahabalipuram'
        ]
    }
    
    try:
        fetcher = MapsDataFetcher()
        all_cities = []
        
        # Process cities in order of importance
        for category in ['metro', 'tourist', 'spiritual', 'hill_stations', 'beach']:
            print(f"\nProcessing {category} cities...")
            for city in cities[category]:
                print(f"\nFetching data for {city}...")
                all_cities.append(city)
                
                try:
                    # Get city details and attractions
                    database = fetcher.build_location_database([city])
                    
                    # Save individual city data for incremental progress
                    output_path = Path(f'data/cities/{city.lower().replace(" ", "_")}.json')
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_path, 'w') as f:
                        json.dump(database, f, indent=2)
                    
                    print(f"Saved data for {city} to {output_path}")
                    
                except Exception as e:
                    print(f"Error processing {city}: {e}")
                    continue
        
        # Combine all city data into one file
        print("\nCombining all city data...")
        combined_database = {
            'cities': {},
            'attractions': {},
            'temples': {},
            'landmarks': {}
        }
        
        city_data_dir = Path('data/cities')
        for city_file in city_data_dir.glob('*.json'):
            with open(city_file, 'r') as f:
                city_data = json.load(f)
                for category in combined_database:
                    combined_database[category].update(city_data.get(category, {}))
        
        # Save the complete database
        output_path = Path('data/india_locations.json')
        with open(output_path, 'w') as f:
            json.dump(combined_database, f, indent=2)
        
        print(f"\nComplete database saved to {output_path}")
        print(f"Processed {len(all_cities)} cities in total")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
