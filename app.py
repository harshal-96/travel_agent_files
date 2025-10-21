import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class TravelPlanningSystem:
    def __init__(self):
        # Initialize API keys
        self.google_maps_api_key = self._get_google_maps_api_key()
        self.tavily_api_key = self._get_tavily_api_key()
        self.gemini_api_key = self._get_gemini_api_key()
        
        # Validate required API keys
        self._validate_api_keys()
        
        # Initialize Vertex AI client
        self.client = genai.Client(vertexai=True, api_key=self.gemini_api_key)

        print("Travel system initialized with Vertex AI")

    def _get_google_maps_api_key(self) -> str:
        """Retrieve Google Maps API key from environment"""
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if not api_key:
            print("ERROR: GOOGLE_MAPS_API_KEY is not set.")
            raise ValueError("GOOGLE_MAPS_API_KEY is required")
        return api_key
    
    def _get_tavily_api_key(self) -> str:
        """Retrieve Tavily API key from environment"""
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            print("ERROR: TAVILY_API_KEY is not set.")
            raise ValueError("TAVILY_API_KEY is required")
        return api_key
    
    def _get_gemini_api_key(self) -> str:
        """Retrieve Gemini API key from environment"""
        api_key = os.environ.get("GOOGLE_API_KEY1")
        if not api_key:
            print("ERROR: GOOGLE_API_KEY1 is not set.")
            raise ValueError("GOOGLE_API_KEY1 is required")
        return api_key
    
    def _validate_api_keys(self):
        """Validate that all required API keys are present"""
        required_keys = {
            'GOOGLE_MAPS_API_KEY': self.google_maps_api_key,
            'TAVILY_API_KEY': self.tavily_api_key,
            'GOOGLE_API_KEY': self.gemini_api_key
        }
        
        missing_keys = [key for key, value in required_keys.items() if not value]
        
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        print("All required API keys are configured")

    async def search_with_tavily(self, query: str) -> str:
        """Search using Tavily API directly"""
        try:
            import httpx
            
            url = "https://api.tavily.com/search"
            headers = {"content-type": "application/json"}
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "advanced",
                "include_answer": True,
                "max_results": 10
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                # Format results
                answer = data.get("answer", "")
                results = data.get("results", [])
                
                formatted = f"Search Answer: {answer}\n\n"
                formatted += "Top Results:\n"
                for i, result in enumerate(results[:5], 1):
                    formatted += f"\n{i}. {result.get('title', 'N/A')}\n"
                    formatted += f"   {result.get('content', 'N/A')}\n"
                    formatted += f"   Source: {result.get('url', 'N/A')}\n"
                
                return formatted
                
        except Exception as e:
            print(f"Tavily search error: {str(e)}")
            return f"Search error: {str(e)}"
    
    async def search_with_maps(self, query: str) -> str:
        """Search using Google Maps API directly"""
        try:
            import httpx
            
            # Use Places API for searches
            url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            params = {
                "query": query,
                "key": self.google_maps_api_key
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") != "OK":
                    return f"Maps API status: {data.get('status')}"
                
                # Format results
                results = data.get("results", [])
                formatted = "Google Maps Results:\n\n"
                
                for i, place in enumerate(results[:5], 1):
                    formatted += f"{i}. {place.get('name', 'N/A')}\n"
                    formatted += f"   Address: {place.get('formatted_address', 'N/A')}\n"
                    formatted += f"   Rating: {place.get('rating', 'N/A')} ({place.get('user_ratings_total', 0)} reviews)\n"
                    formatted += f"   Location: {place.get('geometry', {}).get('location', {})}\n\n"
                
                return formatted
                
        except Exception as e:
            print(f"Maps search error: {str(e)}")
            return f"Maps search error: {str(e)}"
    
    async def generate_with_context(self, prompt: str, context: str = "") -> str:
        """Generate response using Vertex AI with optional context"""
        try:
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model="gemini-2.5-flash-lite",
                contents=full_prompt
            )
            
            return response.text
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return f"Generation error: {str(e)}"
    
    async def process_travel_request(self, travel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a travel request using direct API calls"""
        try:
            print(f"Processing travel request for {travel_data.get('to', 'Unknown')}")
            
            # Extract travel data
            destination = travel_data.get('to', '').replace('(', '').replace(')', '')
            origin = travel_data.get('from', '').replace('(', '').replace(')', '')
            departure_date = travel_data.get('departureDate')
            return_date = travel_data.get('returnDate')
            budget = self._parse_budget(travel_data.get('budget', 'mid'))
            passengers = int(travel_data.get('passengers', 1))
            
            # Calculate duration
            if departure_date and return_date:
                departure = datetime.fromisoformat(departure_date)
                return_dt = datetime.fromisoformat(return_date)
                duration = (return_dt - departure).days
            else:
                duration = 3
            
            print(f"{origin} → {destination}, {duration} days, ₹{budget:,}, {passengers} travelers")

            # Phase 1: Search for destination info
            print("Phase 1: Searching destination information...")
            search_query = f"""Find travel information for {destination} including:
- Top attractions and activities with prices
- Hotel recommendations and costs in Indian Rupees
- Best restaurants and dining costs
- Local transportation options and costs
- Weather and best time to visit
- Safety tips and local customs
- Daily budget estimates for {passengers} travelers"""
            
            search_results = await self.search_with_tavily(search_query)
            print("Search completed")
            
            # Phase 2: Get location info
            print("Phase 2: Getting location information...")
            maps_query = f"hotels restaurants attractions in {destination}"
            maps_results = await self.search_with_maps(maps_query)
            print("Maps search completed")
            
            # Phase 3: Generate comprehensive plan
            print("Phase 3: Creating travel plan...")
            planning_prompt = f"""Create a detailed {duration}-day travel plan for {destination}.

TRIP DETAILS:
- Origin: {origin}
- Destination: {destination}
- Duration: {duration} days
- Budget: ₹{budget:,}
- Travelers: {passengers}
- Dates: {departure_date} to {return_date}

SEARCH RESULTS:
{search_results}

LOCATION DATA:
{maps_results}

Create a comprehensive plan with:
1. Executive Summary
2. Day-by-day detailed itinerary with timings
3. Accommodation recommendations (3-4 options)
4. Transportation guide
5. Food & dining suggestions
6. Complete budget breakdown
7. Practical tips
8. Backup plans

Ensure the plan stays within ₹{budget:,} budget and includes specific costs."""
            
            final_plan = await self.generate_with_context(planning_prompt)
            print("Plan created")
            
            return {
                'success': True,
                'destination': destination,
                'origin': origin,
                'duration': duration,
                'budget': budget,
                'travelers': passengers,
                'search_results': search_results,
                'maps_results': maps_results,
                'comprehensive_plan': final_plan,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def _parse_budget(self, budget_range: str) -> int:
        """Convert budget range to numeric value"""
        budget_mapping = {
            'budget': 10000,
            'mid': 25000,
            'premium': 55000,
            'luxury': 100000
        }
        return budget_mapping.get(budget_range, 25000)

# Sync wrapper for FastAPI
def plan_trip(trip_request: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous wrapper for FastAPI"""
    try:
        travel_system = TravelPlanningSystem()
        return asyncio.run(travel_system.process_travel_request(trip_request))
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# Test code
if __name__ == "__main__":
    sample_request = {
        "from": "Delhi (DEL)",
        "to": "Mumbai (BOM)",
        "departureDate": "2025-10-15",
        "returnDate": "2025-10-18",
        "passengers": "2",
        "travelClass": "economy",
        "tripType": "roundtrip",
        "budget": "mid"
    }
    
    print("\n Testing travel planning system...")
    result = plan_trip(sample_request)
    
    if result['success']:
        print("\n Success!")
        with open('travel_plan.txt', 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE TRAVEL PLAN\n")
            f.write("="*80 + "\n\n")
            f.write(result['comprehensive_plan'])
            f.write("\n\n" + "="*80 + "\n")
            f.write("SEARCH RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(result['search_results'])
            f.write("\n\n" + "="*80 + "\n")
            f.write("MAPS RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(result['maps_results'])
        print("Full plan saved to travel_plan.txt")
    else:
        print(f"\n Failed: {result.get('error')}")