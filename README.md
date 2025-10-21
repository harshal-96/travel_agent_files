# Travel Planning System with Google Maps MCP

AI-powered travel planning system using Google Gemini, Tavily API, and Google Maps integration.

## Features

-  Intelligent travel planning with Gemini 2.5 Flash Lite
-  Google Maps API integration for real-time location data
-  Interactive map visualization with custom markers
-  Multi-phase planning: Web research + Maps intelligence + AI synthesis
-  Budget-aware itinerary generation
-  Day-by-day detailed schedules

## Project Structure

```
├── app.py                    # Main travel planning system & FastAPI backend
├── st_app.py                 # Streamlit web interface
├── TravelMapView.tsx         # React map component
├── requirements.txt          # Python dependencies
└── .env                      # Environment variables
```

## Prerequisites

- Python 3.8+
- Google Cloud account with API keys
- Tavily API key

## Setup

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd travel-planning-mcp
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create `.env` file:

```env
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_API_KEY1=your_google_ai_api_key
```

### 4. Enable Google Cloud APIs

Enable these APIs in Google Cloud Console:
- Directions API
- Routes API
- Places API
- Distance Matrix API
- Geocoding API

## Usage

### Run Standalone Test

```bash
python app.py
```
Generates a sample travel plan and saves to `travel_plan.txt`

### Run Streamlit Interface

```bash
streamlit run st_app.py
```
Interface opens at `http://localhost:8501`

### Use as API

Import and use in your FastAPI application:

```python
from app import plan_trip

trip_request = {
    "from": "Delhi (DEL)",
    "to": "Mumbai (BOM)",
    "departureDate": "2025-12-20",
    "returnDate": "2025-12-25",
    "passengers": "2",
    "budget": "mid"
}

result = plan_trip(trip_request)
```

## Request Format

```json
{
  "from": "Delhi (DEL)",
  "to": "Mumbai (BOM)",
  "departureDate": "2025-12-20",
  "returnDate": "2025-12-25",
  "passengers": "2",
  "budget": "mid"
}
```

## Budget Ranges

- `budget`: ₹10,000
- `mid`: ₹25,000
- `premium`: ₹55,000
- `luxury`: ₹1,00,000

## Response Format

```json
{
  "success": true,
  "destination": "Mumbai",
  "origin": "Delhi",
  "duration": 5,
  "budget": 25000,
  "travelers": 2,
  "comprehensive_plan": "Detailed itinerary...",
  "search_results": "Research data...",
  "maps_results": "Location data...",
  "generated_at": "2025-10-22T10:30:00"
}
```

## System Architecture

**Three-Phase Processing:**

1. **Phase 1 - Web Research (Tavily API)**
   - General travel information
   - Attractions and activities
   - Budget estimates
   - Safety tips and local customs

2. **Phase 2 - Location Discovery (Google Maps API)**
   - Hotels with ratings and reviews
   - Restaurants with addresses
   - Attractions with coordinates
   - Real location data

3. **Phase 3 - AI Planning (Gemini 2.5 Flash Lite)**
   - Synthesizes research and location data
   - Generates day-by-day itineraries
   - Creates budget breakdowns
   - Provides practical recommendations

## Frontend Integration

The React component `TravelMapView.tsx` provides interactive map visualization:

```typescript
import { TravelMapView } from './TravelMapView'

<TravelMapView 
  travelPlanData={planData}
  searchResults={results}
/>
```

**Map Features:**
- Color-coded markers (Hotels: Blue, Restaurants: Amber, Attractions: Red)
- Interactive info windows with ratings and reviews
- Category filters
- Auto-fit bounds

## Technology Stack

**Backend:**
- Python 3.8+
- Google Gemini 2.5 Flash Lite
- Tavily API
- Google Maps Places API
- httpx for async requests

**Frontend:**
- React + TypeScript
- Google Maps JavaScript API
- shadcn/ui components
- Lucide icons

## Security Best Practices

API keys stored in environment variables  
Never commit `.env` to version control  
API key restrictions in Google Cloud Console  
Input validation and error handling  

## Example Output

The system generates:
- Executive summary
- Day-by-day itinerary with timings
- 3-4 accommodation options
- Transportation guide
- Restaurant recommendations
- Complete budget breakdown
- Practical tips
- Backup plans

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## License

MIT License

## Acknowledgments

Built after attending AI Labs '25 Mumbai organized by Hack2skill and Google Cloud.

Special thanks to:
- Siddharth for AI Agents session
- Romin for MCP with Google ADK workshop

## Resources

- [Medium Article](YOUR_MEDIUM_LINK) - Detailed tutorial
- [Google Maps Platform](https://developers.google.com/maps)
- [Gemini API Docs](https://ai.google.dev/gemini-api/docs)
- [Tavily API](https://tavily.com)

## Contact

For questions or feedback, connect on LinkedIn: [Your Profile]

---

Star this repo if you find it helpful!
