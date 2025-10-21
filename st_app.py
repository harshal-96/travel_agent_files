import streamlit as st
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import json

# Import your travel planning system
from app import TravelPlanningSystem

# Page config
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #FF3333;
        border: none;
    }
    .trip-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    h1 {
        color: #FF4B4B;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'travel_plan' not in st.session_state:
        st.session_state.travel_plan = None
    if 'planning_in_progress' not in st.session_state:
        st.session_state.planning_in_progress = False

def format_currency(amount):
    """Format currency in Indian Rupees"""
    return f"₹{amount:,}"

async def create_travel_plan(trip_data: Dict[str, Any]):
    """Create travel plan asynchronously"""
    travel_system = TravelPlanningSystem()
    result = await travel_system.process_travel_request(trip_data)
    return result

def main():
    initialize_session_state()
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("✈️ AI Travel Planner")
        st.markdown("### Plan your perfect trip with AI-powered insights")
    
    st.markdown("---")
    
    # Sidebar for trip details
    with st.sidebar:
        st.header("🎒 Trip Details")
        
        # Origin and Destination
        origin = st.text_input(
            "From (City/Airport)",
            value="Delhi (DEL)",
            help="Enter your departure city or airport code"
        )
        
        destination = st.text_input(
            "To (City/Airport)",
            value="Mumbai (BOM)",
            help="Enter your destination city or airport code"
        )
        
        # Dates
        st.subheader("📅 Travel Dates")
        col1, col2 = st.columns(2)
        
        with col1:
            departure_date = st.date_input(
                "Departure",
                value=datetime.now() + timedelta(days=30),
                min_value=datetime.now()
            )
        
        with col2:
            return_date = st.date_input(
                "Return",
                value=datetime.now() + timedelta(days=33),
                min_value=departure_date
            )
        
        # Trip details
        st.subheader("👥 Travelers & Budget")
        
        passengers = st.number_input(
            "Number of Travelers",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )
        
        budget = st.select_slider(
            "Budget Range",
            options=['budget', 'mid', 'premium', 'luxury'],
            value='mid',
            help="Select your budget preference"
        )
        
        # Budget descriptions
        budget_info = {
            'budget': '₹10,000 - Basic accommodations',
            'mid': '₹25,000 - Comfortable stay',
            'premium': '₹55,000 - High-end experience',
            'luxury': '₹1,00,000+ - Luxurious travel'
        }
        st.caption(budget_info[budget])
        
        travel_class = st.selectbox(
            "Travel Class",
            options=['economy', 'premium_economy', 'business', 'first'],
            index=0
        )
        
        trip_type = st.selectbox(
            "Trip Type",
            options=['roundtrip', 'oneway'],
            index=0
        )
        
        st.markdown("---")
        
        # Plan button
        plan_button = st.button("🚀 Generate Travel Plan", type="primary")
    
    # Main content area
    if plan_button:
        # Validate inputs
        if not origin or not destination:
            st.error("❌ Please enter both origin and destination")
            return
        
        if departure_date >= return_date:
            st.error("❌ Return date must be after departure date")
            return
        
        # Create trip request
        trip_request = {
            "from": origin,
            "to": destination,
            "departureDate": departure_date.isoformat(),
            "returnDate": return_date.isoformat(),
            "passengers": str(passengers),
            "travelClass": travel_class,
            "tripType": trip_type,
            "budget": budget
        }
        
        # Show loading state
        with st.spinner("🔍 Planning your perfect trip..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Phase 1
                status_text.text("📊 Searching destination information...")
                progress_bar.progress(25)
                
                # Phase 2
                status_text.text("🗺️ Finding best locations and attractions...")
                progress_bar.progress(50)
                
                # Phase 3
                status_text.text("✨ Creating your personalized itinerary...")
                progress_bar.progress(75)
                
                # Generate plan
                result = asyncio.run(create_travel_plan(trip_request))
                progress_bar.progress(100)
                status_text.text("✅ Travel plan ready!")
                
                st.session_state.travel_plan = result
                
            except Exception as e:
                st.error(f"❌ Error generating plan: {str(e)}")
                return
    
    # Display results
    if st.session_state.travel_plan:
        result = st.session_state.travel_plan
        
        if result.get('success'):
            st.success("✅ Your travel plan is ready!")
            
            # Trip summary
            st.markdown("## 📋 Trip Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>🌍 Destination</h3>
                        <h2>{result['destination']}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>📅 Duration</h3>
                        <h2>{result['duration']} Days</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>💰 Budget</h3>
                        <h2>{format_currency(result['budget'])}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>👥 Travelers</h3>
                        <h2>{result['travelers']}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs([
                "📝 Complete Itinerary",
                "🔍 Research Results",
                "🗺️ Location Info",
                "💾 Export"
            ])
            
            with tab1:
                st.markdown("## Your Personalized Travel Plan")
                st.markdown(result['comprehensive_plan'])
            
            with tab2:
                st.markdown("## 🔍 Destination Research")
                st.info("Information gathered from travel websites and reviews")
                st.text(result['search_results'])
            
            with tab3:
                st.markdown("## 🗺️ Google Maps Insights")
                st.info("Top-rated places from Google Maps")
                st.text(result['maps_results'])
            
            with tab4:
                st.markdown("## 💾 Export Your Plan")
                
                # Create downloadable content
                full_plan = f"""
COMPREHENSIVE TRAVEL PLAN
{'='*80}

Trip Details:
- Origin: {result['origin']}
- Destination: {result['destination']}
- Duration: {result['duration']} days
- Budget: {format_currency(result['budget'])}
- Travelers: {result['travelers']}
- Generated: {result['generated_at']}

{'='*80}
COMPLETE ITINERARY
{'='*80}

{result['comprehensive_plan']}

{'='*80}
RESEARCH RESULTS
{'='*80}

{result['search_results']}

{'='*80}
LOCATION INFORMATION
{'='*80}

{result['maps_results']}
"""
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📄 Download as Text",
                        data=full_plan,
                        file_name=f"travel_plan_{result['destination']}_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    json_data = json.dumps(result, indent=2)
                    st.download_button(
                        label="📋 Download as JSON",
                        data=json_data,
                        file_name=f"travel_plan_{result['destination']}_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
                
                st.markdown("---")
                st.info("💡 Tip: Save this plan and share it with your travel companions!")
        
        else:
            st.error(f"❌ Failed to generate plan: {result.get('error', 'Unknown error')}")
    
    else:
        # Welcome message when no plan is generated
        st.markdown("## 👋 Welcome to AI Travel Planner!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                ### 🤖 AI-Powered
                Get personalized recommendations based on your preferences and budget
            """)
        
        with col2:
            st.markdown("""
                ### 🔍 Comprehensive Research
                We search multiple sources to find the best options for you
            """)
        
        with col3:
            st.markdown("""
                ### 📱 Easy to Use
                Simple interface to plan your perfect trip in minutes
            """)
        
        st.markdown("---")
        
        st.info("👈 Enter your trip details in the sidebar and click 'Generate Travel Plan' to get started!")
        
        # Sample destinations
        st.markdown("### 🌟 Popular Destinations")
        
        col1, col2, col3, col4 = st.columns(4)
        
        destinations = [
            ("🏖️", "Goa", "Beaches & Nightlife"),
            ("🏔️", "Manali", "Mountains & Adventure"),
            ("🕌", "Jaipur", "History & Culture"),
            ("🌴", "Kerala", "Backwaters & Nature")
        ]
        
        for col, (emoji, city, desc) in zip([col1, col2, col3, col4], destinations):
            with col:
                st.markdown(f"""
                    <div class="trip-card">
                        <h2 style="text-align: center;">{emoji}</h2>
                        <h3 style="text-align: center;">{city}</h3>
                        <p style="text-align: center; color: #666;">{desc}</p>
                    </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>Powered by Google Gemini AI, Tavily Search & Google Maps</p>
            <p>Made with ❤️ for travelers</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()