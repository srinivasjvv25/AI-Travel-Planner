import streamlit as st
import google.genai as genai
from google.genai import types
from google.genai.errors import APIError
import json
import random
import time
import pandas as pd
import os 
from dotenv import load_dotenv 

# --- Configuration & Schema ---
load_dotenv() # NEW: Load environment variables from .env
MODEL_NAME = "gemini-2.5-flash"
CURRENCY_CODE = "INR"

# --- API Key Retrieval ---
api_key = os.getenv("GEMINI_API_KEY")


# Check if the key is available
if not api_key:
    st.error("FATAL ERROR: GEMINI_API_KEY not found. Please ensure it is set in your .env file.")
    st.stop()


ITINERARY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "day": {"type": "integer"},
            "theme": {"type": "string"},
            "activities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "time": {"type": "string"},
                        "description": {"type": "string"},
                        "estimatedCostINR": {"type": "number"},
                        "transportation": {"type": "string", "description": "Mode of transport and approximate cost/fare. E.g., 'Metro Yellow Line, ₹40' or 'Bus 721, ₹15'"},
                        "latitude": {"type": "number", "description": "REQUIRED: Latitude coordinate for the activity location."}, 
                        "longitude": {"type": "number", "description": "REQUIRED: Longitude coordinate for the activity location."}
                    },
                    "required": ["time", "description", "estimatedCostINR", "latitude", "longitude"] 
                },
            },
            "dailyBudgetSummaryINR": {"type": "number"},
            "accommodationSuggestion": {"type": "string", "description": "Suggestion for a budget hostel or budget-friendly area for the night."}
        },
        "required": ["day", "theme", "activities", "dailyBudgetSummaryINR"]
    },
}

SYSTEM_INSTRUCTION = (
    "You are a world-class, budget-focused student travel agent. "
    "Your goal is to create practical, optimized itineraries that strictly use public transport, free attractions, and cheap local eats, under the specified daily budget in Indian Rupees (INR). "
    "For transportation, always provide the specific mode, line, and approximate cost for students. "
    "***CRITICAL: For EVERY activity, you MUST provide the best possible geographical coordinates (latitude and longitude) for accurate map rendering. Do not use null or zero values.*** "
    "The final output MUST be a JSON array that strictly adheres to the ITINERARY_SCHEMA provided, "
    "using the properties 'estimatedCostINR' and 'dailyBudgetSummaryINR'."
)

# Initialize Session State
if 'itinerary' not in st.session_state:
    st.session_state.itinerary = None
if 'total_trip_cost' not in st.session_state:
    st.session_state.total_trip_cost = 0
if 'destination' not in st.session_state:
    st.session_state.destination = "Hyderabad, India" 
    
# --- Utility Functions ---
def format_currency(cost):
    """Formats a number as Indian Rupees (INR)."""
    return f"₹{cost:,.0f}"

def get_gemini_client(api_key):
    """Initializes the Gemini client."""
    return genai.Client(api_key=api_key)

def get_gemini_config():
    """Returns the Gemini configuration object with JSON schema."""
    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ITINERARY_SCHEMA,
        system_instruction=SYSTEM_INSTRUCTION
    )

def calculate_sustainability(itinerary):
    """Calculates a simple sustainability score (1-5) based on transport modes."""
    score = 0
    transport_mentions = 0
    
    for day in itinerary:
        for act in day.get("activities", []):
            transport = act.get("transportation", "").lower()
            if "metro" in transport or "bus" in transport or "walk" in transport:
                score += 1
            if transport:
                transport_mentions += 1
    
    if transport_mentions == 0:
        return 3 
        
    raw_score = (score / transport_mentions) * 5
    return max(1, min(5, round(raw_score))) 

def generate_itinerary(client, destination, duration, daily_budget, interests, pace, nightlife_skip):
    """Handles the main itinerary generation API call."""
    
    # 1. Build the user prompt
    nightlife_constraint = "Ensure all activities are concluded by 6:00 PM (18:00)." if nightlife_skip else ""
    user_prompt = (
        f"Generate a {duration}-day student travel itinerary for {destination}. "
        f"Maximum total daily budget: {daily_budget} {CURRENCY_CODE}. "
        f"Primary interests: {', '.join(interests)}. "
        f"Travel pace: {pace}. "
        f"{nightlife_constraint} "
        "Focus heavily on student-friendly options (cheap eats, free attractions, public transport). "
        f"The response must use the currency Indian Rupee ({CURRENCY_CODE}) and follow the provided schema strictly."
    )

    # 2. Call the API
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=get_gemini_config()
    )

    # 3. Process the response
    return json.loads(response.text)

def regenerate_activity(client, current_itinerary, day_index, activity_index, new_goal, cost_key):
    """Regenerates a single activity or reduces cost based on goal."""
    day = current_itinerary[day_index]
    old_activity = day['activities'][activity_index]

    user_prompt = (
        f"In this itinerary for {day['theme']} in {st.session_state.destination}, the current activity at {old_activity['time']} is: "
        f"'{old_activity['description']}' with a cost of {format_currency(old_activity['estimatedCostINR'])}. "
        f"The theme of the day is '{day['theme']}'. "
        f"The user wants to {new_goal}. "
        "Suggest ONLY the JSON object for the single new activity, replacing the old one. "
        "The suggested activity MUST be significantly cheaper or a different attraction/theme. "
        "Include accurate latitude and longitude coordinates. "
        "Return the output as a single-item JSON array matching the 'activities' item schema."
    )

    with st.spinner(f"♻️ Regenerating activity on Day {day['day']}..."):
        try:
            single_activity_schema = ITINERARY_SCHEMA["items"]["properties"]["activities"]["items"]

            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={"type": "array", "items": single_activity_schema},
                system_instruction="You are a quick-acting travel optimizer. Return only a one-item JSON array for a new activity with coordinates."
            )

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=config
            )

            new_activity = json.loads(response.text)[0]
            
            old_cost = old_activity['estimatedCostINR']
            new_cost = new_activity.get('estimatedCostINR', 0)
            
            day[cost_key] = day.get(cost_key, 0) - old_cost + new_cost
            day['activities'][activity_index] = new_activity
            
            st.session_state.itinerary = current_itinerary
            st.success(f"Activity replaced! New cost: {format_currency(new_cost)}")
            st.rerun()

        except Exception as e:
            st.error(f"Failed to regenerate activity: {e}. Try a full regeneration.")


# --- Mock Offline Demo Itinerary ---
MOCK_ITINERARY = [
    {
        "day": 1,
        "theme": "Charminar, Old City Heritage & Hyderabadi Cuisine",
        "activities": [
            {"time": "10:00 AM", "description": "Visit Charminar and explore the surrounding Laad Bazaar (Hyderabad's main landmark)", "estimatedCostINR": 50, "transportation": "TSRTC Bus from Koti, ₹15", "latitude": 17.3616, "longitude": 78.4747},
            {"time": "12:30 PM", "description": "Lunch: Famous Irani Chai and Osmania Biscuits at a local cafe near Charminar", "estimatedCostINR": 150, "transportation": "Walk", "latitude": 17.3630, "longitude": 78.4755},
            {"time": "02:30 PM", "description": "Visit the Mecca Masjid (Free entry)", "estimatedCostINR": 0, "transportation": "Walk", "latitude": 17.3608, "longitude": 78.4747},
            {"time": "05:00 PM", "description": "Explore the Telangana State Archaeology Museum (small entry fee)", "estimatedCostINR": 30, "transportation": "Metro Red Line (Charminar to Nampally), ₹35", "latitude": 17.3917, "longitude": 78.4746},
            {"time": "07:30 PM", "description": "Dinner: Affordable Chicken or Veg Biryani at a student-friendly eatery", "estimatedCostINR": 350, "transportation": "Metro, ₹40", "latitude": 17.4400, "longitude": 78.4700} # Near Ameerpet/Kukatpally area
        ],
        "dailyBudgetSummaryINR": 620,
        "accommodationSuggestion": "Budget hostel in the Begumpet or Ameerpet area for central metro connectivity."
    }
]

# ---  UI  ---
st.set_page_config(layout="wide", page_title="AI Student Travel Planner")

st.markdown("""
    <style>
        /* New, more vibrant color scheme */
        .main-header {
            font-size: 2.5rem;
            font-weight: 800;
            color: #FF9933; /* Saffron/Orange */
            text-align: center;
        }
        .subheader {
            font-size: 1.25rem;
            color: #138808; /* India Green */
            text-align: center;
        }
        .daily-card {
            background-color: #f0fff0; /* Very light green background */
            padding: 1.5rem;
            border-radius: 0.75rem;
            margin-bottom: 1.5rem;
            border-left: 5px solid #000080; /* Navy blue accent */
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🇮🇳 AI Student Travel Planner</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Budget-friendly itineraries powered by AI</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Planner Setup")
    
    st.text_input(
        "Destination City/Country", 
        "Hyderabad, India",
        key="destination" 
    )
    
    duration = st.slider("Trip Duration (Days)", 1, 14, 3)
    
    daily_budget = st.slider("Max Daily Budget (INR) (Slider: 500 - 9000)", 500, 9000, 4500, step=500)
    
    st.caption(f"Current Target Daily Budget: {format_currency(daily_budget)}")
    
    interests = st.multiselect(
        "Travel Interests",
        ["Historical Sites", "Local Cuisine", "Nightlife", "Museums", "Nature", "Shopping", "Architecture"],
        default=["Local Cuisine", "Historical Sites"]
    )
    pace = st.radio("Travel Pace", ["Relaxed", "Moderate", "Fast"], index=1)
    generate = st.button("✨ Generate Itinerary")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Advanced Options")

# Nightlife Skip Toggle
nightlife_skip = st.sidebar.checkbox("☀️ Skip Nightlife (End by 6 PM)", value=False)


# --- Main Logic ---
is_demo_mode = not api_key

if 'itinerary' not in st.session_state or generate:
    
    if is_demo_mode:
        st.session_state.itinerary = MOCK_ITINERARY
        st.session_state.total_trip_cost = sum(day.get("dailyBudgetSummaryINR", 0) for day in MOCK_ITINERARY)
        if generate:
             st.warning("🚀 Running in Demo Mode (Offline Example Itinerary for Hyderabad)")
    
    if not is_demo_mode and generate:
        try:
            client = get_gemini_client(api_key)
            with st.spinner("✈️ Generating optimized itinerary..."):
                itinerary_data = generate_itinerary(
                    client, 
                    st.session_state.destination,
                    duration, 
                    daily_budget,
                    interests, 
                    pace, 
                    nightlife_skip
                )
                
                if not isinstance(itinerary_data, list):
                    if isinstance(itinerary_data, dict) and "itinerary" in itinerary_data:
                         itinerary_data = itinerary_data["itinerary"]
                    else:
                         raise ValueError("Model output did not return a valid list structure.")

                st.session_state.itinerary = itinerary_data
                st.session_state.total_trip_cost = sum(day.get("dailyBudgetSummaryINR", 0) for day in itinerary_data)
                st.success(f"✅ Itinerary for {st.session_state.destination} ({len(itinerary_data)} days) generated!")

        except APIError as e:
            st.error(f"❌ Gemini API Error: {e.message}. Displaying Demo Itinerary.")
            st.session_state.itinerary = MOCK_ITINERARY
            st.session_state.total_trip_cost = sum(day.get("dailyBudgetSummaryINR", 0) for day in MOCK_ITINERARY)
        except Exception as e:
            st.error(f"❌ General Error: {e}. Displaying Demo Itinerary.")
            st.session_state.itinerary = MOCK_ITINERARY
            st.session_state.total_trip_cost = sum(day.get("dailyBudgetSummaryINR", 0) for day in MOCK_ITINERARY)


# --- Results ---

itinerary = st.session_state.itinerary
cost_key = "dailyBudgetSummaryINR"
act_cost_key = "estimatedCostINR"

if itinerary:
    col_summary, col_cost, col_reduce = st.columns([1, 1, 1])

    sustainability_score = calculate_sustainability(itinerary)
    score_text = "🌳" * sustainability_score + " " * (5 - sustainability_score)

    with col_summary:
        st.metric("Sustainability Score", score_text, help="Based on use of public transport/walking.")
    
    avg_cost = st.session_state.total_trip_cost / len(itinerary) if itinerary else 0
    with col_cost:
        st.metric("Avg Daily Budget", format_currency(avg_cost), help=f"Target: {format_currency(daily_budget)}")

    all_activities = []
    for day_idx, day in enumerate(itinerary):
        for act_idx, act in enumerate(day.get('activities', [])):
            all_activities.append({
                "cost": act.get(act_cost_key, 0),
                "day_idx": day_idx,
                "act_idx": act_idx
            })
            
    if all_activities and not is_demo_mode:
        most_expensive = max(all_activities, key=lambda x: x['cost'])
        
        with col_reduce:
            if st.button(f"📉 Reduce Highest Cost ({format_currency(most_expensive['cost'])})", help="Asks AI to find a cheaper replacement for the most expensive item."):
                client = get_gemini_client(api_key)
                regenerate_activity(
                    client, 
                    itinerary, 
                    most_expensive['day_idx'], 
                    most_expensive['act_idx'], 
                    "suggest a significantly cheaper, similar, student-friendly alternative",
                    cost_key
                )
    
    if is_demo_mode and all_activities:
        with col_reduce:
            st.button(f"📉 Reduce Highest Cost (Demo)", disabled=True, help="Not available in Demo Mode. Provide a GEMINI_API_KEY.")
        
    st.markdown("---")

    map_data = []

    for day_index, day in enumerate(itinerary):
        st.markdown('<div class="daily-card">', unsafe_allow_html=True)
        
        st.subheader(f"📅 Day {day.get('day', 'N/A')}: {day.get('theme', 'No Theme')}")
        
        daily_cost = day.get(cost_key, 0)
        
        if daily_cost > daily_budget:
            st.error(f"⚠️ **OVER BUDGET:** {format_currency(daily_cost)} (Target: {format_currency(daily_budget)})")
        else:
            st.markdown(f"**✅ Estimated Daily Cost:** {format_currency(daily_cost)}")

        st.markdown("---")
        
        for activity_index, act in enumerate(day.get("activities", [])):
            col_act, col_swap = st.columns([6, 1])
            
            with col_act:
                cost = act.get(act_cost_key, 0)
                st.markdown(f"""
                    **🕒 {act.get('time', 'N/A')}** — **{act.get('description', 'Activity details missing')}** ({format_currency(cost)})
                    <br>
                    <small>🚌 Transport: {act.get('transportation', 'Not specified')}</small>
                """, unsafe_allow_html=True)
            
            with col_swap:
                swap_key = f"swap_{day_index}_{activity_index}"
                if st.button("🔄 Swap", key=swap_key, help="Suggest a new activity for this time slot", disabled=is_demo_mode):
                    if not is_demo_mode:
                        client = get_gemini_client(api_key)
                        regenerate_activity(
                            client, 
                            itinerary, 
                            day_index, 
                            activity_index, 
                            "suggest a different attraction/activity nearby with a similar cost",
                            cost_key
                        )
            
            st.markdown("---")
            
            # Collect Map Data
            if act.get('latitude') and act.get('longitude'):
                map_data.append({
                    'lat': act['latitude'], 
                    'lon': act['longitude'],
                    'label': f"Day {day_index+1}: {act['description']}"
                })
        
        st.markdown(f"**🏠 Night Stay:** *{day.get('accommodationSuggestion', 'Not specified')}*")
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. Final Summary and Map Visualization
    st.header("🗺️ Trip Map Overview")
    
    try:
        if map_data and all('lat' in d and 'lon' in d for d in map_data):
            map_df = pd.DataFrame(map_data)
            st.map(map_df, latitude='lat', longitude='lon', zoom=10)
        else:
            st.warning("Map data (latitude/longitude) is missing or incomplete. Displaying a general map of Delhi.")
            st.map(data=pd.DataFrame([{'lat': 28.65, 'lon': 77.22}]), zoom=10)
    except Exception as e:
        st.error(f"Error rendering map: {e}")
        st.map(data=pd.DataFrame([{'lat': 28.65, 'lon': 77.22}]), zoom=10)

    st.info(f"**Total Estimated Trip Cost:** {format_currency(st.session_state.total_trip_cost)} {CURRENCY_CODE}")
