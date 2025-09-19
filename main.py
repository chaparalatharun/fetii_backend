from fastapi import FastAPI, HTTPException, Request, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from pathlib import Path
from typing import Optional, Set, Tuple
import re
from typing import Optional, Dict, Any, List, Tuple
import pytz
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import numpy as np
import signal
from contextlib import contextmanager
import uuid
import time
import redis

# Load environment variables
load_dotenv()

# Safety timeouts configuration
SQL_TIMEOUT_SECONDS = 10  # Maximum time for SQL execution
EXPLAIN_TIMEOUT_SECONDS = 5  # Maximum time for EXPLAIN validation

class TimeoutException(Exception):
    pass

@contextmanager
def timeout_context(seconds):
    """Context manager for SQL execution timeouts."""
    def timeout_handler(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

app = FastAPI(title="Fetii Data Analyst API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development/hackathon - more restrictive in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATA_FILE_PATH = Path("FetiiAI_Data_Austin.xlsx")
TIMEZONE = pytz.timezone('America/Chicago')

# Redis configuration for sessions
def get_redis_client():
    # Try Upstash Redis first (for production)
    upstash_url = os.getenv("UPSTASH_REDIS_REST_URL")
    upstash_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")

    if upstash_url and upstash_token:
        try:
            # Use upstash-redis for REST API
            from upstash_redis import Redis
            redis_client = Redis(url=upstash_url, token=upstash_token)
            redis_client.ping()  # Test connection
            print("✅ Upstash Redis connected successfully!")
            return redis_client
        except Exception as e:
            print(f"⚠️ Upstash Redis connection failed: {e}")

    # Fallback to standard Redis (for local development)
    try:
        redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        redis_client.ping()  # Test connection
        print("✅ Local Redis connected successfully!")
        return redis_client
    except redis.ConnectionError:
        print("⚠️ Redis not available - using in-memory session storage")
        return None

redis_client = get_redis_client()

# In-memory fallback for when Redis is not available
memory_sessions = {}
memory_messages = {}
memory_chats = {}  # session_id -> list of chat objects

TTL = 60 * 60  # 60 minutes
SESSION_COOKIE = "fetii_sid"

def _key(sid: str, suffix=""):
    return f"sess:{sid}{(':' + suffix) if suffix else ''}"

def get_session_id(request: Request, x_session_id: str = Header(None)) -> str:
    """Get session ID from cookie or header, validate it exists and extend TTL"""
    sid = request.cookies.get(SESSION_COOKIE) or x_session_id
    if not sid:
        raise HTTPException(400, "Missing session id")

    # Check session exists in Redis or memory
    if redis_client:
        if not redis_client.exists(_key(sid)):
            raise HTTPException(410, "Session expired")
        # Sliding TTL - extend session on each request
        redis_client.expire(_key(sid), TTL)
        redis_client.expire(_key(sid, "messages"), TTL)
    else:
        # Check memory storage
        if sid not in memory_sessions:
            raise HTTPException(410, "Session expired")
        # Update session timestamp for TTL
        memory_sessions[sid] = time.time()

    return sid

def create_chat(session_id: str, title: str = None) -> str:
    """Create a new chat for a session"""
    chat_id = f"chat_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    chat = {
        "id": chat_id,
        "title": title or "New Chat",
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "message_count": 0
    }

    if redis_client:
        # Store chat list in Redis
        redis_client.hset(_key(session_id, "chats"), chat_id, json.dumps(chat))
        redis_client.expire(_key(session_id, "chats"), TTL)
    else:
        # Use memory storage
        if session_id not in memory_chats:
            memory_chats[session_id] = {}
        memory_chats[session_id][chat_id] = chat

    return chat_id

def get_chats(session_id: str) -> list:
    """Get all chats for a session"""
    if redis_client:
        chats_data = redis_client.hgetall(_key(session_id, "chats"))
        chats = [json.loads(chat_json) for chat_json in chats_data.values()]
    else:
        chats = list(memory_chats.get(session_id, {}).values())

    # Sort by updated_at descending (most recent first)
    return sorted(chats, key=lambda x: x["updated_at"], reverse=True)

def update_chat(session_id: str, chat_id: str, title: str = None, increment_messages: bool = False):
    """Update chat metadata"""
    if redis_client:
        chat_data = redis_client.hget(_key(session_id, "chats"), chat_id)
        if chat_data:
            chat = json.loads(chat_data)
            if title:
                chat["title"] = title
            if increment_messages:
                chat["message_count"] = chat.get("message_count", 0) + 1
            chat["updated_at"] = int(time.time())
            redis_client.hset(_key(session_id, "chats"), chat_id, json.dumps(chat))
            redis_client.expire(_key(session_id, "chats"), TTL)
    else:
        if session_id in memory_chats and chat_id in memory_chats[session_id]:
            chat = memory_chats[session_id][chat_id]
            if title:
                chat["title"] = title
            if increment_messages:
                chat["message_count"] = chat.get("message_count", 0) + 1
            chat["updated_at"] = int(time.time())

def delete_chat(session_id: str, chat_id: str) -> bool:
    """Delete a chat and its messages"""
    try:
        # Delete chat metadata
        if redis_client:
            redis_client.hdel(_key(session_id, "chats"), chat_id)
        else:
            if session_id in memory_chats and chat_id in memory_chats[session_id]:
                del memory_chats[session_id][chat_id]

        # Delete chat messages
        key = f"{session_id}:{chat_id}"
        if redis_client:
            redis_client.delete(_key(key, "messages"))
        else:
            memory_messages.pop(key, None)

        return True
    except Exception:
        return False

# Legacy function for backward compatibility
def append_message_legacy(session_id: str, role: str, content: str, metadata: dict = None):
    """Legacy append_message for backwards compatibility"""
    # Create a default chat if no chats exist for this session
    existing_chats = get_chats(session_id)
    if existing_chats:
        chat_id = existing_chats[0]["id"]  # Use the most recent chat
    else:
        # Auto-generate chat title from content if it's a user message
        if role == "user" and content:
            words = content.split()[:4]
            title = " ".join(words) + ("..." if len(content.split()) > 4 else "")
        else:
            title = "New Chat"
        chat_id = create_chat(session_id, title)

    append_message(session_id, chat_id, role, content, metadata)

def append_message(session_id: str, chat_id: str, role: str, content: str, metadata: dict = None):
    """Append a message to a specific chat"""
    message = {
        "role": role,
        "content": content,
        "timestamp": int(time.time()),
        "metadata": metadata or {}
    }

    key = f"{session_id}:{chat_id}"

    if redis_client:
        redis_client.rpush(_key(key, "messages"), json.dumps(message))
        redis_client.ltrim(_key(key, "messages"), -80, -1)  # Keep last 80 messages
        redis_client.expire(_key(key, "messages"), TTL)
    else:
        # Use memory storage
        if key not in memory_messages:
            memory_messages[key] = []
        memory_messages[key].append(message)
        # Keep last 80 messages
        memory_messages[key] = memory_messages[key][-80:]

    # Update chat metadata
    update_chat(session_id, chat_id, increment_messages=True)

def get_chat_messages(session_id: str, chat_id: str) -> list:
    """Get messages for a specific chat"""
    key = f"{session_id}:{chat_id}"

    if redis_client:
        msgs = redis_client.lrange(_key(key, "messages"), -40, -1)  # Last 40 messages
        return [json.loads(m) for m in msgs]
    else:
        # Use memory storage
        return memory_messages.get(key, [])[-40:]  # Last 40 messages

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    dev_mode: Optional[bool] = False

class StreamQueryRequest(BaseModel):
    question: str

class Block(BaseModel):
    type: str  # "chart" | "metric" | "heatmap" | "map" | ...
    figure: Optional[Dict[str, Any]] = None
    caption: Optional[str] = None  # one-line "what am I seeing?"
    alt: Optional[str] = None      # for a11y
    id: Optional[str] = None       # stable id for memoization
    priority: Optional[int] = 0    # if multiple blocks, render order
    value: Optional[float] = None  # for metric blocks
    label: Optional[str] = None    # for metric blocks

class QueryResponse(BaseModel):
    success: bool
    message: str                 # natural language answer
    blocks: Optional[List[Block]] = None
    # keep these for dev only (set when dev flag provided)
    sql_query: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None

class DataSummary(BaseModel):
    total_trips: int
    unique_users: int
    date_range_start: str
    date_range_end: str
    latest_month: str
    last_month: str

# Chat management models
class CreateChatRequest(BaseModel):
    title: Optional[str] = None

class Chat(BaseModel):
    id: str
    title: str
    created_at: int
    updated_at: int
    message_count: int

class ChatListResponse(BaseModel):
    chats: List[Chat]

class ChatResponse(BaseModel):
    chat: Chat

class ChatMessagesResponse(BaseModel):
    messages: List[Dict[str, Any]]

# Global analyst instance
class FetiiAnalyst:
    def __init__(self):
        self.conn = duckdb.connect()
        self.openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.data_loaded = False
        self.data_max_month_ym = None
        self.last_month_ym = None

    def clean_for_json(self, obj):
        """Recursively convert numpy types and other non-serializable objects to JSON-compatible types."""
        if isinstance(obj, dict):
            return {k: self.clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj

    def normalize_column_name(self, col_name: str) -> str:
        """Convert column name to snake_case."""
        col_name = col_name.strip().lower()
        col_name = re.sub(r'[^a-z0-9_]', '_', col_name)
        col_name = re.sub(r'_+', '_', col_name)
        col_name = col_name.strip('_')
        return col_name

    def load_excel_data(self) -> bool:
        """Load Excel data into DuckDB tables."""
        if not DATA_FILE_PATH.exists():
            return False

        try:
            # Read Excel sheets
            xl_file = pd.ExcelFile(DATA_FILE_PATH)

            # Process Trip Data
            if "Trip Data" in xl_file.sheet_names:
                trips_df = pd.read_excel(xl_file, sheet_name="Trip Data")
                trips_df.columns = [self.normalize_column_name(col) for col in trips_df.columns]

                # Handle datetime columns (actual column name is 'trip_date_and_time')
                if 'trip_date_and_time' in trips_df.columns:
                    trips_df['trip_ts'] = pd.to_datetime(trips_df['trip_date_and_time'], errors='coerce')
                    trips_df['trip_date'] = trips_df['trip_ts'].dt.date
                    trips_df['trip_hour'] = trips_df['trip_ts'].dt.hour
                    trips_df['trip_dow'] = trips_df['trip_ts'].dt.day_name()
                    trips_df['month_ym'] = trips_df['trip_ts'].dt.to_period('M').astype(str)

                    # Calculate time anchors
                    valid_months = trips_df['month_ym'].dropna().unique()
                    if len(valid_months) > 0:
                        self.data_max_month_ym = max(valid_months)
                        # Calculate last month (one month before max)
                        max_period = pd.Period(self.data_max_month_ym)
                        self.last_month_ym = str(max_period - 1)

                # Register DataFrame as view (alternative: CREATE TABLE for more stable EXPLAIN behavior)
                self.conn.register('trips', trips_df)

            # Process Checked in User IDs
            if "Checked in User ID's" in xl_file.sheet_names:
                checked_df = pd.read_excel(xl_file, sheet_name="Checked in User ID's")
                checked_df.columns = [self.normalize_column_name(col) for col in checked_df.columns]
                self.conn.register('checked_in', checked_df)

            # Process Customer Demographics
            if "Customer Demographics" in xl_file.sheet_names:
                demo_df = pd.read_excel(xl_file, sheet_name="Customer Demographics")
                demo_df.columns = [self.normalize_column_name(col) for col in demo_df.columns]
                self.conn.register('demographics', demo_df)

            # Create deterministic trip_time_dims view
            self.conn.execute("""
CREATE OR REPLACE VIEW trip_time_dims AS
SELECT
  trip_id,
  CAST(trip_ts AS DATE) AS date,
  strftime(trip_ts, '%Y-%m') AS month_ym,
  CASE strftime(trip_ts, '%w')
    WHEN '0' THEN 'Sunday' WHEN '1' THEN 'Monday' WHEN '2' THEN 'Tuesday'
    WHEN '3' THEN 'Wednesday' WHEN '4' THEN 'Thursday' WHEN '5' THEN 'Friday'
    WHEN '6' THEN 'Saturday' END AS dow,
  CAST(strftime(trip_ts, '%H') AS INTEGER) AS hour,
  CASE WHEN strftime(trip_ts, '%w') IN ('0','6') THEN TRUE ELSE FALSE END AS is_weekend,
  CASE
    WHEN (strftime(trip_ts, '%w')='6' AND CAST(strftime(trip_ts, '%H') AS INTEGER) BETWEEN 20 AND 23)
      OR (strftime(trip_ts, '%w')='0' AND CAST(strftime(trip_ts, '%H') AS INTEGER) BETWEEN 0 AND 2)
  THEN TRUE ELSE FALSE END AS is_sat_night
FROM trips
WHERE trip_ts IS NOT NULL
            """)

            # Create additional helper views
            self.conn.execute("""
CREATE OR REPLACE VIEW trip_with_ages AS
SELECT
  t.trip_id,
  /* true if ANY rider on the trip is 18–24 */
  COALESCE((MAX(CASE WHEN d.age BETWEEN 18 AND 24 THEN 1 ELSE 0 END) = 1), FALSE) AS has_18_24,
  /* Include all trip data for consistency */
  t.booking_user_id,
  t.drop_off_address,
  t.trip_ts,
  t.total_passengers
FROM trips t
LEFT JOIN checked_in c ON t.trip_id = c.trip_id
LEFT JOIN demographics d ON c.user_id = d.user_id
GROUP BY t.trip_id, t.booking_user_id, t.drop_off_address, t.trip_ts, t.total_passengers
            """)

            self.conn.execute("""
CREATE OR REPLACE VIEW trip_riders AS
SELECT
  trip_id,
  COUNT(DISTINCT user_id) AS rider_count
FROM checked_in
GROUP BY trip_id
            """)

            # Helper view for large group queries
            self.conn.execute("""
CREATE OR REPLACE VIEW trip_large_group AS
SELECT
    t.trip_id,
    t.total_passengers,
    COALESCE(tr.rider_count, t.total_passengers) AS riders_effective
FROM trips t
LEFT JOIN trip_riders tr USING(trip_id)
            """)

            # Helper view for time bucket queries
            self.conn.execute("""
CREATE OR REPLACE VIEW trip_time_buckets AS
SELECT
    td.trip_id,
    td.hour,
    td.dow,
    td.is_weekend,
    td.is_sat_night,
    CASE WHEN td.hour BETWEEN 18 AND 23 THEN TRUE ELSE FALSE END AS is_evening,
    td.month_ym
FROM trip_time_dims td
            """)

            self.data_loaded = True
            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def normalize_intent(self, question: str) -> Dict[str, Any]:
        """Extract intent and parameters from natural language question."""
        question_lower = question.lower()

        hints = {
            'user_specific': False,
            'user_id': None,
            'location_analysis': False,
            'time_analysis': False,
            'saturday_night': False,
            'large_groups': False,
            'top_locations': False,
            'unique_count': False,
            'top_n': 10,
            'busy_hours': False,
            'weekend_patterns': False,
            'specific_location': None,
            'demographics': False,
            'last_month': False
        }

        # User-specific queries
        user_match = re.search(r'user\s+(\d+)', question_lower)
        if user_match:
            hints['user_specific'] = True
            hints['user_id'] = user_match.group(1)

        # Location analysis
        if any(word in question_lower for word in ['location', 'drop-off', 'dropoff', 'pickup', 'pick-up', 'address']):
            hints['location_analysis'] = True

        # Time analysis
        if any(word in question_lower for word in ['time', 'hour', 'when', 'day', 'night', 'morning', 'evening']):
            hints['time_analysis'] = True

        # Saturday night specific
        if 'saturday' in question_lower and 'night' in question_lower:
            hints['saturday_night'] = True

        # Large groups
        if any(phrase in question_lower for phrase in ['large group', 'big group', 'group size']):
            hints['large_groups'] = True

        # Top N locations
        top_match = re.search(r'top\s+(\d+)', question_lower)
        if top_match:
            hints['top_locations'] = True
            hints['top_n'] = int(top_match.group(1))
        elif 'top' in question_lower and 'location' in question_lower:
            hints['top_locations'] = True

        # Unique count
        if any(word in question_lower for word in ['unique', 'distinct', 'different', 'how many']):
            hints['unique_count'] = True

        # Busy hours
        if any(phrase in question_lower for phrase in ['busy hour', 'busiest hour', 'peak hour']):
            hints['busy_hours'] = True

        # Weekend patterns
        if any(word in question_lower for word in ['weekend', 'weekday', 'pattern']):
            hints['weekend_patterns'] = True

        # Demographics
        if any(word in question_lower for word in ['demographic', 'age', 'gender', 'customer']):
            hints['demographics'] = True

        # Last month
        if 'last month' in question_lower:
            hints['last_month'] = True

        return hints

    def _get_schema_info(self) -> str:
        """Get current schema information."""
        return f"""
TABLE trips(
  trip_id VARCHAR, booking_user_id VARCHAR,
  pick_up_latitude DOUBLE, pick_up_longitude DOUBLE,
  drop_off_latitude DOUBLE, drop_off_longitude DOUBLE,
  pick_up_address VARCHAR, drop_off_address VARCHAR,
  trip_ts TIMESTAMP, total_passengers INTEGER
)
TABLE checked_in(trip_id VARCHAR, user_id VARCHAR)
TABLE demographics(user_id VARCHAR, age INTEGER)
VIEW trip_time_dims(trip_id, date, month_ym, dow, hour, is_weekend, is_sat_night)
VIEW trip_with_ages(trip_id, has_18_24, booking_user_id, drop_off_address, trip_ts, total_passengers)
VIEW trip_riders(trip_id, rider_count)
"""

    def _current_anchors(self) -> Dict[str, str]:
        """Get current time anchors."""
        return {
            "latest_month": self.data_max_month_ym or "",
            "last_month": self.last_month_ym or ""
        }

    async def _stream_response(self, question: str) -> str:
        """Generate streaming response using OpenAI Chat Completions."""
        system_prompt = f"""You are a senior data analyst for Fetii, an Austin rideshare service. You analyze trip patterns, user demographics, and operational metrics.

CRITICAL DATA STRUCTURE - WHAT EACH ROW REPRESENTS:

1. trips: Each row = ONE TRIP/RIDE (booking event)
   - One booking can have multiple passengers
   - Core trip records with pickup/dropoff locations, timestamps, passenger counts

2. checked_in: Each row = ONE USER-TRIP PARTICIPATION (many-to-many)
   - Maps which users actually rode on each trip
   - Multiple rows per trip for group rides
   - booking_user_id ≠ actual riders in checked_in

3. demographics: Each row = ONE USER's demographic info
   - User age and demographic information
   - One row per unique user

TIME ANCHORS:
- Latest month in data: {self._current_anchors()['latest_month']}
- Previous month: {self._current_anchors()['last_month']}

BUSINESS CONTEXT:
- Austin market focus, university town with young demographics
- Saturday nights are key revenue periods
- Location patterns matter for supply positioning
- Group rides (multiple passengers) are valuable
- Age demographics drive product features

Write natural-language responses that are conversational, professional, and provide business insights. Think step by step through the analysis and explain your reasoning as you work through the data patterns.
"""

        try:
            # Stream from OpenAI
            stream = self.openai_client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {question}\n\nAnalyze this question about our rideshare data and provide insights. Think through the data patterns step by step."}
                ],
                stream=True,
                temperature=0.7,
                max_completion_tokens=800
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        except Exception as e:
            yield f"Error generating response: {str(e)}"

    async def _stream_sql_thinking(self, question: str):
        """Stream the SQL generation thinking process."""
        system_prompt = f"""You are a SQL expert for Austin rideshare data analysis.

Think step by step about how to answer this question. Explain your reasoning process for:
1. What data tables you need to examine
2. What relationships/joins are required
3. What filters or conditions to apply
4. What aggregations or calculations are needed

Then provide the final SQL query or JSON plan at the end of your response.

Temperature 0.0. Return JSON multi-step when the question has multiple sub-goals (e.g., filter, join, rank, and compute share). Otherwise return a single SELECT.

Return ONE of the following at the end:
1) A single DuckDB SELECT query (no CTEs, no comments, no semicolons), OR
2) A JSON object: {{"steps":[{{"id":"s1","purpose":"...","sql":"<ONE SELECT>"}}, ...]}}
   - Each step must contain exactly ONE DuckDB SELECT.
   - No prose outside the JSON.
   - Prefer JSON plan when the question involves multiple constraints (time + age + location) or when computing both rankings and shares.

Current data anchors:
- this_month = {self._current_anchors()['latest_month']}
- last_month = {self._current_anchors()['last_month']}

CRITICAL SCHEMA DEFINITIONS - EACH ROW REPRESENTS:

1. trips: Each row = ONE TRIP (ride booking event)
   - trip_id VARCHAR (UNIQUE) - Primary key for each trip
   - booking_user_id VARCHAR - User who booked the trip (not necessarily a rider)
   - pick_up_latitude DOUBLE, pick_up_longitude DOUBLE - Pickup GPS coordinates
   - drop_off_latitude DOUBLE, drop_off_longitude DOUBLE - Dropoff GPS coordinates
   - pick_up_address VARCHAR, drop_off_address VARCHAR - Human-readable addresses
   - trip_ts TIMESTAMP - When the trip occurred
   - total_passengers INTEGER - Number of passengers on this trip

2. checked_in: Each row = ONE USER-TRIP PARTICIPATION (many-to-many)
   - trip_id VARCHAR - References trips.trip_id
   - user_id VARCHAR - User who actually rode on this trip

3. demographics: Each row = ONE USER's demographic info
   - user_id VARCHAR (UNIQUE) - Primary key for each user
   - age INTEGER - User's age in years

4. trip_time_dims: Each row = ONE TRIP's time calculations (1:1 with trips)
   - trip_id VARCHAR (UNIQUE) - References trips.trip_id
   - date DATE, month_ym VARCHAR - Formatted time fields
   - dow VARCHAR, hour INTEGER - Day of week, hour of day
   - is_weekend BOOLEAN, is_sat_night BOOLEAN - Calculated flags

5. trip_with_ages: Each row = ONE TRIP's age group flags (1:1 with trips)
   - trip_id VARCHAR (UNIQUE) - References trips.trip_id
   - has_18_24 BOOLEAN - TRUE if ANY rider on trip is age 18-24

6. trip_riders: Each row = ONE TRIP's rider count (1:1 with trips)
   - trip_id VARCHAR (UNIQUE) - References trips.trip_id
   - rider_count INTEGER - COUNT(DISTINCT user_id) from checked_in for this trip

KEY COUNTING RULES:
- COUNT(*) FROM trips = number of trips/rides
- COUNT(DISTINCT user_id) FROM checked_in = number of unique users who rode
- COUNT(DISTINCT trip_id) FROM checked_in = number of trips that had riders
- To count users of age X: JOIN checked_in + demographics, then COUNT(DISTINCT user_id)
- To count trips with riders of age X: JOIN trips + checked_in + demographics, then COUNT(DISTINCT trip_id)

Rules:
- ALWAYS use table aliases to avoid ambiguous column references: t.trip_id, td.trip_id, c.trip_id, etc.
- When joining with checked_in, never select raw columns without grouping; always aggregate or group by to avoid row-explosion.
- After joining checked_in, never count trips with COUNT(*); use COUNT(DISTINCT t.trip_id).
- To compute group size from riders, join trip_riders and use rider_count; don't infer from COUNT(*).
- When counting trips after joining checked_in/demographics: COUNT(DISTINCT t.trip_id). When counting riders: COUNT(DISTINCT c.user_id) or use trip_riders.rider_count.
- For any time filter (dow/hour/night/weekend/month), JOIN trip_time_dims td USING(trip_id) and filter on td.*.
- For Saturday night: JOIN trip_time_dims td USING(trip_id) AND td.is_sat_night=TRUE.
- "last month" → td.month_ym='{self._current_anchors()['last_month']}'; "this month" → td.month_ym='{self._current_anchors()['latest_month']}'.
- Do NOT use latitude/longitude or map outputs. Avoid geospatial filtering in SQL.
- For rankings: ORDER BY COUNT(*) DESC or ORDER BY COUNT(DISTINCT t.trip_id) DESC, then LIMIT N.
- If you need actual riders per trip, JOIN trip_riders tr USING(trip_id) and use tr.rider_count.
- For arbitrary age ranges (e.g., 21–25), join checked_in + demographics and filter d.age BETWEEN ...; use COUNT(DISTINCT ...) depending on entity.
- Use trip_with_ages(has_18_24) only as a special case optimization for the "any rider is 18–24" scenario.
- For large groups, prefer trip_large_group.riders_effective >= 6.
- Prefer trip_time_buckets for hour/day/evening/weekend filters.
- Prefer full expressions in ORDER BY when possible (e.g., ORDER BY COUNT(*) DESC), but aliases are acceptable in DuckDB.
- If the question specifies "top K", apply LIMIT K. For large result sets, consider adding LIMIT for performance.
"""

        try:
            # Stream from OpenAI with thinking process
            stream = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {question}\n\nThink through how to query the data step by step, then provide the SQL query or JSON plan."}
                ],
                stream=True,
                # GPT-5 only supports default temperature (1)
                max_completion_tokens=1200  # Increased to accommodate thinking + SQL
            )

            full_response = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_response += delta
                    yield delta

            # Return the full response through a special marker
            yield f"\n__FULL_RESPONSE_MARKER__{full_response}__END_MARKER__"

        except Exception as e:
            yield f"Error generating SQL thinking: {str(e)}"

    def _extract_sql_from_thinking(self, thinking_text: str) -> str:
        """Extract SQL query or JSON plan from thinking text."""
        text = thinking_text.strip()

        # Try to find SQL/JSON within code blocks first
        code_block_match = re.search(r'```(?:sql|json)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            content = code_block_match.group(1).strip()
            if content:
                return content

        # Try to find JSON plan (look for the steps array pattern)
        json_match = re.search(r'(\{"steps":\s*\[.*?\]\s*\})', text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()

        # Try to find a SELECT statement at the end
        lines = text.split('\n')

        # Look for the last SELECT statement
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip().upper()
            if line.startswith('SELECT'):
                # Found a SELECT, now collect the full query
                sql_lines = []
                for j in range(i, len(lines)):
                    sql_lines.append(lines[j])
                    # Stop at obvious end markers
                    if any(marker in lines[j].upper() for marker in [';', '-- END', 'LIMIT']):
                        break
                sql = '\n'.join(sql_lines).strip()
                # Clean up
                sql = re.sub(r';.*$', '', sql, flags=re.DOTALL)  # Remove everything after semicolon
                sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)  # Remove comments
                return sql.strip()

        # Fallback: try to extract anything that looks like SQL
        select_match = re.search(r'(SELECT\s+.*?(?:FROM|;|$).*?)(?:\n\n|\nExplanation|\nThis query|$)', text, re.DOTALL | re.IGNORECASE)
        if select_match:
            sql = select_match.group(1).strip()
            sql = re.sub(r';.*$', '', sql, flags=re.DOTALL)
            return sql.strip()

        # If nothing found, return the last substantial chunk
        substantial_lines = [line for line in lines if len(line.strip()) > 10]
        if substantial_lines:
            return substantial_lines[-1].strip()

        return text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _generate_sql(self, question: str) -> str:
        """Generate SQL query or JSON plan using OpenAI."""

        system_prompt = f"""You are a SQL expert for Austin rideshare data analysis.

Temperature 0.0. Return JSON multi-step when the question has multiple sub-goals (e.g., filter, join, rank, and compute share). Otherwise return a single SELECT.

Return ONE of the following:
1) A single DuckDB SELECT query (no CTEs, no comments, no semicolons), OR
2) A JSON object: {{"steps":[{{"id":"s1","purpose":"...","sql":"<ONE SELECT>"}}, ...]}}
   - Each step must contain exactly ONE DuckDB SELECT.
   - No prose outside the JSON.
   - Prefer JSON plan when the question involves multiple constraints (time + age + location) or when computing both rankings and shares.

Current data anchors:
- this_month = {self._current_anchors()['latest_month']}
- last_month = {self._current_anchors()['last_month']}

CRITICAL SCHEMA DEFINITIONS - EACH ROW REPRESENTS:

1. trips: Each row = ONE TRIP (ride booking event)
   - trip_id VARCHAR (UNIQUE) - Primary key for each trip
   - booking_user_id VARCHAR - User who booked the trip (not necessarily a rider)
   - pick_up_latitude DOUBLE, pick_up_longitude DOUBLE - Pickup GPS coordinates
   - drop_off_latitude DOUBLE, drop_off_longitude DOUBLE - Dropoff GPS coordinates
   - pick_up_address VARCHAR, drop_off_address VARCHAR - Human-readable addresses
   - trip_ts TIMESTAMP - When the trip occurred
   - total_passengers INTEGER - Number of passengers on this trip

2. checked_in: Each row = ONE USER-TRIP PARTICIPATION (many-to-many)
   - trip_id VARCHAR - References trips.trip_id
   - user_id VARCHAR - User who actually rode on this trip
   - NOTE: Multiple rows per trip_id (for multi-passenger trips)
   - NOTE: booking_user_id may differ from actual riders in checked_in

3. demographics: Each row = ONE USER's demographic info
   - user_id VARCHAR (UNIQUE) - Primary key for each user
   - age INTEGER - User's age in years

4. trip_time_dims: Each row = ONE TRIP's time calculations (1:1 with trips)
   - trip_id VARCHAR (UNIQUE) - References trips.trip_id
   - date DATE, month_ym VARCHAR - Formatted time fields
   - dow VARCHAR, hour INTEGER - Day of week, hour of day
   - is_weekend BOOLEAN, is_sat_night BOOLEAN - Calculated flags

5. trip_with_ages: Each row = ONE TRIP's age group flags (1:1 with trips)
   - trip_id VARCHAR (UNIQUE) - References trips.trip_id
   - has_18_24 BOOLEAN - TRUE if ANY rider on trip is age 18-24

6. trip_riders: Each row = ONE TRIP's rider count (1:1 with trips)
   - trip_id VARCHAR (UNIQUE) - References trips.trip_id
   - rider_count INTEGER - COUNT(DISTINCT user_id) from checked_in for this trip

KEY COUNTING RULES:
- COUNT(*) FROM trips = number of trips/rides
- COUNT(DISTINCT user_id) FROM checked_in = number of unique users who rode
- COUNT(DISTINCT trip_id) FROM checked_in = number of trips that had riders
- To count users of age X: JOIN checked_in + demographics, then COUNT(DISTINCT user_id)
- To count trips with riders of age X: JOIN trips + checked_in + demographics, then COUNT(DISTINCT trip_id)

Rules:
- ALWAYS use table aliases to avoid ambiguous column references: t.trip_id, td.trip_id, c.trip_id, etc.
- When joining with checked_in, never select raw columns without grouping; always aggregate or group by to avoid row-explosion.
- After joining checked_in, never count trips with COUNT(*); use COUNT(DISTINCT t.trip_id).
- To compute group size from riders, join trip_riders and use rider_count; don't infer from COUNT(*).
- When counting trips after joining checked_in/demographics: COUNT(DISTINCT t.trip_id). When counting riders: COUNT(DISTINCT c.user_id) or use trip_riders.rider_count.
- For any time filter (dow/hour/night/weekend/month), JOIN trip_time_dims td USING(trip_id) and filter on td.*.
- For Saturday night: JOIN trip_time_dims td USING(trip_id) AND td.is_sat_night=TRUE.
- "last month" → td.month_ym='{self._current_anchors()['last_month']}'; "this month" → td.month_ym='{self._current_anchors()['latest_month']}'.
- Do NOT use latitude/longitude or map outputs. Avoid geospatial filtering in SQL.
- For 'downtown' or spatial terms, answer with time/size proxies or say spatial filters are not supported.
- For fuzzy location by name, use token-AND on LOWER(pick_up_address) and/or LOWER(drop_off_address) based on query intent: e.g., "moody center" ⇒ LOWER(t.drop_off_address) LIKE '%moody%' AND LOWER(t.drop_off_address) LIKE '%center%'.
- For rankings: ORDER BY COUNT(*) DESC or ORDER BY COUNT(DISTINCT t.trip_id) DESC, then LIMIT N.
- If you need actual riders per trip, JOIN trip_riders tr USING(trip_id) and use tr.rider_count.
- For arbitrary age ranges (e.g., 21–25), join checked_in + demographics and filter d.age BETWEEN ...; use COUNT(DISTINCT ...) depending on entity.
- Use trip_with_ages(has_18_24) only as a special case optimization for the "any rider is 18–24" scenario.
- For large groups, prefer trip_large_group.riders_effective >= 6.
- Prefer trip_time_buckets for hour/day/evening/weekend filters.
- Prefer full expressions in ORDER BY when possible (e.g., ORDER BY COUNT(*) DESC), but aliases are acceptable in DuckDB.
- If the question specifies "top K", apply LIMIT K. For large result sets, consider adding LIMIT for performance.
- In JSON multi-step plans, later steps must SELECT FROM the exact view name tmp_step_<id>, where <id> is the prior step's id sanitized to [A-Za-z0-9_].
- To build complete hour×dow grids (including zeros), use range generators or CROSS JOIN only with small helper tables (24×7 sized).
- Never use geospatial predicates or map functions."""

        # Parse question hints for context
        hints = self.normalize_intent(question)
        top_n_hint = f" (top_n={hints['top_n']})" if hints['top_n'] != 10 else ""
        user_prompt = f"Question: {question}{top_n_hint}"

        # LOG PROMPT for debugging
        print(f"\n🤖 GPT-4 PROMPT:")
        print(f"Question: {question}")
        print(f"Hints: {hints}")

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # GPT-5 only supports default temperature (1)
                max_completion_tokens=800
            )

            result = response.choices[0].message.content.strip()

            # LOG GPT RESPONSE for debugging
            print(f"🎯 GPT-4 RESPONSE:")
            print(f"Raw output: {result[:200]}...")  # First 200 chars

            # Clean up any markdown formatting
            result = re.sub(r'^```(?:sql|json)?\s*', '', result, flags=re.IGNORECASE)
            result = re.sub(r'\s*```$', '', result)
            result = result.strip()

            return result

        except Exception as e:
            print(f"Error generating SQL: {e}")
            return ""

    def _parse_plan_or_sql(self, text: str) -> Dict[str, Any]:
        """Parse response as JSON plan or single SQL."""
        txt = text.strip()
        # Try JSON plan
        if txt.startswith('{'):
            try:
                obj = json.loads(txt)
                if isinstance(obj, dict) and "steps" in obj and isinstance(obj["steps"], list):
                    return {"type": "plan", "plan": obj}
            except Exception:
                pass
        # Else treat as single SQL
        return {"type": "sql", "sql": txt}

    def _sanitize_step_id(self, step_id: str) -> str:
        """Sanitize step ID to prevent SQL injection and naming conflicts."""
        if not step_id:
            return ""
        # Keep only alphanumeric and underscore, limit length
        safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', step_id)[:48]
        return f"tmp_step_{safe_id}"

    def _maybe_inject_limit(self, sql: str) -> str:
        """Add safety LIMIT to unbounded queries using structural checks."""
        U = sql.upper().strip()

        # Already has LIMIT
        if " LIMIT " in U:
            return sql

        # Window functions are typically bounded by partitions
        if " OVER " in U or " WINDOW " in U:
            return sql

        # Aggregate functions or GROUP BY - these reduce result set size
        if any(k in U for k in (" GROUP BY", " HAVING")):
            return sql

        # Aggregate function calls - these return single values
        if U.startswith(("SELECT COUNT", "SELECT SUM", "SELECT AVG", "SELECT MIN", "SELECT MAX")):
            return sql

        # DISTINCT present (de-dup) - keep unbounded (often entity lists)
        if " DISTINCT " in U:
            return sql

        # Otherwise cap to prevent huge scans
        return sql + " LIMIT 50"

    def _validate_sql_safety(self, sql: str, valid_step_ids: Optional[Set[str]] = None) -> Tuple[bool, Optional[str]]:
        """Validate SQL query syntax and safety with hardened checks and timeouts."""
        q = sql.strip()
        U = q.upper()

        # Length check - prevent extremely long queries
        if len(q) > 10000:
            return False, "Query exceeds maximum length (10KB)"

        # Must be SELECT
        if not U.startswith("SELECT"):
            return False, "Query must start with SELECT"

        # Block semicolons, comments, CTEs, DDL/DML, admin, etc.
        if ";" in q or "--" in q or "/*" in q or "*/" in q:
            return False, "Comments, semicolons not allowed"

        # Enhanced forbidden keywords list
        forbidden = ("ATTACH","DETACH","PRAGMA","COPY","CREATE","ALTER","DROP",
                     "INSERT","UPDATE","DELETE","VACUUM","CALL","EXEC",
                     "MERGE","REPLACE","WITH","LOAD","EXPORT","IMPORT",
                     "INSTALL","FORCE","SET","RESET","ROLLBACK","COMMIT",
                     "BEGIN","TRANSACTION","SAVEPOINT")

        # Check for forbidden keywords as whole words (not substrings)
        forbidden_pattern = r'\b(?:' + '|'.join(forbidden) + r')\b'
        if re.search(forbidden_pattern, U):
            return False, f"Forbidden SQL keywords detected"

        # Check for potentially dangerous patterns
        dangerous_patterns = [
            r'\bRECURSIVE\b',  # Recursive CTEs can cause infinite loops
            r'\*\s*FROM\s+\(\s*SELECT\s+\*',  # Nested SELECT * patterns
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, U):
                return False, f"Potentially dangerous SQL pattern detected: {pattern}"

        # Validate parentheses are balanced
        if q.count('(') != q.count(')'):
            return False, "Unbalanced parentheses in query"

        # Allowlist relations (include step IDs for multi-step queries)
        allowed = {"TRIPS","CHECKED_IN","DEMOGRAPHICS","TRIP_TIME_DIMS","TRIP_WITH_AGES","TRIP_RIDERS","TRIP_LARGE_GROUP","TRIP_TIME_BUCKETS"}
        if valid_step_ids:
            allowed.update(step_id.upper() for step_id in valid_step_ids)

        # Updated regex to handle aliases: capture table name before optional alias
        # Patterns: FROM table_name, FROM table_name alias, FROM table_name AS alias
        refs = re.findall(r'\bFROM\s+([a-zA-Z_][\w]*)(?:\s+(?:AS\s+)?[a-zA-Z_][\w]*)?\b|\bJOIN\s+([a-zA-Z_][\w]*)(?:\s+(?:AS\s+)?[a-zA-Z_][\w]*)?\b', q, flags=re.IGNORECASE)
        tables = [t for pair in refs for t in pair if t]
        if tables and not all(t.upper() in allowed for t in tables):
            forbidden_tables = [t for t in tables if t.upper() not in allowed]
            return False, f"Access to tables not allowed: {forbidden_tables}"

        # Syntax check with timeout protection
        try:
            with timeout_context(EXPLAIN_TIMEOUT_SECONDS):
                self.conn.execute(f"EXPLAIN {q}")
        except TimeoutException:
            return False, f"SQL validation timed out after {EXPLAIN_TIMEOUT_SECONDS} seconds"
        except Exception as e:
            print(f"SQL validation failed for: {q[:100]}...")  # Only log first 100 chars
            print(f"Error: {e}")
            return False, f"SQL validation failed: {str(e)}"

        return True, None

    def execute_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query with timeout protection and return structured results."""
        try:
            # LOG EVERY QUERY for debugging
            print(f"\n🔍 EXECUTING FULL SQL:")
            print(f"Query: {sql_query}")  # Log full query for debugging age discrepancies

            # Execute with timeout protection
            with timeout_context(SQL_TIMEOUT_SECONDS):
                result_df = self.conn.execute(sql_query).fetchdf()

            # LOG RESULTS for debugging
            print(f"📊 RESULT: {len(result_df)} rows")
            if len(result_df) > 0 and len(result_df.columns) > 0:
                print(f"First few values: {result_df.iloc[0].to_dict()}")

            # Enforce row limit for safety
            MAX_ROWS = 1000
            if len(result_df) > MAX_ROWS:
                print(f"⚠️ Result truncated to {MAX_ROWS} rows (was {len(result_df)})")
                result_df = result_df.head(MAX_ROWS)

            result = {
                "columns": result_df.columns.tolist(),
                "row_count": len(result_df),
                "data": result_df.to_dict('records'),
                "error": None
            }

            # Clean all numpy types
            return self.clean_for_json(result)
        except TimeoutException:
            return {
                "columns": [],
                "row_count": 0,
                "data": [],
                "error": f"Query execution timed out after {SQL_TIMEOUT_SECONDS} seconds"
            }
        except Exception as e:
            return {
                "columns": [],
                "row_count": 0,
                "data": [],
                "error": str(e)
            }

    async def should_visualize(self, question: str, result: Dict[str, Any]) -> bool:
        """Determine if a visualization would be helpful for this question and result."""
        has_rows = bool(result.get("raw_data"))
        if not has_rows:
            return False

        q = question.lower()

        # Crisp rules - fast and reliable
        keywords = ("trend","over time","by hour","by day","top","busiest",
                   "distribution","histogram","rank","share","heatmap","map","geo","pattern")
        if any(k in q for k in keywords):
            return True

        # If first row hints time buckets
        first = {}
        try:
            first = result["raw_data"][0] if result.get("raw_data") else {}
            if any(k in first for k in ("hour","dow","date","month_ym")):
                return True
        except Exception:
            first = {}

        # Optional: low-cost LLM hint (short prompt, temp=0)
        try:
            hint = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                # GPT-5 only supports default temperature (1)
                max_completion_tokens=3,
                messages=[
                    {"role":"system","content":"Answer only yes or no. Chart helpful for this query and rows?"},
                    {"role":"user","content":f"Q: {question}\nColumns: {list(first.keys())}\nRowCount: {len(result.get('raw_data',[]))}"}
                ]
            )
            return hint.choices[0].message.content.strip().lower().startswith('y')
        except Exception:
            return False

    def _find_column(self, df: pd.DataFrame, *candidates) -> Optional[str]:
        """Find first column that matches any of the candidate names (case-insensitive)."""
        df_cols_lower = {c.lower(): c for c in df.columns}
        for candidate in candidates:
            if candidate.lower() in df_cols_lower:
                return df_cols_lower[candidate.lower()]
        return None

    def create_chart(self, df: pd.DataFrame, question: str) -> Optional[Dict[str, Any]]:
        """Create appropriate chart based on data and question."""
        if df is None or df.empty:
            return None

        builders = [
            self._build_metric_card,
            self._build_hour_dow_heatmap,
            self._build_month_line,
            self._build_hour_line,
            self._build_dow_bar,
            # self._build_map_scatter,  # disabled until geo ready
            self._build_topn_horizontal,
            self._build_histogram,
            self._build_box_by_dow,
            self._build_generic_bar,
        ]

        for b in builders:
            try:
                cfg = b(df, question)
                if cfg:
                    return cfg
            except Exception as e:
                print(f"[chart] {b.__name__} skipped: {e}")

        return None

    def _build_metric_card(self, df: pd.DataFrame, question: str) -> Optional[Dict[str, Any]]:
        """Single numeric metric → show metric card"""
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(df) == 1 and len(numeric_cols) == 1:
            col = numeric_cols[0]
            val = float(df.iloc[0][col])
            return {
                "type": "metric",
                "label": col.replace('_', ' ').title(),
                "value": val,
                "id": f"metric-{col}"
            }
        return None

    def _build_hour_dow_heatmap(self, df: pd.DataFrame, question: str) -> Optional[Dict[str, Any]]:
        """Hour × Day heatmap (great for ops staffing)"""
        hour_col = self._find_column(df, 'hour')
        dow_col = self._find_column(df, 'dow', 'day_of_week')
        count_col = self._find_column(df, 'count', 'trip_count', 'rides', 'n')

        if hour_col and dow_col and count_col:
            p = df[[hour_col, dow_col, count_col]].copy()
            p.columns = ['hour', 'dow', 'count']

            # ensure categorical order - always coerce
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            p['dow'] = pd.Categorical(p['dow'], categories=dow_order, ordered=True)
            p = p.dropna(subset=['dow'])

            fig = px.density_heatmap(p, x='hour', y='dow', z='count', nbinsx=24,
                                   title="Trip heatmap by hour × day", color_continuous_scale="Viridis")
            return {
                "type": "chart",
                "figure": fig.to_dict(),
                "caption": "Where rides cluster by time of week",
                "id": "hour-dow-heatmap"
            }
        return None

    def _build_month_line(self, df: pd.DataFrame, question: str) -> Optional[Dict[str, Any]]:
        """Time series by month"""
        month_col = self._find_column(df, 'month_ym', 'month')
        if month_col and len(df) > 1:
            df_sorted = df.sort_values(month_col)
            value_col = [c for c in df.columns if c != month_col][-1]  # Use last non-month column
            fig = px.line(
                df_sorted, x=month_col, y=value_col,
                title=f"{value_col.replace('_', ' ').title()} by Month"
            )
            fig.update_xaxes(type='category')
            return {
                "type": "chart",
                "figure": fig.to_dict(),
                "caption": f"Monthly trend of {value_col}",
                "id": "month-line"
            }
        return None

    def _build_hour_line(self, df: pd.DataFrame, question: str) -> Optional[Dict[str, Any]]:
        """Hour patterns"""
        hour_col = self._find_column(df, 'hour')
        if hour_col and len(df) > 1:
            df_sorted = df.sort_values(hour_col)
            value_col = [c for c in df.columns if c != hour_col][-1]  # Use last non-hour column
            fig = px.line(
                df_sorted, x=hour_col, y=value_col,
                title=f"{value_col.replace('_', ' ').title()} by Hour",
                markers=True
            )
            return {
                "type": "chart",
                "figure": fig.to_dict(),
                "caption": f"Hourly pattern of {value_col}",
                "id": "hour-line"
            }
        return None

    def _build_dow_bar(self, df: pd.DataFrame, question: str) -> Optional[Dict[str, Any]]:
        """Day-of-week patterns"""
        dow_col = self._find_column(df, 'dow', 'day_of_week')
        if dow_col and len(df) > 1:
            df_clean = df[df[dow_col].notna()]
            if len(df_clean) > 0:
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                if df_clean[dow_col].isin(day_order).any():
                    df_clean = df_clean.copy()
                    df_clean[dow_col] = pd.Categorical(df_clean[dow_col], categories=day_order, ordered=True)
                df_sorted = df_clean.sort_values(dow_col)
                value_col = [c for c in df.columns if c != dow_col][-1]  # Use last non-dow column
                fig = px.bar(
                    df_sorted, x=dow_col, y=value_col,
                    title=f"{value_col.replace('_', ' ').title()} by Day of Week"
                )
                fig.update_xaxes(type='category')
                return {
                    "type": "chart",
                    "figure": fig.to_dict(),
                    "caption": f"Weekly pattern of {value_col}",
                    "id": "dow-bar"
                }
        return None

    def _build_map_scatter(self, df: pd.DataFrame, question: str) -> Optional[Dict[str, Any]]:
        """Geographic scatter map (auto if lat/long present)"""
        lat_cols = [c for c in df.columns if 'latitude' in c.lower()]
        lon_cols = [c for c in df.columns if 'longitude' in c.lower()]

        if lat_cols and lon_cols:
            lat_col, lon_col = lat_cols[0], lon_cols[0]
            p = df[[lat_col, lon_col]].dropna().copy()
            if len(p) > 3000:
                p = p.sample(3000, random_state=1)  # Sample for performance
            if len(p) > 5:
                fig = px.scatter_mapbox(
                    p, lat=lat_col, lon=lon_col,
                    zoom=10, height=420,
                    title="Geographic Distribution"
                )
                fig.update_layout(
                    mapbox_style="carto-positron",
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                return {
                    "type": "chart",
                    "figure": fig.to_dict(),
                    "caption": "Geographic spread of locations",
                    "id": "map-scatter"
                }
        return None

    def _build_topn_horizontal(self, df: pd.DataFrame, question: str) -> Optional[Dict[str, Any]]:
        """Top-N horizontal bar (great for "top locations")"""
        if len(df.columns) == 2 and df.shape[0] > 1 and pd.api.types.is_numeric_dtype(df.iloc[:,1]):
            x_col, y_col = df.columns[0], df.columns[1]
            p = df.sort_values(y_col, ascending=True).tail(15)  # cap to 15
            fig = px.bar(p, x=y_col, y=x_col, orientation='h',
                        title=f"Top {min(15, len(p))} {x_col}")
            fig.update_layout(yaxis={'type': 'category'})
            return {
                "type": "chart",
                "figure": fig.to_dict(),
                "caption": f"Top {min(15, len(p))} {x_col} by {y_col}",
                "id": "topn-horizontal"
            }
        return None

    def _build_histogram(self, df: pd.DataFrame, question: str) -> Optional[Dict[str, Any]]:
        """Histogram for distributions (e.g., passenger counts)"""
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 1 and df.shape[0] > 5:
            col = numeric_cols[0]
            fig = px.histogram(df, x=col, nbins=20, title=f"Distribution of {col.replace('_', ' ')}")
            return {
                "type": "chart",
                "figure": fig.to_dict(),
                "caption": f"How {col} values are distributed",
                "id": "histogram"
            }
        return None

    def _build_box_by_dow(self, df: pd.DataFrame, question: str) -> Optional[Dict[str, Any]]:
        """Box plot (spot outliers quickly)"""
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        dow_col = self._find_column(df, 'dow', 'day_of_week')
        if len(numeric_cols) == 1 and dow_col and df[dow_col].nunique() > 1:
            fig = px.box(df, x=dow_col, y=numeric_cols[0], title=f"{numeric_cols[0]} by day of week")
            return {
                "type": "chart",
                "figure": fig.to_dict(),
                "caption": "Spread & outliers by day",
                "id": "box-dow"
            }
        return None

    def _build_generic_bar(self, df: pd.DataFrame, question: str) -> Optional[Dict[str, Any]]:
        """Generic 2-column categorical bar"""
        if len(df.columns) == 2:
            x_col, y_col = df.columns[0], df.columns[1]

            # If looks like an ID (user_id/trip_id or all digits), treat as categorical text
            looks_like_id = bool(re.search(r'(id|user|trip)', x_col, re.I)) or df[x_col].astype(str).str.fullmatch(r'\d+').all()
            if looks_like_id:
                df = df.assign(**{x_col: df[x_col].astype(str)})

            # If only one category → show metric, not a weird single bar
            if df[x_col].nunique() <= 1:
                return {
                    "type": "chart",
                    "figure": None,
                    "caption": f"Single value: {y_col}",
                    "id": "single-metric"
                }

            df_sorted = df.sort_values(y_col, ascending=False)
            fig = px.bar(
                df_sorted, x=x_col, y=y_col,
                title=f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}"
            )
            fig.update_layout(xaxis_tickangle=45)
            if looks_like_id:
                fig.update_xaxes(type='category')  # stop numeric k-formatting for IDs
            return {
                "type": "chart",
                "figure": fig.to_dict(),
                "caption": f"{y_col} breakdown by {x_col}",
                "id": "generic-bar"
            }
        return None

    async def _generate_response(self, question: str, results_bundle: Dict[str, Any]) -> str:
        """Generate analyst response from structured results bundle."""
        system_prompt_analyst = f"""You are a senior data analyst for Fetii, an Austin rideshare service. You analyze trip patterns, user demographics, and operational metrics.

CRITICAL DATA STRUCTURE - WHAT EACH ROW REPRESENTS:

1. trips: Each row = ONE TRIP/RIDE (booking event)
   - One booking can have multiple passengers
   - Core trip records with pickup/dropoff locations, timestamps, passenger counts

2. checked_in: Each row = ONE USER-TRIP PARTICIPATION (many-to-many)
   - Maps which users actually rode on each trip
   - Multiple rows per trip for group rides
   - booking_user_id ≠ actual riders in checked_in

3. demographics: Each row = ONE USER's demographic info
   - User age and demographic information
   - One row per unique user

4. trip_time_dims: Each row = ONE TRIP's time calculations (1:1 with trips)
   - Pre-calculated time dimensions (day of week, hour, weekend flags, Saturday night flags)

5. trip_with_ages: Each row = ONE TRIP's age flags (1:1 with trips)
   - Boolean flag if ANY rider on trip is 18-24 years old

6. trip_riders: Each row = ONE TRIP's rider count (1:1 with trips)
   - Count of distinct riders per trip

CRITICAL INTERPRETATION RULES:
- When data shows "1,114 rows where age=19" → This could mean 1,114 trips OR 1,114 users, depending on query
- COUNT(*) from joined tables ≠ COUNT(DISTINCT entity) - be explicit about what you're counting
- "customers" = unique users, "trips" = ride events, "passengers" = people on trips
- Always verify if numbers represent trips, users, or something else before making claims

TIME ANCHORS:
- Latest month in data: {self._current_anchors()['latest_month']}
- Previous month: {self._current_anchors()['last_month']}

BUSINESS CONTEXT:
- Austin market focus, university town with young demographics
- Saturday nights are key revenue periods
- Location patterns matter for supply positioning
- Group rides (multiple passengers) are valuable
- Age demographics drive product features

Write natural-language responses that are conversational, professional, and provide business insights. Always include specific numbers and timeframes."""

        user_prompt = f"""User question: "{question}"

Here are the query results as structured JSON (possibly multiple steps):
{json.dumps(results_bundle, indent=2, default=str)}

Write a natural-language analyst answer:
- First sentence = direct answer with specific numbers from the data
- Do NOT invent or reference specific dates, months, or years - just use "the data shows"
- Then 2–4 bullets with patterns/trends/caveats
- Do not mention SQL or steps"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": system_prompt_analyst},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_completion_tokens=400
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback response
            all_results = results_bundle.get("results", [])
            if all_results and all_results[0].get("data"):
                return f"Based on the analysis, I found {len(all_results[0]['data'])} results for your question about {question.lower()}."
            return "I was able to process your question, but no results were found."

    def process_question(self, question: str) -> Dict[str, Any]:
        """Process question with multi-query plan support."""
        plan_or_sql = self._generate_sql(question)
        if not plan_or_sql:
            return {
                "answer": "Could not generate a query plan for your question.",
                "sql_used": "",
                "raw_data": [],
                "chart": None
            }

        parsed = self._parse_plan_or_sql(plan_or_sql)

        all_results = []
        sql_used_concat = []

        if parsed["type"] == "plan":
            # Track completed step IDs for chaining
            completed_step_ids = set()

            for step in parsed["plan"]["steps"]:
                sql = step["sql"].strip()
                step_id = step.get("id")
                safe_view_name = self._sanitize_step_id(step_id) if step_id else None

                # Validate SQL with completed step IDs for chaining
                is_valid, error_msg = self._validate_sql_safety(sql, valid_step_ids=completed_step_ids)
                if not is_valid:
                    return {
                        "answer": f"Invalid query generated: {error_msg}. Please rephrase your question.",
                        "sql_used": "",
                        "raw_data": [],
                        "chart": None
                    }

                sql = self._maybe_inject_limit(sql)
                res = self.execute_query(sql)
                sql_used_concat.append(sql)
                all_results.append({
                    "id": step_id,
                    "purpose": step.get("purpose"),
                    "columns": res.get("columns", []),
                    "row_count": res.get("row_count", 0),
                    "data": res.get("data", []),
                    "error": res.get("error")
                })

                # Create temporary view for this step's results for chaining
                if safe_view_name and res.get("data") and not res.get("error"):
                    try:
                        # Convert results to DataFrame and create temporary view
                        df = pd.DataFrame(res["data"])
                        if not df.empty:
                            self.conn.register(safe_view_name, df)
                            completed_step_ids.add(safe_view_name.upper())
                    except Exception as e:
                        print(f"Warning: Could not create temporary view for step {step_id}: {e}")
                        # Continue without step chaining for this step
        else:
            sql = parsed["sql"]
            is_valid, error_msg = self._validate_sql_safety(sql)
            if not is_valid:
                return {
                    "answer": f"Invalid query generated: {error_msg}. Please rephrase your question.",
                    "sql_used": "",
                    "raw_data": [],
                    "chart": None
                }
            sql = self._maybe_inject_limit(sql)
            res = self.execute_query(sql)
            sql_used_concat.append(sql)
            all_results.append({
                "id": "single",
                "purpose": "single_query",
                "columns": res.get("columns", []),
                "row_count": res.get("row_count", 0),
                "data": res.get("data", []),
                "error": res.get("error")
            })

        # Hand structured results to analyst
        nl_answer = self._generate_response(
            question,
            {
                "anchors": self._current_anchors(),
                "results": all_results
            }
        )

        return {
            "answer": nl_answer,
            "sql_used": "\n\n".join(sql_used_concat),
            "raw_data": (all_results[0]["data"][:100] if all_results and all_results[0]["data"] else []),
            "all_results": all_results,  # Include all results for multi-metric support
            "chart": None,  # Will be handled by existing chart creation logic if needed
            "plan": parsed if parsed["type"] == "plan" else None  # Include plan for debugging
        }

    # Compatibility method for existing API
    def generate_sql(self, question: str, hints: Dict[str, Any]) -> Optional[str]:
        """Legacy compatibility wrapper."""
        result = self._generate_sql(question)
        parsed = self._parse_plan_or_sql(result)
        if parsed["type"] == "sql":
            return parsed["sql"]
        else:
            # Return first SQL from plan for compatibility
            steps = parsed["plan"].get("steps", [])
            return steps[0]["sql"] if steps else None

# Global analyst instance
analyst = FetiiAnalyst()

@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    if DATA_FILE_PATH.exists():
        success = analyst.load_excel_data()
        if success:
            print("✅ Data loaded successfully!")
        else:
            print("❌ Failed to load data")
    else:
        print(f"❌ Data file not found: {DATA_FILE_PATH}")

# Session Management Endpoints
@app.post("/session")
async def create_session(request: dict, response: Response):
    """Create or restore an ephemeral session"""
    # Accept client-provided session_id or generate new one
    sid = request.get("session_id") or str(uuid.uuid4())
    now = int(time.time())

    # Check if session already exists (for restoration)
    is_new = True
    if redis_client:
        if redis_client.exists(_key(sid)):
            is_new = False
            # Extend existing session
            redis_client.expire(_key(sid), TTL)
            redis_client.expire(_key(sid, "messages"), TTL)
        else:
            # Create new session in Redis
            redis_client.hset(_key(sid), "created", now)
            redis_client.expire(_key(sid), TTL)
            redis_client.expire(_key(sid, "messages"), TTL)
    else:
        # Use memory storage
        if sid in memory_sessions:
            is_new = False
            # Update timestamp
            memory_sessions[sid] = now
        else:
            # Create new session in memory
            memory_sessions[sid] = now

    response.set_cookie(
        SESSION_COOKIE,
        sid,
        max_age=TTL,
        samesite="Lax",
        httponly=True,
        secure=False  # Set to True in production with HTTPS
    )

    return {"session_id": sid, "expires_in": TTL, "is_new": is_new}

@app.delete("/session")
async def end_session(request: Request, x_session_id: str = Header(None)):
    """End current session and clean up all data (chats, messages, session)"""
    sid = request.cookies.get(SESSION_COOKIE) or x_session_id
    if sid:
        if redis_client:
            # Clean up session
            redis_client.delete(_key(sid))
            redis_client.delete(_key(sid, "messages"))  # Legacy messages

            # Clean up all chats and their messages
            redis_client.delete(_key(sid, "chats"))
            # Delete all chat messages (pattern: session_id:chat_id)
            pattern = _key(f"{sid}:*", "messages")
            for key in redis_client.scan_iter(match=pattern):
                redis_client.delete(key)
        else:
            # Clean up memory storage
            memory_sessions.pop(sid, None)
            memory_messages.pop(sid, None)  # Legacy messages
            memory_chats.pop(sid, None)

            # Clean up all chat messages
            keys_to_delete = [key for key in memory_messages.keys() if key.startswith(f"{sid}:")]
            for key in keys_to_delete:
                memory_messages.pop(key, None)

    return {"ok": True}

@app.get("/history")
async def get_history(request: Request, x_session_id: str = Header(None)):
    """Get session message history"""
    try:
        sid = get_session_id(request, x_session_id)
    except HTTPException:
        return {"messages": []}  # Return empty for expired sessions

    if redis_client:
        msgs = redis_client.lrange(_key(sid, "messages"), -40, -1)  # Last 40 messages
        return {"messages": [json.loads(m) for m in msgs]}
    else:
        # Use memory storage
        msgs = memory_messages.get(sid, [])
        return {"messages": msgs[-40:]}  # Last 40 messages

# Chat management endpoints
@app.get("/chats", response_model=ChatListResponse)
async def get_chats_endpoint(request: Request, x_session_id: str = Header(None)):
    """Get all chats for current session"""
    try:
        sid = get_session_id(request, x_session_id)
    except HTTPException:
        return ChatListResponse(chats=[])

    chats = get_chats(sid)
    return ChatListResponse(chats=chats)

@app.post("/chats", response_model=ChatResponse)
async def create_chat_endpoint(
    chat_request: CreateChatRequest,
    request: Request,
    x_session_id: str = Header(None)
):
    """Create a new chat"""
    try:
        sid = get_session_id(request, x_session_id)
    except HTTPException:
        raise HTTPException(401, "Valid session required")

    chat_id = create_chat(sid, chat_request.title)

    # Get the created chat details
    chats = get_chats(sid)
    chat = next((c for c in chats if c["id"] == chat_id), None)
    if not chat:
        raise HTTPException(500, "Failed to create chat")

    return ChatResponse(chat=chat)

@app.get("/chats/{chat_id}/messages", response_model=ChatMessagesResponse)
async def get_chat_messages_endpoint(
    chat_id: str,
    request: Request,
    x_session_id: str = Header(None)
):
    """Get messages for a specific chat"""
    try:
        sid = get_session_id(request, x_session_id)
    except HTTPException:
        raise HTTPException(401, "Valid session required")

    messages = get_chat_messages(sid, chat_id)
    return ChatMessagesResponse(messages=messages)

@app.delete("/chats/{chat_id}")
async def delete_chat_endpoint(
    chat_id: str,
    request: Request,
    x_session_id: str = Header(None)
):
    """Delete a specific chat"""
    try:
        sid = get_session_id(request, x_session_id)
    except HTTPException:
        raise HTTPException(401, "Valid session required")

    success = delete_chat(sid, chat_id)
    if not success:
        raise HTTPException(404, "Chat not found")

    return {"success": True}

@app.get("/")
async def root():
    return {"message": "Fetii Data Analyst API"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "data_loaded": analyst.data_loaded,
        "openai_configured": bool(os.getenv('OPENAI_API_KEY'))
    }

@app.get("/summary", response_model=DataSummary)
async def get_data_summary():
    """Get data summary statistics"""
    if not analyst.data_loaded:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        trips_count = analyst.conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
        users_count = analyst.conn.execute("SELECT COUNT(DISTINCT user_id) FROM checked_in").fetchone()[0]

        date_range = analyst.conn.execute("""
            SELECT MIN(CAST(trip_ts AS DATE)), MAX(CAST(trip_ts AS DATE))
            FROM trips WHERE trip_ts IS NOT NULL
        """).fetchone()

        return DataSummary(
            total_trips=trips_count,
            unique_users=users_count,
            date_range_start=str(date_range[0]) if date_range[0] else "",
            date_range_end=str(date_range[1]) if date_range[1] else "",
            latest_month=analyst.data_max_month_ym or "",
            last_month=analyst.last_month_ym or ""
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting summary: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, req: Request, x_session_id: str = Header(None)):
    """Process natural language query with multi-step plan support"""
    if not analyst.data_loaded:
        raise HTTPException(status_code=503, detail="Data not loaded")

    if not os.getenv('OPENAI_API_KEY'):
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    # Session handling
    sid = get_session_id(req, x_session_id)
    append_message_legacy(sid, "user", request.question)

    try:
        # Use new process_question method
        result = analyst.process_question(request.question)

        if not result["raw_data"] and "Invalid" in result["answer"]:
            return QueryResponse(
                success=False,
                message=result["answer"]
            )

        # Create blocks array
        blocks = []

        # Let the analyst decide whether a chart helps based on the question & result
        should_chart = analyst.should_visualize(request.question, result)

        if should_chart:
            # Try to create a chart; if it fails, just skip (don't fall back to table)
            try:
                import pandas as pd

                # Check if we have multiple results (like weekend vs weekday)
                all_results = result.get("all_results", [])
                has_multiple_results = len(all_results) > 1

                if has_multiple_results:
                    # Handle multiple metric results (e.g., weekend vs weekday)
                    for i, step_result in enumerate(all_results):
                        step_data = step_result.get("data", [])
                        step_purpose = step_result.get("purpose", "")

                        if not step_data:
                            continue

                        df = pd.DataFrame(step_data)
                        chart_config = analyst.create_chart(df, request.question) if df is not None else None

                        if chart_config and chart_config.get("type") == "metric":
                            # Create a more descriptive label based on purpose
                            if "weekend" in step_purpose.lower():
                                label = "Weekend Trips"
                            elif "weekday" in step_purpose.lower():
                                label = "Weekday Trips"
                            else:
                                label = chart_config.get("label", step_purpose)

                            blocks.append(Block(
                                type="metric",
                                figure=None,
                                value=chart_config.get("value"),
                                label=label,
                                id=f"metric-{step_result.get('id', i)}",
                                priority=i,
                            ))
                else:
                    # Handle single result as before
                    df = pd.DataFrame(result["raw_data"]) if result["raw_data"] else None
                    chart_config = analyst.create_chart(df, request.question) if df is not None else None
                    if chart_config and chart_config.get("figure"):
                        blocks.append(Block(
                            type="chart",
                            figure=analyst.clean_for_json(chart_config["figure"]),
                            caption=chart_config.get("caption"),
                            id=chart_config.get("id"),
                            priority=chart_config.get("priority", 0),
                        ))
                    elif chart_config and chart_config.get("type") == "metric":
                        blocks.append(Block(
                            type="metric",
                            figure=None,
                            value=chart_config.get("value"),
                            label=chart_config.get("label"),
                            id=chart_config.get("id"),
                            priority=chart_config.get("priority", 0),
                        ))
            except Exception:
                pass  # silent: UX stays clean like ChatGPT

        # Save assistant response to session
        append_message_legacy(sid, "assistant", result["answer"], {
            "blocks": [block.dict() for block in blocks],
            "sql_query": result["sql_used"] if request.dev_mode else None,
            "plan": result.get("plan") if request.dev_mode else None
        })

        return QueryResponse(
            success=True,
            message=result["answer"],
            blocks=blocks,
            sql_query=result["sql_used"] if request.dev_mode else None,
            plan=result.get("plan") if request.dev_mode else None
        )

    except Exception as e:
        return QueryResponse(
            success=False,
            message=f"Error processing query: {str(e)}"
        )

@app.get("/chat/stream")
async def stream_chat(q: str, request: Request, x_session_id: str = Header(None), session_id: str = None, chat_id: str = None):
    """Stream chat response with thinking + actual results using Server-Sent Events"""
    if not analyst.data_loaded:
        raise HTTPException(status_code=503, detail="Data not loaded")

    if not os.getenv('OPENAI_API_KEY'):
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    # Session handling - prefer query param for EventSource
    session_id_to_use = session_id or x_session_id
    if not session_id_to_use:
        raise HTTPException(400, "Missing session id")

    # For streaming, we bypass the full session validation since Redis may not be available
    # and we just need the session ID to append messages
    if redis_client and not redis_client.exists(_key(session_id_to_use)):
        raise HTTPException(410, "Session expired")

    # Extend TTL if Redis is available
    if redis_client:
        redis_client.expire(_key(session_id_to_use), TTL)
        redis_client.expire(_key(session_id_to_use, "messages"), TTL)

    sid = session_id_to_use

    # Handle chat ID - create default chat if none provided
    if not chat_id:
        # Check if session has any existing chats
        existing_chats = get_chats(sid)
        if existing_chats:
            # Use the most recent chat (first in the sorted list)
            chat_id = existing_chats[0]["id"]
        else:
            # Create a new chat with auto-generated name from the question
            # Take first few words as title
            words = q.split()[:4]
            title = " ".join(words) + ("..." if len(q.split()) > 4 else "")
            chat_id = create_chat(sid, title)

    append_message(sid, chat_id, "user", q)

    async def generate():
        import asyncio
        import json

        # Anti-buffer prelude (Safari/NGINX) - 2KB padding so browser flushes immediately
        yield f": {' ' * 2048}\n\n"
        yield "event: ready\ndata: {}\n\n"
        await asyncio.sleep(0)

        try:
            # Phase 1: Stream the SQL thinking process AND capture the generated SQL
            yield f"event: thinking_start\ndata: {json.dumps({'section': 'thinking'})}\n\n"
            await asyncio.sleep(0)

            generated_sql_or_plan = ""
            async for chunk in analyst._stream_sql_thinking(q):
                if not chunk:
                    continue
                # Check for the full response marker
                if "__FULL_RESPONSE_MARKER__" in chunk:
                    # Extract the full response from the marker
                    marker_match = re.search(r'__FULL_RESPONSE_MARKER__(.*)__END_MARKER__', chunk, re.DOTALL)
                    if marker_match:
                        generated_sql_or_plan = marker_match.group(1)
                    # Don't yield the marker to the frontend
                    continue

                yield f"event: thinking\ndata: {json.dumps({'delta': chunk})}\n\n"
                await asyncio.sleep(0)  # Let the server flush

            yield f"event: thinking_end\ndata: {json.dumps({'section': 'thinking_complete'})}\n\n"
            await asyncio.sleep(0)

            # Phase 2: Execute the captured SQL and stream the analysis
            yield f"event: analysis_start\ndata: {json.dumps({'section': 'analysis'})}\n\n"
            await asyncio.sleep(0)

            # Parse the generated SQL/plan and execute it directly (skip redundant LLM calls)
            if generated_sql_or_plan.strip():
                # Extract SQL/JSON from the thinking output
                sql_or_plan = analyst._extract_sql_from_thinking(generated_sql_or_plan)

                # Parse and execute the SQL/plan
                parsed = analyst._parse_plan_or_sql(sql_or_plan)

                all_results = []
                sql_used_concat = []

                if parsed["type"] == "plan":
                    # Execute multi-step plan
                    completed_step_ids = set()

                    for step in parsed["plan"]["steps"]:
                        sql = step["sql"].strip()
                        step_id = step.get("id")
                        safe_view_name = analyst._sanitize_step_id(step_id) if step_id else None

                        # Validate and execute SQL
                        is_valid, error_msg = analyst._validate_sql_safety(sql, valid_step_ids=completed_step_ids)
                        if not is_valid:
                            yield f"event: error\ndata: {json.dumps({'error': f'Invalid query: {error_msg}'})}\n\n"
                            return

                        sql = analyst._maybe_inject_limit(sql)
                        res = analyst.execute_query(sql)
                        sql_used_concat.append(sql)
                        all_results.append({
                            "id": step_id,
                            "purpose": step.get("purpose"),
                            "columns": res.get("columns", []),
                            "row_count": res.get("row_count", 0),
                            "data": res.get("data", []),
                            "error": res.get("error")
                        })

                        # Create temporary view for chaining
                        if safe_view_name and res.get("data") and not res.get("error"):
                            try:
                                df = pd.DataFrame(res["data"])
                                if not df.empty:
                                    analyst.conn.register(safe_view_name, df)
                                    completed_step_ids.add(safe_view_name.upper())
                            except Exception as e:
                                print(f"Warning: Could not create temporary view for step {step_id}: {e}")
                else:
                    # Execute single SQL
                    sql = parsed["sql"]
                    is_valid, error_msg = analyst._validate_sql_safety(sql)
                    if not is_valid:
                        yield f"event: error\ndata: {json.dumps({'error': f'Invalid query: {error_msg}'})}\n\n"
                        return

                    sql = analyst._maybe_inject_limit(sql)
                    res = analyst.execute_query(sql)
                    sql_used_concat.append(sql)
                    all_results.append({
                        "id": "single",
                        "purpose": "single_query",
                        "columns": res.get("columns", []),
                        "row_count": res.get("row_count", 0),
                        "data": res.get("data", []),
                        "error": res.get("error")
                    })

                # Generate the natural language response using the results
                nl_answer = await analyst._generate_response(
                    q,
                    {
                        "anchors": analyst._current_anchors(),
                        "results": all_results
                    }
                )

                # Stream the analysis result character by character
                if nl_answer:
                    print(f"🔍 DEBUG: Full nl_answer length: {len(nl_answer)}")
                    print(f"🔍 DEBUG: nl_answer content: {repr(nl_answer)}")
                    for char in nl_answer:
                        yield f"event: analysis\ndata: {json.dumps({'delta': char})}\n\n"
                        await asyncio.sleep(0.03)  # Small delay to make streaming visible

                # Build proper blocks for the frontend
                blocks = []
                try:
                    if all_results and all_results[0].get("data"):
                        df = pd.DataFrame(all_results[0]["data"])
                        chart_cfg = analyst.create_chart(df, q)
                        if chart_cfg:
                            if chart_cfg.get("type") == "metric":
                                blocks.append({
                                    "type": "metric",
                                    "value": chart_cfg["value"],
                                    "label": chart_cfg["label"],
                                    "id": chart_cfg["id"]
                                })
                            elif chart_cfg.get("figure"):
                                blocks.append({
                                    "type": "chart",
                                    "figure": analyst.clean_for_json(chart_cfg["figure"]),
                                    "caption": chart_cfg.get("caption"),
                                    "id": chart_cfg.get("id")
                                })
                except Exception:
                    pass

                # Send blocks if we have any
                if blocks:
                    yield f"event: blocks\ndata: {json.dumps({'blocks': blocks})}\n\n"
                    await asyncio.sleep(0)

                # Save complete response to session
                if nl_answer:
                    append_message(sid, chat_id, "assistant", nl_answer, {
                        "blocks": blocks,
                        "sql_query": all_results[0].get("sql") if all_results else None
                    })
            else:
                yield f"event: error\ndata: {json.dumps({'error': 'No SQL generated from thinking process'})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        # Send done event
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx/traefik
            "Access-Control-Allow-Origin": "*"
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)