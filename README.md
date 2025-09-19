# Fetii Data Analyst Backend

FastAPI backend for the Fetii Data Analyst chatbot - an AI-powered assistant for analyzing Austin rideshare data.

## Features

- 🤖 OpenAI GPT-powered natural language queries
- 📊 DuckDB for fast analytical queries
- 📈 Plotly for data visualization
- 🔄 Server-Sent Events for real-time streaming responses
- 💾 Redis session management with Upstash
- 🛡️ Input validation and SQL injection protection

## Tech Stack

- **Framework**: FastAPI 0.115.0 + Uvicorn
- **Database**: DuckDB (embedded analytical database)
- **AI**: OpenAI GPT API
- **Caching**: Redis (Upstash)
- **Data Processing**: Pandas, Plotly
- **Language**: Python 3.11+

## Environment Variables

```
OPENAI_API_KEY=your-openai-api-key
UPSTASH_REDIS_REST_URL=https://your-upstash-url.upstash.io
UPSTASH_REDIS_REST_TOKEN=your-upstash-token
```

## Development

```bash
pip install -r requirements.txt
python main.py
```

Server runs on http://0.0.0.0:8000

## Deployment

Optimized for deployment on Render, Railway, or similar platforms.

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python main.py`
- **Python Version**: 3.11+