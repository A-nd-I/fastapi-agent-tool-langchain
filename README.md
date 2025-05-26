
### run back
uvicorn server:app --reload

### test
POST http://localhost:8000/calendar-agent/invoke
{
  "input": {
    "input": "What times are available on Monday?"
  }
}

### run streamlit
streamlit run streamlit_app.py