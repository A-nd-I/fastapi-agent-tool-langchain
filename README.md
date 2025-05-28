
### run back
uvicorn server:app --reload
uvicorn main:app --host 0.0.0.0 --port 8000

### test
POST http://localhost:8000/calendar-agent/invoke
{
  "input": {
    "input": "What times are available on Monday?"
  }
}

### run streamlit
streamlit run streamlit_app.py