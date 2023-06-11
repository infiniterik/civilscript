# CivilScript

## Instructions

1. Run Stance-Detection server via docker container.
- `docker run -it -p 5000:5000 --network host detection_app flask run`

2. Set a `.env` file with keys for `OPENAI_API_KEY` and `STANCE_SERVER`

```
pip install -r requirements.txt
streamlit run streamlit.py
```
