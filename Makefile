install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

launch:
	streamlit run app.py
