install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
lint:
	pylint --disable=R,C $$(echo *.py)
test:
	python -m pytest -vv --cov=api test_api.py