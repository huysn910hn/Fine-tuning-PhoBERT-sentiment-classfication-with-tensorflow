FROM python:3.10

WORKDIR /APP/app

COPY . /APP/app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]