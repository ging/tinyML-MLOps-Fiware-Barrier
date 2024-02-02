FROM python:3.11

WORKDIR /prediction-job

COPY ./requirements.txt ./

RUN pip install -r requirements.txt


COPY . .

CMD ["python", "train.py"]
