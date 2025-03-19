FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y 

COPY ./requirements.txt /app/
RUN pip3 install -r ./requirements.txt
RUN rm requirements.txt
EXPOSE 8501
COPY ./ /app/

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501"]