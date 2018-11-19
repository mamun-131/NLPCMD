FROM    tensorflow/tensorflow:1.2.1-py3
ENV     PROJECT_DIR="/app"
WORKDIR /app
COPY    . .
RUN     pip install -r requirements.txt
EXPOSE  5000
CMD     ["python3", "nlp.py"]
