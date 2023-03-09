FROM python:3.8

WORKDIR /

COPY . .

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

# expose port 8000 
EXPOSE 8000

# run 
CMD python3 -u test_server.py