WORKDIR /

COPY . .

RUN ls -la

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

# expose port 8000 
EXPOSE 8000

CMD python3 -u server.py