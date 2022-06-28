FROM pytorch/pytorch

WORKDIR ~

COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

WORKDIR /app

COPY . .

ENTRYPOINT ["python3"]