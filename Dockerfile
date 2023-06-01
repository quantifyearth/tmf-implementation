FROM python:3.10-bullseye
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./
RUN make type && make test
# Show the dependencies that passed in the log
RUN pip freeze