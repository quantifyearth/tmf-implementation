FROM golang:bullseye AS littlejohn
RUN git clone https://github.com/carboncredits/littlejohn.git
WORKDIR littlejohn
RUN go build

FROM ghcr.io/osgeo/gdal:ubuntu-small-3.10.1

COPY --from=littlejohn /go/littlejohn/littlejohn /bin/littlejohn

RUN apt-get update -qqy && \
apt-get install -qy \
	git \
	libpq-dev \
	python3-pip \
&& rm -rf /var/lib/apt/lists/* \
&& rm -rf /var/cache/apt/*

# You must install numpy before anything else otherwise
# gdal's python bindings are sad. Pandas we full out as its slow
# to build, and this means it'll be cached
RUN pip config set global.break-system-packages true
RUN pip install gdal[numpy]==3.10.1

WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./
RUN make lint && make type && make test
# Show the dependencies that passed in the log
RUN pip freeze
