FROM ghcr.io/osgeo/gdal:ubuntu-small-3.6.4

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
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install gdal[numpy]==3.6.4

WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./
RUN make type && make test
# Show the dependencies that passed in the log
RUN pip freeze