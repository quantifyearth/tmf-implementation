FROM golang:bullseye as build
WORKDIR /src
# Version 2.30.0
RUN git clone https://github.com/mmarkdown/mmark . && git checkout a7178c47f0bcc66d1b1970f1bd5607fb4c151e76
RUN go get && go build
WORKDIR /xml2rfc
FROM debian:bullseye
RUN apt-get update --fix-missing && \
    apt-get install -y --fix-missing \
        libpango-1.0-0 \
        libssl-dev \
        fontconfig \
        libharfbuzz0b \
        libpangoft2-1.0-0 \
        pkg-config \
        libxml2-utils \
        inotify-tools \
        groff \
        git \
        wget \
        unzip \
        locales \
        python3.10 \
        python3-pip && \
    locale-gen en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/* /var/log/dpkg.log && \
    apt-get autoremove -y && \
    apt-get clean -y
WORKDIR /xml2rfc
RUN mkdir -p ~/.fonts/opentype && \
    wget -q https://noto-website-2.storage.googleapis.com/pkgs/Noto-unhinted.zip && \
    unzip -q Noto-unhinted.zip -d ~/.fonts/opentype/ && \
    rm Noto-unhinted.zip && \
    wget -q https://fonts.google.com/download?family=Roboto%20Mono -O roboto-mono.zip && \
    unzip -q roboto-mono.zip -d ~/.fonts/opentype/ && \
    rm roboto-mono.zip && \
    ln -sf ~/.fonts/opentype/*.[to]tf /usr/share/fonts/truetype/ && \
    fc-cache -f
RUN pip3 install \
    tox \
    decorator \
    dict2xml \
    "pypdf>=3.2.1"
RUN echo break2
RUN git clone https://github.com/patricoferris/xml2rfc . && git checkout pact && pip install -r requirements.txt && make install
COPY --from=build /src/mmark /usr/bin/mmark
COPY ./build-draft.sh /usr/bin/build-draft
WORKDIR  /drafts