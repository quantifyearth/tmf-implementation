#!/bin/bash

set -e

COMMIT=`git rev-parse HEAD`
REPO=`git remote get-url origin`

docker build . -t tmf \
    "--label" "org.opencontainers.image.description=A python implementation of tmf-methodology" \
    "--label" "org.opencontainers.image.title=TMF" \
    "--label" "org.opencontainers.image.licenses=ISC" \
    "--label" "org.opencontainers.image.source=${REPO}" \
    "--label" "org.opencontainers.image.url=https://quantify.earth" \
    "--label" "org.opencontainers.image.revision=${COMMIT}" \
    "--label" "org.opencontainers.image.version="
