#!/bin/bash

aws s3 sync "s3://openalex" "../data/openalex-full-data" --exclude "*" --include "data/works/*" --no-sign-request
