#!/bin/bash

# on create cloud:
cd /scratch/prj/dh_golden_triangle/full_data
aws s3 sync "s3://openalex" openalex-snapshot  --exclude "*" --include "data/works/*" --no-sign-request
ln -s /scratch/prj/dh_golden_triangle/full_data/openalex-snapshot ~/golden-triangle/data/openalex-full-data
