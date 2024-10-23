#!/bin/bash
set -euo pipefail

function ml ()
{
    eval $($LMOD_DIR/ml_cmd "$@")
}

ml python

# cd $HOME/golden-triangle
export PATH=$HOME/bin:$PATH
# source .venv/bin/activate

cd /scratch/prj/dh_golden_triangle/full_data
aws s3 sync "s3://openalex" openalex-snapshot  --exclude "*" --include "data/works/*" --no-sign-request

