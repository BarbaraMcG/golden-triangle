#!/bin/bash
set -euo pipefail

function ml ()
{
    eval $($LMOD_DIR/ml_cmd "$@")
}

ml python

cd $HOME/golden-triangle
source .venv/bin/activate
python3 api_src/async_version.py 
