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

python3 /users/k21191796/golden-triangle/hpc-jobs/parse_jsonls/parse_jsonl.py
