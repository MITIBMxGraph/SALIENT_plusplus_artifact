#!/usr/bin/env bash

THIS_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PYTHONPATH="$THIS_SCRIPT_DIR/.." python -m caching.experiment_communication_caching $@

exit 0
