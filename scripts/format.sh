#!/usr/bin/env bash

set -e
set -x

black  src app.py
isort  src app.py