#!/usr/bin/env bash

set -exu

chunk=$1

python -m kdcovid.encode_sentences --chunk $chunk