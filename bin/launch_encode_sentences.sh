#!/usr/bin/env bash

set -exu

chunk=$1
mem=${2:-50000}
threads=${3:-1}

TIME=`(date +%Y-%m-%d-%H-%M-%S)`

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads



dataset="covid19"
model_name="encode"
job_name="$model_name-$dataset-$TIME"
log_dir=logs/$model_name/$dataset/$TIME
log_base=$log_dir/$job_name
mkdir -p $log_dir

sbatch -J $job_name \
            -e $log_base.err \
            -o $log_base.log \
            --cpus-per-task $threads \
            --partition=cpu \
            --ntasks=1 \
            --nodes=1 \
            --mem=$mem \
            --time=0-04:00 \
            bin/run_encode_sentences.sh $chunk