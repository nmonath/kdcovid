#!/usr/bin/env bash

set -exu

mem=${1:-50000}
threads=${2:-1}

TIME=`(date +%Y-%m-%d-%H-%M-%S)`

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads


dataset="covid19"
model_name="reduce_vecs"
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
            bin/run_gather_sentence_embeddings.sh