#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# basic settings
#TODO
stage=4       # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=16      # number of parallel jobs in feature extraction

# NOTE(kan-bayashi): renamed to conf to avoid conflict in parse_options.sh
#TODO
#conf=conf/parallel_wavegan.v1.yaml
#conf=conf/fmax12k_parallel_wavegan.v1.yaml
conf=conf/fmax8k_parallel_wavegan.v1.yaml

# directory path setting
download_dir=downloads # direcotry to save downloaded files
#TODO
#dumpdir=dump           # directory to dump features
#dumpdir=dump_kf           # directory to dump features
dumpdir=dump_kf8k           # directory to dump features
#TODO
#datasetname=csmsc
#datasetname=kuangfei
datasetname=kuangfei8k
#TODO
filename=dump/tacotron/tmp.npy
filename=dump_kf8k/tacotron/

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)
#TODO
# resume=exp/train_nodev_kuangfei_fmax12k_parallel_wavegan.v1/checkpoint-315000steps.pkl
resume=exp/train_nodev_kuangfei8k_fmax8k_parallel_wavegan.v1/checkpoint-265000steps.pkl

# decoding related setting
checkpoint="" # checkpoint path to be used for decoding
              # if not provided, the latest one will be used
              # (e.g. <path>/<to>/checkpoint-400000steps.pkl)

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

train_set="train_nodev" # name of training data directory
dev_set="dev"           # name of development data direcotry
eval_set="eval"         # name of evaluation data direcotry

set -euo pipefail

if [ "${stage}" -le -1 ] && [ "${stop_stage}" -ge -1 ]; then
    echo "Stage -1: Data download"
    local/data_download.sh "${download_dir}"
    exit
fi

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Data preparation"
    # data_root data_dir
    local/data_prep.sh \
        --train_set "${train_set}" \
        --dev_set "${dev_set}" \
        --eval_set "${eval_set}" \
        "${download_dir}/CSMSC" data
    exit
fi

stats_ext=$(grep -q "hdf5" <(yq ".format" "${conf}") && echo "h5" || echo "npy")
echo $stats_ext

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Feature extraction"
    # extract raw features
    pids=()
    for name in "${train_set}" "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        # src, num, dst --> wav.1.scp, segments.1
        utils/make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/raw"
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            parallel-wavegan-preprocess \
                --config "${conf}" \
                --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --segments "${dumpdir}/${name}/raw/segments.JOB" \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --verbose "${verbose}"
        echo "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished feature extraction."

    # calculate statistics for normalization
    echo "Statistics computation start. See the progress via ${dumpdir}/${train_set}/compute_statistics.log."
    ${train_cmd} "${dumpdir}/${train_set}/compute_statistics.log" \
        parallel-wavegan-compute-statistics \
            --config "${conf}" \
            --rootdir "${dumpdir}/${train_set}/raw" \
            --dumpdir "${dumpdir}/${train_set}" \
            --verbose "${verbose}"
    echo "Successfully finished calculation of statistics."

    # normalize and dump them
    pids=()
    for name in "${train_set}" "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/norm" ] && mkdir -p "${dumpdir}/${name}/norm"
        echo "Nomalization start. See the progress via ${dumpdir}/${name}/norm/normalize.*.log."
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/norm/normalize.JOB.log" \
            parallel-wavegan-normalize \
                --config "${conf}" \
                --stats "${dumpdir}/${train_set}/stats.${stats_ext}" \
                --rootdir "${dumpdir}/${name}/raw/dump.JOB" \
                --dumpdir "${dumpdir}/${name}/norm/dump.JOB" \
                --verbose "${verbose}"
        echo "Successfully finished normalization of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished normalization."
    exit
fi

if [ -z "${tag}" ]; then
    expdir="exp/${train_set}_${datasetname}_$(basename "${conf}" .yaml)"
else
    expdir="exp/${train_set}_${datasetname}_${tag}"
fi
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    #TODO
    # python local/taco2pwgdata.py
    echo "Stage 2: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    [ -e "${dumpdir}/${train_set}/stats.${stats_ext}" ] && cp "${dumpdir}/${train_set}/stats.${stats_ext}" "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        train="python -m parallel_wavegan.distributed.launch --nproc_per_node ${n_gpus} -c parallel-wavegan-train"
    else
        train="parallel-wavegan-train"
    fi
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --config "${conf}" \
            --train-dumpdir "${dumpdir}/${train_set}/norm" \
            --dev-dumpdir "${dumpdir}/${dev_set}/norm" \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --verbose "${verbose}"
    echo "Successfully finished training."
    exit
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
    pids=()
    for name in "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
            parallel-wavegan-decode \
                --dumpdir "${dumpdir}/${name}/norm" \
                --checkpoint "${checkpoint}" \
                --outdir "${outdir}/${name}" \
                --verbose "${verbose}"
        echo "Successfully finished decoding of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished decoding."
    exit
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: decode one"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    echo $checkpoint
    outdir=res
    python local/npy2h5feats.py ${filename}
    [ ! -e "${outdir}" ] && mkdir -p "${outdir}"
    [ "${n_gpus}" -gt 1 ] && n_gpus=1
    echo "Decoding start. See the progress via ${outdir}/decode.log."
            #--filename "$(dirname ${filename})/$(basename ${filename} .npy).h5" \
    ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/decode.log" \
        parallel-wavegan-decode \
            --dumpdir "${filename}" \
            --checkpoint "${checkpoint}" \
            --outdir "${outdir}" \
            --verbose "${verbose}"
    echo "Successfully finished decoding."
fi
echo "Finished."
