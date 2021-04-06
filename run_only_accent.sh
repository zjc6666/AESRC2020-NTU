#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


cuda_cmd="slurm.pl --quiet --exclude=node0[2-7]"
decode_cmd="slurm.pl --quiet --exclude=node0[1-4,8]"
cmd="slurm.pl --quiet --exclude=node0[1-4]"
# general configuration
backend=pytorch
steps=1
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=20
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
log=100
vocab_size=2000
bpemode=bpe
# feature configuration
do_delta=false

train_config=conf/espnet_train.yaml
train_track1_config=conf/e2e_asr_transformer_only_accent.yaml
lm_config=conf/espnet_lm.yaml
decode_config=conf/espnet_decode.yaml
preprocess_config=conf/espnet_specaug.yaml

# rnnlm related
lm_resume=         # specify a snapshot file to resume LM training
lmtag=0             # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=5

# others
accum_grad=2
n_iter_processes=2
lsm_weight=0.0
epochs=30
elayers=6
batch_size=32

# exp tag
tag="base" # tag for managing experiments.

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
# set -u
set -o pipefail

. utils/parse_options.sh || exit 1;
. path2.sh

steps=$(echo $steps | perl -e '$steps=<STDIN>;  $has_format = 0;
  if($steps =~ m:(\d+)\-$:g){$start = $1; $end = $start + 10; $has_format ++;}
        elsif($steps =~ m:(\d+)\-(\d+):g) { $start = $1; $end = $2; if($start == $end){}
        elsif($start < $end){ $end = $2 +1;}else{die;} $has_format ++; }
      if($has_format > 0){$steps=$start;  for($i=$start+1; $i < $end; $i++){$steps .=":$i"; }}
      print $steps;' 2>/dev/null)  || exit 1

if [ ! -z "$steps" ]; then
#  echo $steps
  for x in $(echo $steps|sed 's/[,:]/ /g'); do
     index=$(printf "%02d" $x);
    # echo $index
     declare step$index=1
  done
fi

data=$1 # data
exp=$2 # exp-espnet-epoch-50
train_set="train"
recog_set="cv_all test"
valid_set="valid"
# recog_set="cv/UK cv/US cv/CHN cv/JPN cv/KR cv/RU cv/IND cv/PT"

if [ ! -z $step01 ]; then
   echo "extracting filter-bank features and cmvn"
   for i in $recog_set $valid_set $train_set;do # $train_dev $recog_set;do
      utils/fix_data_dir.sh $data/$i
      steps/make_fbank_pitch.sh --cmd "$cmd" --nj $nj --write_utt2num_frames true \
          $data/$i $data/$i/feats/log $data/$i/feats/ark
      utils/fix_data_dir.sh $data/$i
   done

   compute-cmvn-stats scp:$data/${train_set}/feats.scp $data/${train_set}/cmvn.ark
   echo "step01 Extracting filter-bank features and cmvn Done"
fi

### prepare for track1
if [ ! -z $step06 ]; then
    for x in $train_set $valid_set $recog_set;do
        awk '{printf "%s %s\n", $1, $1 }' $data/$x/text > $data/$x/spk2utt.utt
        cp $data/$x/spk2utt.utt $data/$x/utt2spk.utt
        compute-cmvn-stats --spk2utt=ark:$data/$x/spk2utt.utt scp:$data/$x/feats.scp \
            ark,scp:$data/$x/cmvn_utt.ark,$data/$x/cmvn_utt.scp
        local/tools/dump_spk_yzl23.sh --cmd "$cmd" --nj 20 \
            $data/$x/feats.scp $data/$x/cmvn_utt.scp \
            $data/$x/dump_utt/log $data/$x/dump_utt $data/$x/utt2spk.utt
    done
    echo "### step 06 dump utt Done"
fi
### prepare for track1
if [ ! -z $step07 ]; then
    for x in $recog_set $valid_set;do
        local/tools/data2json.sh --nj 20 --cmd "$cmd" --feat $data/$x/dump_utt/feats.scp --text $data/$x/utt2accent --oov 8 $data/$x $data/lang/accent.dict > $data/$x/${train_set}_accent.json
    done
fi

dict=$data/lang/accent.dict
epochs=30
if [ ! -z $step10 ]; then
    train_set=train
    elayers=3
    expname=${train_set}_${elayers}_layers_verification_${backend}
    expdir=$exp/${expname}
    epoch_stage=0
    mkdir -p ${expdir}
    echo "stage 2: Network Training"
    ngpu=1
    if  [ ${epoch_stage} -gt 0 ]; then
        echo "stage 6: Resume network from epoch ${epoch_stage}"
        resume=${exp}/${expname}/results/snapshot.ep.${epoch_stage}
    fi
    train_track1_config=conf/track1_accent_transformer.yaml
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_track1_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --report-interval-iters ${log} \
        --accum-grad ${accum_grad} \
        --n-iter-processes ${n_iter_processes} \
        --elayers ${elayers} \
        --lsm-weight ${lsm_weight} \
        --epochs ${epochs} \
        --batch-size ${batch_size} \
        --dict ${dict} \
        --num-save-ctc 0 \
        --train-json $data/${train_set}/${train_set}_accent.json \
        --valid-json $data/${valid_set}/${train_set}_accent.json
fi

# pretrained asr model
pretrained_model=/home/maison2/lid/zjc/w2020/AESRC2020/result/track2-accent-160/train_12enc_6dec_pytorch/results/model.val5.avg.best
if [ ! -z $step13 ]; then
    train_set=train
    elayers=12
    expname=${train_set}_${elayers}_layers_init_libri_${backend}
    expdir=$exp/${expname}
    epoch_stage=0
    mkdir -p ${expdir}
    echo "stage 2: Network Training"
    ngpu=1
    if  [ ${epoch_stage} -gt 0 ]; then
        echo "stage 6: Resume network from epoch ${epoch_stage}"
        resume=${exp}/${expname}/results/snapshot.ep.${epoch_stage}
    fi
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_track1_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --report-interval-iters ${log} \
        --accum-grad ${accum_grad} \
        --n-iter-processes ${n_iter_processes} \
        --elayers ${elayers} \
        --lsm-weight ${lsm_weight} \
        --epochs ${epochs} \
        --batch-size ${batch_size} \
        --dict ${dict} \
        --num-save-ctc 0 \
        --train-json $data/${train_set}/${train_set}_accent.json \
        --valid-json $data/${train_valid}/${train_set}_accent.json \
        ${pretrained_model:+--pretrained-model $pretrained_model}

fi
if [ ! -z $step15 ]; then
    echo "stage 2: Decoding"
    nj=100
    for expname in train_3_layers_init_accent_verification_2_pytorch;do
    for recog_set in test cv_all;do
    decode_dir=decode_${recog_set}_${log_step}
    use_valbest_average=true
    expdir=$exp/$expname
    
    if [[ $(get_yaml.py ${train_track1_config} model-module) = *transformer* ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            [ -f ${expdir}/results/model.val5.avg.best ] && rm ${expdir}/results/model.val5.avg.best
            recog_model=model.val${n_average}_${log_step}.avg.best
            opt="--log ${expdir}/results/log"
        else
            [ -f ${expdir}/results/model.last5.avg.best ] && rm ${expdir}/results/model.last5.avg.best
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        # recog_model=model.acc.best
        echo "$recog_model"
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}
    fi
    # split data
    dev_root=$data/${recog_set}
    splitjson.py --parts ${nj} ${dev_root}/${train_set}_accent.json
    #### use CPU for decoding
    ngpu=0

    ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_recog.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --batchsize 0 \
        --recog-json ${dev_root}/split${nj}utt/${train_set}_accent.JOB.json \
        --result-label ${expdir}/${decode_dir}/${train_set}_accent.JOB.json \
        --model ${expdir}/results/${recog_model} 
    concatjson.py ${expdir}/${decode_dir}/${train_set}_accent.*.json >  ${expdir}/${decode_dir}/${train_set}_accent.json
    python local/tools/parse_track1_jsons.py  ${expdir}/${decode_dir}/${train_set}_accent.json ${expdir}/${decode_dir}/result.txt
    python local/tools/parse_track1_jsons.py  ${expdir}/${decode_dir}/${train_set}_accent.json ${expdir}/${decode_dir}/result.txt > ${expdir}/${decode_dir}/acc.txt
    done
    done
    echo "Decoding finished"
fi

