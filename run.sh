#!/bin/bash
#SBATCH --job-name=back-translation
#SBATCH --partition=gpu
#SBATCH -n 1
#SBATCH --gres=gpu:3
#SBATCH --mem=80G
#SBATCH -c 4
#SBATCH -a 0

# =============== General Settings ===============
ROOT=$(pwd)
DATA_DIR="${ROOT}/data"
RESOURCE_DIR="${ROOT}/resource"
MONO_DIR="${DATA_DIR}/nonpara"

SIMP_TRAIN_PATH="${MONO_DIR}/simp_train.txt"
SIMP_DEV_PATH="${MONO_DIR}/simp_dev.txt"
COMP_TRAIN_PATH="${MONO_DIR}/comp_train.txt"
COMP_DEV_PATH="${MONO_DIR}/comp_dev.txt"
SUPERVISED_RATE=0

PARA_DIR="${DATA_DIR}/parallel"
DEV_DATASET="newsela"
PARA_DEV_PATH="${PARA_DIR}/${DEV_DATASET}/dev.txt"
PARA_TEST_PATH="${PARA_DIR}/${DEV_DATASET}/test.txt"
PARA_TRAIN_PATH="${PARA_DIR}/${DEV_DATASET}/train${SUPERVISED_RATE}.txt"
FRC_PATH="${RESOURCE_DIR}/enwiki_vocab_min200.txt"

EMBEDDING_PATH="${RESOURCE_DIR}/embedding.pkl"
LM_PATH="${RESOURCE_DIR}/lang.pkl"

COMP_FREQUENT_LIST="${RESOURCE_DIR}/denoise/frequent_comp.list"
SIMP_FREQUENT_LIST="${RESOURCE_DIR}/denoise/frequent_simp.list"
COMP_PPDB_RULES="${RESOURCE_DIR}/denoise/comp_rules.pkl"
SIMP_PPDB_RULES="${RESOURCE_DIR}/denoise/simp_rules.pkl"
STOP_LIST="${RESOURCE_DIR}/stop.list"

VOCAB_PATH="${DATA_DIR}/vocab.list"
CHECK_POINT="${ROOT}/checkpoints"

# ============ Transformer Parameters ============
D_MODEL=512
DFF=1024
ENC_LAYERS=3
DEC_LAYERS=3
ENC_SHARE=2
DEC_SHARE=0
HEADS=8
D_K=$[${D_MODEL}/${HEADS}]
D_V=$[${D_MODEL}/${HEADS}]
MAX_LEN=120
BEAM_SIZE=0

# =========== Denoising Parameters ===============
WORD_SHUFFLE=3
SHUFFLE_MODE='bigram'
WORD_DROPOUT=0.6
DROP_TYPE='Two'
WORD_REPLACE=0.9
ADDITIVE=1
RL_FINETUNE=0
LR=0.00005
GAMMA=0.9
DELATA=0.5

# ============== Training Settings ===============

NAME="${DEV_DATASET}_SUPERVISED_RATE_${SUPERVISED_RATE}_RL_FINETUNE_${RL_FINETUNE}"
DUMP_PATH="${ROOT}/saved_model/${NAME}"

mkdir -p ${DUMP_PATH}
mkdir -p ${CHECK_POINT}

python -u main.py \
--use_check 0 \
--n_enc_layers ${ENC_LAYERS} \
--n_dec_layers ${DEC_LAYERS} \
--d_model ${D_MODEL} \
--d_k ${D_K} \
--d_v ${D_V} \
--d_inner ${DFF} \
--len_max_seq ${MAX_LEN} \
--share_enc ${ENC_SHARE} \
--share_dec ${DEC_SHARE} \
--share_encdec_emb 1 \
--share_output_emb 1 \
--share_decpro_emb 0 \
--simp_train_path ${SIMP_TRAIN_PATH} \
--simp_dev_path ${SIMP_DEV_PATH} \
--comp_train_path ${COMP_TRAIN_PATH} \
--comp_dev_path ${COMP_DEV_PATH} \
--para_train_path ${PARA_TRAIN_PATH} \
--para_dev_path ${PARA_DEV_PATH} \
--para_test_path ${PARA_TEST_PATH} \
--vocab_path ${VOCAB_PATH} \
--lm_path ${LM_PATH} \
--word_shuffle ${WORD_SHUFFLE} \
--shuffle_mode ${SHUFFLE_MODE} \
--drop_type ${DROP_TYPE} \
--word_dropout ${WORD_DROPOUT} \
--word_replace ${WORD_REPLACE} \
--frc_path ${FRC_PATH} \
--batch_size 16 \
--epoch_size 150000 \
--freeze_enc_emb 0 \
--freeze_dec_emb 0 \
--pretrain_autoencoder 200000 \
--lr ${LR} \
--use_multi_process 1 \
--otf_num_processes 2 \
--otf_sync_params_every 300 \
--us_pretrain_embedding 1 \
--embedding_path ${EMBEDDING_PATH} \
--stopping_criterion 'sari' \
--name ${NAME} \
--dump_path ${DUMP_PATH} \
--checkpoint_path ${CHECK_POINT} \
--comp_frequent_list ${COMP_FREQUENT_LIST} \
--simp_frequent_list ${SIMP_FREQUENT_LIST} \
--comp_ppdb_rules ${COMP_PPDB_RULES} \
--simp_ppdb_rules ${SIMP_PPDB_RULES} \
--stoplist_path ${STOP_LIST} \
--beam_size ${BEAM_SIZE} \
--supervised_rate ${SUPERVISED_RATE} \
--rl_finetune  ${RL_FINETUNE} \
--use_pretrained_model 0 \
--gamma ${GAMMA} \
--delta ${DELATA} \
--otf_back_translation 1 \
--otf_autoencoding 1 \
--use_lm ${RL_FINETUNE} \
--additive ${ADDITIVE} \
