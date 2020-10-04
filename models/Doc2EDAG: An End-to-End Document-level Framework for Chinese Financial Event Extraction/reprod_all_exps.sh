#! /bin/bash

DATA_DIR=./Data
EXP_DIR=./Exps
COMMON_TASK_NAME=HelloEDAG
RESUME_TRAIN=True
SAVE_CPT=True
N_EPOCH=100
TRAIN_BS=64
EVAL_BS=2
NUM_GPUS=8
GRAD_ACC_STEP=8
# There is one parameter update for every GRAD_ACC_STEP back-propagation steps,
# so the real runtime batch size is TRAIN_BS / GRAD_ACC_STEP.
# In this way, we can achieve a large batch training with only a few GPUs


# Doc2EDAG Models: Doc2EDAG, GreedyDec
MODEL_TYPE=Doc2EDAG
MODEL_STR=Doc2EDAG
echo "---> ${MODEL_STR} Run"
./train_multi.sh ${NUM_GPUS} --resume_latest_cpt ${RESUME_TRAIN} --save_cpt_flag ${SAVE_CPT} \
    --data_dir ${DATA_DIR} --exp_dir ${EXP_DIR} --task_name ${COMMON_TASK_NAME} --num_train_epochs ${N_EPOCH} \
    --train_batch_size ${TRAIN_BS} --gradient_accumulation_steps ${GRAD_ACC_STEP} --eval_batch_size ${EVAL_BS} \
    --model_type ${MODEL_TYPE} --cpt_file_name ${MODEL_STR} --add_greedy_dec True


# DCFEE Baselines: DCFEE-O, DCFEE-M
MODEL_TYPE=DCFEE
MODEL_STR=DCFEE
echo "---> ${MODEL_STR} Run"
./train_multi.sh ${NUM_GPUS} --resume_latest_cpt ${RESUME_TRAIN} --save_cpt_flag ${SAVE_CPT} \
    --data_dir ${DATA_DIR} --exp_dir ${EXP_DIR} --task_name ${COMMON_TASK_NAME} --num_train_epochs ${N_EPOCH} \
    --train_batch_size ${TRAIN_BS} --gradient_accumulation_steps ${GRAD_ACC_STEP} --eval_batch_size ${EVAL_BS} \
    --model_type ${MODEL_TYPE} --cpt_file_name ${MODEL_STR}


# Ablation Tests of Doc2EDAG
MODEL_TYPE=Doc2EDAG

# Ablation Test 1
MODEL_STR=Doc2EDAG-NoPathMem
echo "---> ${MODEL_STR} Run"
./train_multi.sh ${NUM_GPUS} --resume_latest_cpt ${RESUME_TRAIN} --save_cpt_flag ${SAVE_CPT} \
    --data_dir ${DATA_DIR} --exp_dir ${EXP_DIR} --task_name ${COMMON_TASK_NAME} --num_train_epochs ${N_EPOCH} \
    --train_batch_size ${TRAIN_BS} --gradient_accumulation_steps ${GRAD_ACC_STEP} --eval_batch_size ${EVAL_BS} \
    --model_type ${MODEL_TYPE} --cpt_file_name ${MODEL_STR} --use_path_mem False

# Ablation Test 2
MODEL_STR=Doc2EDAG-NoScheduledSampling
echo "---> ${MODEL_STR} Run"
./train_multi.sh ${NUM_GPUS} --resume_latest_cpt ${RESUME_TRAIN} --save_cpt_flag ${SAVE_CPT} \
    --data_dir ${DATA_DIR} --exp_dir ${EXP_DIR} --task_name ${COMMON_TASK_NAME} --num_train_epochs ${N_EPOCH} \
    --train_batch_size ${TRAIN_BS} --gradient_accumulation_steps ${GRAD_ACC_STEP} --eval_batch_size ${EVAL_BS} \
    --model_type ${MODEL_TYPE} --cpt_file_name ${MODEL_STR} --use_scheduled_sampling False

# Ablation Test 3
MODEL_STR=Doc2EDAG-NoDocEnc
echo "---> ${MODEL_STR} Run"
./train_multi.sh ${NUM_GPUS} --resume_latest_cpt ${RESUME_TRAIN} --save_cpt_flag ${SAVE_CPT} \
    --data_dir ${DATA_DIR} --exp_dir ${EXP_DIR} --task_name ${COMMON_TASK_NAME} --num_train_epochs ${N_EPOCH} \
    --train_batch_size ${TRAIN_BS} --gradient_accumulation_steps ${GRAD_ACC_STEP} --eval_batch_size ${EVAL_BS} \
    --model_type ${MODEL_TYPE} --cpt_file_name ${MODEL_STR} --use_doc_enc False

# Ablation Test 4
MODEL_STR=Doc2EDAG-NoFPPenalty
echo "---> ${MODEL_STR} Run"
./train_multi.sh ${NUM_GPUS} --resume_latest_cpt ${RESUME_TRAIN} --save_cpt_flag ${SAVE_CPT} \
    --data_dir ${DATA_DIR} --exp_dir ${EXP_DIR} --task_name ${COMMON_TASK_NAME} --num_train_epochs ${N_EPOCH} \
    --train_batch_size ${TRAIN_BS} --gradient_accumulation_steps ${GRAD_ACC_STEP} --eval_batch_size ${EVAL_BS} \
    --model_type ${MODEL_TYPE} --cpt_file_name ${MODEL_STR} --neg_field_loss_scaling 1.0

