CURRENT_DIR=`pwd`
# export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/roberta_wwm_large_ext
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base-chinese
# export GLUE_DIR=$CURRENT_DIR/CLUEdatasets
export GLUE_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="cluener"
time=$(date "+%Y%m%d%H%M%S")
output_time=${time}

python run_ner_softmax.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --loss_type=ce \
  --do_predict \
  --data_dir=$GLUE_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=24 \
  --per_gpu_eval_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=10.0 \
  --logging_steps=224 \
  --save_steps=224 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/${output_time} \
  --overwrite_output_dir \
  --seed=42

#python run_ner_softmax.py \
#  --model_type=bert \
#  --model_name_or_path=$BERT_BASE_DIR \
#  --task_name=$TASK_NAME \
#  --do_predict \
#  --do_lower_case \
#  --loss_type=ce \
#  --data_dir=$GLUE_DIR/${TASK_NAME}/ \
#  --train_max_seq_length=128 \
#  --eval_max_seq_length=512 \
#  --per_gpu_train_batch_size=24 \
#  --per_gpu_eval_batch_size=24 \
#  --learning_rate=3e-5 \
#  --num_train_epochs=4.0 \
#  --logging_steps=224 \
#  --save_steps=224 \
#  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
#  --overwrite_output_dir \
#  --seed=42