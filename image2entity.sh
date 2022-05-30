CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/pytorch_version/prev_trained_model/chinese-roberta-wwm-ext
export GLUE_DIR=$CURRENT_DIR/pytorch_version/datasets
TASK_NAME="cluener"
time=$(date "+%Y%m%d%H%M%S")
output_time=${time}

python ./ocr/EasyOCR/demo.py

# **************************test**************************
# 根据test数据集是自己做的还是CLUENER的，有三处地方需要相应修改(若自己做的就命名为test则不用改下面三个)：
# 后续回来加上(重点！！！)：若自己做的不命名为test，实际得到的测试结果还是基于test的，可能是哪里的feature没改过来
# 1 run_ner_crf.py的predict函数定义中，路径output_predict_file
# 2 run_ner_crf.py的predict函数定义中，路径output_submit_file
# 3 ner_seq.py的class CluenerProcessor中，函数get_test_examples的返回值
# 另外注意，datasets/cluener里面有很多缓存的features，如果换测试集，要记得清除需要清除的缓存

python ./pytorch_version/run_ner_crf.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_predict \
  --do_lower_case \
  --data_dir=$GLUE_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=24 \
  --per_gpu_eval_batch_size=24 \
  --learning_rate=1e-5 \
  --num_train_epochs=8.0 \
  --logging_steps=224 \
  --save_steps=2000 \
  --output_dir='./pytorch_version/outputs/cluener_output/20220512171954_roberta1' \
  --overwrite_output_dir \
  --overwrite_cache \
  --seed=42 \
  --output_time=${output_time}

# **************************test**************************