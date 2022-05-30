import glob
import logging
import os
import json
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from tools.common import seed_everything,json_to_text
from tools.common import init_logger, logger

from models.transformers import WEIGHTS_NAME, BertConfig, AlbertConfig
from models.bert_for_ner import BertCrfForNer
from models.albert_for_ner import AlbertCrfForNer
from processors.utils_ner import CNerTokenizer, get_entities
from processors.ner_seq import convert_examples_to_features
from processors.ner_seq import ner_processors as processors
from processors.ner_seq import collate_fn
from metrics.ner_metrics import SeqEntityScore
from tools.finetuning_argparse import get_argparse

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertCrfForNer, CNerTokenizer),
    'albert': (AlbertConfig, AlbertCrfForNer, CNerTokenizer)
}

# 函数调用：global_step, tr_loss = train(args, train_dataset, model, tokenizer)
def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    # 计算批量大小
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # 默认args.local_rank == -1，采样器为RandomSampler
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # 可迭代对象
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    # args.max_steps默认为-1
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:    
        # len(train_dataloader)代表一轮里的批量数
        # args.gradient_accumulation_steps: Number of updates steps to accumulate before performing a backward/update pass
        # 根据网上所说，args.gradient_accumulation_steps = n被用于显存不够时，通过将一个批量的batch size均分为n份运行n次，将这n次的梯度累计起来形成一次更新所用的梯度
        # 但这里args.gradient_accumulation_steps = n貌似表示的是，n个批量以后，才累计n个批量所有梯度执行一次更新，相当于显存不变大的情况下，批量大小变大n倍
        # 经试验，args.gradient_accumulation_steps变大一倍，批量大小变大一倍，总的更新次数减少一半
        # 此处取默认，args.gradient_accumulation_steps = 1
        # args.num_train_epochs是训练的epoch数
        # t_total是训练过程总的更新步数
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        # print("*******************************************************************")
        # print("len(train_dataloader):", len(train_dataloader))
        # print("args.gradient_accumulation_steps:", args.gradient_accumulation_steps)
        # print("args.num_train_epochs:", args.num_train_epochs)
        # print("t_total:", t_total)
        # # len(train_dataloader): 224
        # # args.gradient_accumulation_steps: 1
        # # args.num_train_epochs: 5.0
        # # t_total: 1120.0
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    # 优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # 学习率调度
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    # 默认情况下不存在
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    # fp16 indicates Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit
    # 默认False
    # print("******************args.fp16********************", args.fp16)
    # # ******************args.fp16******************** False
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    # 3090上n_gpu为2
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    # 默认local_rank = -1
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
                )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    # 记录总的更新的步数
    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # print("**********************\"checkpoint\" detected in args.model_name_or_path")
        # # 没有打印
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    # seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    # 遍历所有epoch
    for _ in range(int(args.num_train_epochs)):
        # 进度条
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        # 遍历所有iterations
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            # 训练模式
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], 'input_lens': batch[4]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            # if step == 1:
            #     print("type(outputs):", type(outputs))
            #     # type(outputs): <class 'tuple'>
            #     print("np.shape(outputs):", np.shape(outputs))
            #     print("np.shape(outputs[0]):", np.shape(outputs[0]))
            #     print("np.shape(outputs[1]):", np.shape(outputs[1]))
            #     # 元组第一个元素是loss，第二个元素是输出结果
            #     # np.shape(outputs): (2,)
            #     # 两张卡，两个loss
            #     # np.shape(outputs[0]): torch.Size([2])
            #     # 第一个维度48代表batch size，第二个维度52代表句子最大长度，第三个维度34代表分类数
            #     # np.shape(outputs[1]): torch.Size([48, 52, 34])
            # 计算并处理loss
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # 反向传播
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # 进度条
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            # 判断是否到了需要更新的步骤
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # Update learning rate schedule
                scheduler.step()  
                # 更新参数
                optimizer.step()
                # 梯度置为0
                model.zero_grad()
                global_step += 1
                # 默认设置下，每轮batch数为224，args.logging_steps为448，所以一般两轮验证一次
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    print(" ")
                    # 默认args.local_rank为-1
                    if args.local_rank == -1:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        evaluate(args, model, tokenizer)
                # 默认设置下，每轮batch数为224，args.save_steps为448，所以一般两轮存一次权重到以步数结尾的文件夹里
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
        logger.info("\n")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='dev')
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], 'input_lens': batch[4]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            tags = model.crf.decode(logits, inputs['attention_mask'])
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        input_lens = inputs['input_lens'].cpu().numpy().tolist()
        tags = tags.squeeze(0).cpu().numpy().tolist()
        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == input_lens[i] - 1:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(args.id2label[out_label_ids[i][j]])
                    temp_2.append(args.id2label[tags[i][j]])
        pbar(step)
    logger.info("\n")
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    return results


def predict(args, model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='test')
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)
    results = []
    # output_predict_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    output_predict_file = os.path.join(pred_output_dir, prefix, "test_prediction_{}.json".format(args.output_time))
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")

    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None, 'input_lens': batch[4]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            logits = outputs[0]
            tags = model.crf.decode(logits, inputs['attention_mask'])
            tags  = tags.squeeze(0).cpu().numpy().tolist()
        preds = tags[0][1:-1]  # [CLS]XXXX[SEP]
        label_entities = get_entities(preds, args.id2label, args.markup)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join([args.id2label[x] for x in preds])
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(step)
    logger.info("\n")
    with open(output_predict_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')
    if args.task_name == 'cluener':
        # output_submit_file = os.path.join(pred_output_dir, prefix, "test_submit.json")
        output_submit_file = os.path.join(pred_output_dir, prefix, "testd_submit_{}.json".format(args.output_time))
        test_text = []
        with open(os.path.join(args.data_dir,"test.json"), 'r') as fr:
            for line in fr:
                test_text.append(json.loads(line))
        test_submit = []
        for x, y in zip(test_text, results):
            json_d = {}
            json_d['id'] = x['id']
            json_d['label'] = {}
            entities = y['entities']
            words = list(x['text'])
            if len(entities) != 0:
                for subject in entities:
                    tag = subject[0]
                    start = subject[1]
                    end = subject[2]
                    word = "".join(words[start:end + 1])
                    if tag in json_d['label']:
                        if word in json_d['label'][tag]:
                            json_d['label'][tag][word].append([start, end])
                        else:
                            json_d['label'][tag][word] = [[start, end]]
                    else:
                        json_d['label'][tag] = {}
                        json_d['label'][tag][word] = [[start, end]]
            test_submit.append(json_d)
        json_to_text(output_submit_file,test_submit)

def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_crf-{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.train_max_seq_length if data_type == 'train' \
                                                    else args.eval_max_seq_length,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset


def main():
    # 传入参数
    args = get_argparse().parse_args()

    # 输出结果与日志的路径
    # --model_type=bert
    # TASK_NAME="cluener"
    # --task_name=$TASK_NAME
    # if not os.path.exists(args.output_dir):
    #     os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
        
    # print(f"***************args.local_rank: {args.local_rank}***************")
    # print(f"***************args.no_cuda: {args.no_cuda}***************")
    # # ***************args.local_rank: -1***************
    # # ***************args.no_cuda: False***************
        
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
        
    # Setup CUDA, GPU & distributed training
    # 默认情况，args.local_rank: -1，args.no_cuda: False，进入这个if
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    # 默认情况不进入这个else
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )
    # Process rank: -1, device: cuda, n_gpu: 2, distributed training: False, 16-bits training: False
    
    # Set seed
    # seed_everything(args.seed)
    # Prepare NER task
    args.task_name = args.task_name.lower()
    # from ... import ner_processors as processors
    # ner_processors = {
    #     "cner": CnerProcessor,
    #     'cluener':CluenerProcessor,
    #     'duiener': DuienerProcessor
    # }
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    # 下面的一行代码已经实例化了DuienerProcessor类对象processor，因为有括号
    processor = processors[args.task_name]()
    # for clue:
    # print(f"***************processor: {processor}***************")
    # # ***************processor: <processors.ner_seq.CluenerProcessor object at 0x7f3965534160>***************
    label_list = processor.get_labels()
    # for clue:
    # print(f"***************label_list: {label_list}***************")
    # # ***************label_list: ['X', 'B-address', 'B-book', 'B-company', 'B-game', 'B-government', 'B-movie', 'B-name', 'B-organization', 'B-position', 'B-scene', 'I-address', 'I-book', 'I-company', 'I-game', 'I-government', 'I-movie', 'I-name', 'I-organization', 'I-position', 'I-scene', 'S-address', 'S-book', 'S-company', 'S-game', 'S-government', 'S-movie', 'S-name', 'S-organization', 'S-position', 'S-scene', 'O', '[START]', '[END]']***************
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # 默认情况，args.local_rank: -1，不进入这个if
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    # args.model_type默认为bert
    args.model_type = args.model_type.lower()
    
    # MODEL_CLASSES = {
    #     ## bert ernie bert_wwm bert_wwwm_ext
    #     'bert': (BertConfig, BertCrfForNer, CNerTokenizer),
    #     'albert': (AlbertConfig, AlbertCrfForNer, CNerTokenizer)
    # }
    # config_class, model_class, tokenizer_class默认分别为BertConfig, BertCrfForNer, CNerTokenizer
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # print(f"***************config_class: {config_class}***************")
    # # ***************config_class: <class 'models.transformers.configuration_bert.BertConfig'>***************
    # print(f"***************model_class: {model_class}***************")
    # # ***************model_class: <class 'models.bert_for_ner.BertCrfForNer'>***************
    # print(f"***************tokenizer_class: {tokenizer_class}***************")
    # # ***************tokenizer_class: <class 'processors.utils_ner.CNerTokenizer'>***************
    
    # args.model_name_or_path = 'bert-base-chinese'
    
    # config是一个BertConfig类对象
    # from_pretrained is a classmethod
    # from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
    # Instantiate a :class:`~transformers.PretrainedConfig` (or a derived class) from a pre-trained model configuration
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, cache_dir=args.cache_dir if args.cache_dir else None, )
    
    # tokenizer调用from_pretrained函数，传入args.model_name_or_path，就包含了vocab的信息
    # from_pretrained is a classmethod
    # from_pretrained(cls, *inputs, **kwargs)
    # _from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs)
    # Instantiate a :class:`~transformers.PreTrainedTokenizer` (or a derived class, in this case, CNerTokenizer) from a predefined tokenizer.
    # do_lower_case: Set this flag if you are using an uncased model
    # cased: 支持大小写，uncased: 仅支持小写(词表中只有小写,数据处理时需要进行lower处理)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None, )
    
    
    # Instantiate a pretrained pytorch model from a pre-trained model configuration
    # def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config, cache_dir=args.cache_dir if args.cache_dir else None)
    # 默认情况，args.local_rank: -1，不进入这个if
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # 模型放到设备上
    model.to(args.device)
    
    logger.info("Training/evaluation parameters %s", args)
    
    # Training
    if args.do_train:
        # 得到训练数据集，load_and_cache_examples函数返回TensorDataset类对象
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        # 默认不进入此if
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    # predict
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        # checkpoints = ['/home/user/xiongdengrui/cluener/CLUENER2020/pytorch_version/outputs/cluener_output/20220513202105_roberta2bert']
        checkpoints = [args.output_dir]
        # 默认不进入此if
        if args.predict_checkpoints > 0:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(args.predict_checkpoints)]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            predict(args, model, tokenizer, prefix=prefix)


if __name__ == "__main__":
    main()
