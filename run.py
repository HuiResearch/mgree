# -*- coding: UTF-8 -*-
# author    : huanghui
import os
import logging
import argparse
from models import MgreeTrainer, HetGraphTrainer
from utils import check_dir, set_seed, write_to_json, load_from_json

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
logger = logging.getLogger('root')

TRAINER_MAP = {
    'mgree': MgreeTrainer,
    'hetgraph': HetGraphTrainer
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adv_type",
        default=None,
        type=str,
    )
    parser.add_argument('--trainer_type', type=str, default='mgree', choices=list(TRAINER_MAP.keys()))
    parser.add_argument('--data_dir', type=str, default="data/ace05",
                        help="path to the preprocessed dataset")
    parser.add_argument('--model_dir', type=str, default='bert-base-uncased',
                        help="the base model name ")

    parser.add_argument('--output_dir', type=str, default='outputs',
                        help="output directory of the entity model")

    parser.add_argument('--project', type=str, default="event_detection",
                        help="project name for wandb")
    parser.add_argument('--run_name', type=str, default=None,
                        help="run name for wandb")

    parser.add_argument('--per_gpu_train_batch_size', type=int, default=32,
                        help="batch size during training")
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=8,
                        help="batch size during inference")
    parser.add_argument('--max_query_length', type=int, default=76,
                        help="the max query length for mrc input")
    parser.add_argument('--max_length', type=int, default=256,
                        help="the max seq length for the input")
    parser.add_argument('--context_window', type=int, default=0,
                        help="context window")
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help="learning rate for the BERT encoder")
    parser.add_argument('--task_learning_rate', type=float, default=1e-3,
                        help="learning rate for task-specific parameters, i.e., classification head")

    parser.add_argument('--max_grad_norm', type=float, default=1.,
                        help="Max gradient norm.")
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help="Weight decay for AdamW if we apply some.")
    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                        help="Epsilon for AdamW optimizer.")
    parser.add_argument('--lr_scheduler_type', type=str, default="linear",
                        choices=[None, 'linear', 'cosine',
                                 'cosine_with_restarts',
                                 'polynomial', 'constant',
                                 'constant_with_warmup'],
                        help="The scheduler type to use.")
    parser.add_argument('--warmup_ratio', type=float, default=0.2,
                        help="Linear warmup over warmup_ratio fraction of total steps.")

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument('--num_train_epochs', type=int, default=10,
                        help="number of the training epochs")
    parser.add_argument('--eval_per_epoch', type=int, default=1,
                        help="how often evaluating the trained model on dev set during training")
    parser.add_argument('--save_per_epoch', type=int, default=1,
                        help="how often saving the model during training")

    parser.add_argument('--do_train', action='store_true',
                        help="whether to run training")
    parser.add_argument('--do_eval', action='store_true',
                        help="whether to run evaluation")
    parser.add_argument('--do_test', action='store_true',
                        help="whether to evaluate on test set")
    parser.add_argument('--search', action='store_true',
                        help="whether to search best threshold")
    parser.add_argument('--mixed_precision', default='no',  # fp16 or bf16
                        help="whether to use mixed precision")
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0, help='num workers for dataloader')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--swa_fn', type=str, default='ema', choices=['no', 'ema'])
    parser.add_argument('--swa_start_epoch', type=int, default=1)
    parser.add_argument('--ema_decay_1', type=float, default=0.9)
    parser.add_argument('--ema_decay_2', type=float, default=0.999)
    parser.add_argument('--use_wandb', action='store_true', help="whether to use wandb")
    parser.add_argument('--single_word', action='store_true', help="触发词是否为单个词")
    parser.add_argument('--compile', action='store_true', help="使用torch.compile优化")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    set_seed(args.seed)
    if "ace" in args.data_dir:
        args.single_word = not ('05+' in args.data_dir)
    args.train_data = os.path.join(args.data_dir, 'train.json')
    args.dev_data = os.path.join(args.data_dir, 'dev.json')
    args.test_data = os.path.join(args.data_dir, 'test.json')

    check_dir(args.output_dir)

    logger.info(args)

    if args.do_train:
        trainer = TRAINER_MAP[args.trainer_type](args, from_scratch=True)
        test_dataset = trainer.create_dataset(args.test_data, training=False)
        eval_dataset = trainer.create_dataset(args.dev_data, training=False)
        train_dataset = trainer.create_dataset(args.train_data, training=True)
        trainer.train(train_dataset, eval_dataset, test_dataset)

    if args.do_eval or args.do_test or args.search:
        trainer = TRAINER_MAP[args.trainer_type](args, from_scratch=False)
        if args.search:
            eval_dataset = trainer.create_dataset(args.dev_data, training=False)
            thresholds = trainer.search_threshold(eval_dataset, scope=50, ratio=0.1)
            write_to_json(thresholds, os.path.join(args.output_dir, 'threshold.json'))
        else:
            if os.path.exists(os.path.join(args.output_dir, 'threshold.json')):
                thresholds = load_from_json(os.path.join(args.output_dir, 'threshold.json'))
                trainer.thresholds = thresholds
        if args.do_eval:
            eval_dataset = trainer.create_dataset(args.dev_data, training=False)
            trainer.evaluate(eval_dataset, prefix='eval', save_result=True, log=True)
        if args.do_test:
            eval_dataset = trainer.create_dataset(args.test_data, training=False)
            trainer.evaluate(eval_dataset, prefix='test', save_result=True, log=True)


if __name__ == '__main__':
    main()
