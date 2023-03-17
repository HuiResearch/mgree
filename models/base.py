# -*- coding: UTF-8 -*-
# author    : huanghui
import contextlib
import json
import logging
import shutil
import sys
import time
import os
import torch
import torch.nn as nn
from packaging import version
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from typing import List, Tuple, Union
import copy
from utils import (
    check_dir, write_to_json, load_from_json,
    set_seed, format_time, get_label_map, get_model_path_list)
from utils.metric import score
from torch.cuda.amp import autocast
from transformers import get_scheduler, AdamW
from utils.training_utils import FGM, PGD, SWA, EMA
from transformers import AutoConfig, AutoModel, AutoTokenizer

logger = logging.getLogger("main")


class BaseModel(nn.Module):
    def __init__(self, model_name_or_path, from_scratch=False):
        super(BaseModel, self).__init__()
        if from_scratch:
            self.pretrained_model = AutoModel.from_pretrained(model_name_or_path, output_hidden_states=True)
        else:
            config = AutoConfig.from_pretrained(model_name_or_path, output_hidden_states=True)
            self.pretrained_model = AutoModel.from_config(config)

        self.from_scratch = from_scratch
        self.model_name_or_path = model_name_or_path

    def init_weights(self):
        if not self.from_scratch:
            self.load_state_dict(torch.load(os.path.join(
                self.model_name_or_path, "pytorch_model.bin"), map_location='cpu'))

    @property
    def hidden_size(self):
        return self.pretrained_model.config.hidden_size

    def get_embeddings(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            inputs_embeds=None):
        outputs = self.pretrained_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )
        return outputs


class Trainer:
    task_param_names = ["cls"]  # 这部分参数学习率单独设置，一般高点
    label_names = []  # 模型计算loss需要label 参数名
    trigger_label_file = "trigger.json"
    argument_label_file = "argument.json"
    entity_label_file = "entity.json"
    event_role_file = 'event_role.json'
    data_save_dir = "data"
    trigger_use_bio = False
    argument_use_bio = False
    entity_use_bio = False
    trigger_default_none = 'O'
    argument_default_none = None
    entity_default_none = None
    logit_indexs = [1]
    thresholds = {}

    def __init__(self, args, from_scratch=False):

        set_seed(args.seed)
        self.args = args
        self.n_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args.train_batch_size = args.per_gpu_train_batch_size * max(1, self.n_gpu)
        self.args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, self.n_gpu)
        self.data_dir = args.data_dir if from_scratch else os.path.join(args.output_dir, self.data_save_dir)
        self.model_dir = self.args.model_dir if from_scratch else self.args.output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)

        if self.trigger_label_file is not None:
            self.trigger2id, self.id2trigger = get_label_map(
                os.path.join(self.data_dir, self.trigger_label_file),
                use_bio=self.trigger_use_bio, default_none=self.trigger_default_none)
        if self.argument_label_file is not None:
            self.argument2id, self.id2argument = get_label_map(
                os.path.join(self.data_dir, self.argument_label_file),
                use_bio=self.argument_use_bio, default_none=self.argument_default_none)
        # if self.entity_label_file is not None:
        #     self.entity2id, self.id2entity = get_label_map(
        #         os.path.join(self.data_dir, self.entity_label_file),
        #         use_bio=self.entity_use_bio, default_none=self.entity_default_none
        #     )

        # self.fp16 = bool(self.args.fp16)
        # self.scaler = None
        # if self.fp16:
        #     self.scaler = torch.cuda.amp.GradScaler()

        self.use_amp = False
        self.scaler, self.amp_dtype = None, None
        if self.args.mixed_precision == 'fp16' or self.args.mixed_precision == 'bf16':
            self.amp_dtype = torch.float16 if self.args.mixed_precision == 'fp16' else torch.bfloat16
            self.scaler = torch.cuda.amp.GradScaler()
            self.use_amp = True

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.collate_fn = None
        self.adv_fn = None  # 对抗
        self.avg_model = None  # ema or swa

    def autocast_smart_context_manager(self):
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation.
        """
        if self.use_amp:
            if version.parse(torch.__version__) >= version.parse("1.10"):
                ctx_manager = autocast(dtype=self.amp_dtype)
            else:
                ctx_manager = autocast()
        else:
            ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()

        return ctx_manager

    def add_wandb_config(self):
        wandb.config.update(self.args)

    def get_data_file(self, filename):
        filename = os.path.join(self.args.data_dir, filename)
        return filename

    def get_optimizer_params(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        if len(self.task_param_names) == 0:
            optimizer_params = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
        else:
            optimizer_params = [
                {'params': [p for n, p in param_optimizer if
                            all(nd not in n for nd in no_decay) and all([tn not in n for tn in self.task_param_names])],
                 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in param_optimizer if
                            any(nd in n for nd in no_decay) and all([tn not in n for tn in self.task_param_names])],
                 'weight_decay': 0.0},
                {'params': [p for n, p in param_optimizer if
                            all(nd not in n for nd in no_decay) and any([tn in n for tn in self.task_param_names])],
                 'weight_decay': self.args.weight_decay, 'lr': self.args.task_learning_rate},
                {'params': [p for n, p in param_optimizer if
                            any(nd in n for nd in no_decay) and any([tn in n for tn in self.task_param_names])],
                 'weight_decay': 0.0, 'lr': self.args.task_learning_rate},
            ]
            optimizer_params = [params for params in optimizer_params if len(params['params']) > 0]

        return optimizer_params

    def create_schedule(self, t_total, optimizer=None):
        if self.scheduler is None and self.args.lr_scheduler_type is not None:
            self.scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=int(self.args.warmup_ratio * t_total),
                num_training_steps=t_total,
            )

    def create_optimizer(self):
        optimizer_grouped_parameters = self.get_optimizer_params()
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.args.learning_rate,
            eps=self.args.adam_epsilon, no_deprecation_warning=True)

    def create_optimizer_and_schedule(self, t_total):
        self.create_optimizer()
        self.create_schedule(t_total)

    def set_device(self):
        if self.model is not None:
            self.model.to(self.device)
        if self.n_gpu > 1:
            if self.model is not None:
                self.model = torch.nn.DataParallel(self.model)

    def prepare_inputs(self, data):
        inputs = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        return inputs

    def _save(self, output_dir):
        output_dir = output_dir or self.args.output_dir
        check_dir(output_dir)
        assert self.model is not None
        logger.info(f'  Saving model to {output_dir}...')

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        model_to_save.pretrained_model.config.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        args_to_save = copy.deepcopy(self.args.__dict__)
        data_save_dir = os.path.join(output_dir, self.data_save_dir)
        check_dir(data_save_dir)
        write_to_json(args_to_save, os.path.join(data_save_dir, "args.json"))
        should_save_files = [self.trigger_label_file, self.entity_label_file,
                             self.argument_label_file, self.event_role_file]
        for filename in should_save_files:
            if os.path.exists(os.path.join(self.data_dir, filename)):
                if filename is not None:
                    shutil.copyfile(
                        os.path.join(self.data_dir, filename),
                        os.path.join(data_save_dir, filename)
                    )

        return output_dir

    def save(self, output_dir=None):
        self._save(output_dir)

    def limit_checkpoint(self, limit, output_dir=None):
        output_dir = output_dir or self.args.output_dir
        model_list = get_model_path_list(output_dir)
        while len(model_list) > limit:
            shutil.rmtree(model_list[0])
            model_list = model_list[1:]

    def clip_grad_norm(self, max_grad_norm=1.0):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

    def _optimizer_step(self, max_grad_norm):

        if self.scaler is not None:
            # AMP: gradients need unscaling
            self.scaler.unscale_(self.optimizer)
        if isinstance(max_grad_norm, float):
            self.clip_grad_norm(max_grad_norm)
        optimizer_was_run = True
        if self.scaler is not None:
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler.get_scale()
            optimizer_was_run = scale_before <= scale_after
        else:
            self.optimizer.step()
        if optimizer_was_run and self.scheduler is not None:
            self.scheduler.step()
        self.model.zero_grad()

    def optimizer_step(self, max_grad_norm=1.0):
        self._optimizer_step(max_grad_norm)

    def compute_loss(self, inputs, backward=True, return_outputs=False):

        with self.autocast_smart_context_manager():
            outputs = self.model(**inputs)

        if isinstance(outputs, tuple):
            loss = outputs[0]
        elif isinstance(outputs, dict):
            loss = outputs['loss']
        else:
            loss = outputs

        if backward:
            loss = self.backward(loss)

        return loss if not return_outputs else (loss, outputs)

    def backward(self, loss):
        if self.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        return loss.detach()

    def train_step(self, batch_data):
        self.model.train()
        inputs = self.prepare_inputs(batch_data)
        if self.adv_fn is not None:
            if isinstance(self.adv_fn, FGM):
                loss = self.compute_loss(inputs, backward=True)
                self.adv_fn.attack()
                loss_adv = self.compute_loss(inputs, backward=True)
                self.adv_fn.restore()
            elif isinstance(self.adv_fn, PGD):
                loss = self.compute_loss(inputs, backward=True)
                self.adv_fn.backup_grad()
                for t in range(self.adv_fn.adv_steps):
                    self.adv_fn.attack(is_first_attack=(t == 0))
                    if t != self.adv_fn.adv_steps - 1:
                        self.model.zero_grad()
                    else:
                        self.adv_fn.restore_grad()
                    loss_adv = self.compute_loss(
                        inputs, backward=True
                    )
                self.adv_fn.restore()

            else:
                raise ValueError("adv type error")
        else:
            loss = self.compute_loss(inputs, backward=True)
        return loss

    def predict_step(self, batch_data):
        self.model.eval()
        inputs = self.prepare_inputs(batch_data)

        with torch.no_grad():
            for label in self.label_names:
                inputs[label] = None
            with self.autocast_smart_context_manager():
                outputs = self.model(**inputs)

        return outputs

    def create_dataset(self, filename, training=False):
        raise NotImplementedError

    def create_dataloader(self, dataset, training=False):
        if training:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.args.train_batch_size if training else self.args.eval_batch_size,
            collate_fn=self.collate_fn,
            drop_last=training,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        return dataloader

    def _predict(self, dataset):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, dataset):
        raise NotImplementedError

    def train(self, train_dataset, eval_dataset=None, test_dataset=None):
        set_seed(self.args.seed)

        # 对抗训练
        if self.args.adv_type == 'fgm':
            self.adv_fn = FGM(self.model, epsilon=1., param_name='word_embeddings')
        elif self.args.adv_type == 'pgd':
            self.adv_fn = PGD(self.model, adv_steps=3, epsilon=1., alpha=0.3, param_name='word_embeddings')

        if self.args.use_wandb:
            import wandb
            wandb.init(project=self.args.project, entity="huanghui_gz",
                       name=self.args.run_name)
            self.add_wandb_config()
            wandb.watch(self.model, log=None)  # 此处可修改 "gradients", "parameters", "all", or None

        train_dataloader = self.create_dataloader(train_dataset, training=True)
        num_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        t_total = num_steps_per_epoch * self.args.num_train_epochs
        self.create_optimizer_and_schedule(t_total)

        best_score = 0.
        best_step = -1
        tr_loss = 0
        tr_examples = 0
        global_step = 0
        contain_best_model = False
        eval_step = num_steps_per_epoch // self.args.eval_per_epoch
        if self.args.save_per_epoch > 0:
            save_step = num_steps_per_epoch // self.args.save_per_epoch
        else:
            save_step = -1

        if self.args.swa_fn == "ema":
            self.avg_model = EMA(self.model, (self.args.ema_decay_1, self.args.ema_decay_2))
            self.avg_model.register()  # ema一开始就注册

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Instances batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel & accumulation) = %d",
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
        )
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        train_iter = tqdm(total=t_total, desc='training')
        set_seed(self.args.seed)
        report = []
        for epoch in range(self.args.num_train_epochs):
            if self.avg_model is not None and self.args.swa_start_epoch <= epoch:
                if self.args.swa_fn == 'ema' and not self.avg_model.changed:
                    self.avg_model.changed = True  # 修改ema的decay值

            for step, train_batch in enumerate(train_dataloader):
                self.model.train()
                loss = self.train_step(train_batch)

                tr_loss += loss.item()
                tr_examples += len(train_batch['input_ids'])

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer_step(max_grad_norm=self.args.max_grad_norm)
                    # 同步更新shadow参数
                    if self.args.swa_fn == 'ema' and self.avg_model is not None and self.avg_model.registered:
                        self.avg_model.update()

                    global_step += 1
                    train_iter.update(1)
                    loss_item = tr_loss / tr_examples
                    train_iter.set_postfix(loss=loss_item)
                    if self.args.use_wandb:
                        wandb.log({"loss": loss_item})
                    tr_loss = 0
                    tr_examples = 0

                    if global_step % eval_step == 0:
                        if self.avg_model is not None and self.avg_model.updated:
                            self.avg_model.apply_shadow()  # 将shadow参数复制到当前模型
                        eval_score = -1
                        logger.info("***** Running Evaluation *****")
                        logger.info("  Current Epoch = %d" % epoch)
                        logger.info("  Current Step = %d" % global_step)
                        report.append({'global_step': global_step})
                        if eval_dataset is not None:
                            eval_res = self.evaluate(eval_dataset)
                            argument_score = eval_res['argument-cls-f1']
                            trigger_score = eval_res['trigger-cls-f1']
                            report[-1]["eval_trigger_f1"] = trigger_score
                            report[-1]["eval_argument_f1"] = argument_score
                            if self.args.use_wandb:
                                wandb.log({"eval_trigger_f1": trigger_score, "eval_argument_f1": argument_score})
                            logger.info("  Eval Trigger F1 = %.2f" % trigger_score)
                            logger.info("  Eval Argument F1 = %.2f" % argument_score)
                            eval_score = (trigger_score + argument_score) / 2
                            logger.info("  Avg F1 = %.2f" % eval_score)

                        if test_dataset is not None:
                            eval_res = self.evaluate(test_dataset, prefix='test')
                            argument_score = eval_res['argument-cls-f1']
                            trigger_score = eval_res['trigger-cls-f1']
                            report[-1]["test_trigger_f1"] = trigger_score
                            report[-1]["test_argument_f1"] = argument_score
                            if self.args.use_wandb:
                                wandb.log({"test_trigger_f1": trigger_score, "test_argument_f1": argument_score})

                            logger.info("  Test Trigger F1 = %.2f" % trigger_score)
                            logger.info("  Test Argument F1 = %.2f" % argument_score)
                            eval_score = (trigger_score + argument_score) / 2
                            logger.info("  Avg F1 = %.2f" % eval_score)

                        if eval_score > best_score:
                            best_score = eval_score
                            best_step = global_step
                            should_save = True
                        else:
                            should_save = False
                        logger.info("  Best F1 = %.2f" % best_score)
                        if should_save:
                            self.save(os.path.join(self.args.output_dir, "checkpoint-best"))
                            contain_best_model = True

                        if self.avg_model is not None and self.avg_model.updated:
                            self.avg_model.restore()  # 恢复参数继续训练

                    if isinstance(save_step, int) and save_step > 0 and global_step % save_step == 0:
                        if self.avg_model is not None and self.avg_model.updated:
                            self.avg_model.apply_shadow()
                        self.save(os.path.join(self.args.output_dir, f"checkpoint-{global_step}"))

                        if self.avg_model is not None and self.avg_model.updated:
                            self.avg_model.restore()

        report.append({"best_step": best_step, "best_score": best_score})
        write_to_json(report, os.path.join(self.args.output_dir, 'report.json'))
        if self.avg_model is not None and self.avg_model.updated:
            self.avg_model.apply_shadow()
        if contain_best_model:
            self.model.load_state_dict(torch.load(
                os.path.join(self.args.output_dir, "checkpoint-best", "pytorch_model.bin"),
                map_location='cpu'
            ))

        self.save(self.args.output_dir)
        if self.args.use_wandb:
            wandb.finish()
        return report

    def evaluate(self, dataset, prefix='eval', save_result=False, log=False):
        if log:
            logger.info(f'Evaluating {prefix} dataset...')
        c_time = time.time()
        golds, predictions = self.predict(dataset)
        eval_result = score(predictions, golds)
        if log:
            logger.info(f'{eval_result}')
            logger.info(f'Evaluation Used time: {format_time(time.time() - c_time)}')

        if save_result:
            save_dir = os.path.join(self.args.output_dir, prefix)
            check_dir(save_dir)
            write_to_json(eval_result, os.path.join(save_dir, f"{prefix}_results.json"))
            with open(os.path.join(save_dir, "predictions.json"), 'w', encoding='utf-8') as w:
                for prediction in predictions:
                    line = {'doc_id': prediction.doc_id, 'sent_id': prediction.sent_id, 'events': []}
                    for event in prediction.events:
                        event_dict = {
                            'trigger': {'start': int(event.trigger.start),
                                        'end': int(event.trigger.end),
                                        'label': event.trigger.label},
                            'arguments': [
                                {'start': int(arg.start), 'end': int(arg.end), 'label': arg.label} for arg in
                                event.arguments]
                        }
                        line['events'].append(event_dict)
                    w.write(json.dumps(line, ensure_ascii=False) + '\n')
        return eval_result

    @classmethod
    def detach_tensor(cls, tensor: Union[Tuple[torch.Tensor], List[torch.Tensor], torch.Tensor]):
        if isinstance(tensor, torch.Tensor):
            t = tensor.cpu()
            if t.dtype == torch.bfloat16:
                t = t.to(torch.float32)
            return t.numpy()
        res = ()
        for t in tensor:
            t = t.cpu()
            if t.dtype == torch.bfloat16:
                t = t.to(torch.float32)
            res += (t.detach().cpu().numpy(),)
        return res

    def swa(self, model_dir=None):
        model_dir = self.model_dir if model_dir is None else model_dir
        model_path_list = get_model_path_list(model_dir)
        assert 0 < self.args.swa_num <= len(model_path_list), \
            f'Using swa, swa start should smaller than {len(model_path_list)} and bigger than 0'
        self.model.to(torch.device("cpu"))
        swa_model = copy.deepcopy(self.model)
        swa_n = 0.

        with torch.no_grad():
            for sub_dir in tqdm(model_path_list[-self.args.swa_num:], desc='swa'):
                model_path = os.path.join(sub_dir, "pytorch_model.bin")
                self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                tmp_para_dict = dict(self.model.named_parameters())

                alpha = 1. / (swa_n + 1.)

                for name, para in swa_model.named_parameters():
                    para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

                swa_n += 1
        self.model.load_state_dict(swa_model.state_dict())
        self.set_device()

    def search_threshold(self, dataset, scope=50, ratio=0.1):
        # 提前预测，加速搜索
        predictions = self._predict(dataset)
        golds = dataset.data()

        def eval_(pred_results, return_all=False):
            eval_res = score(pred_results, golds)
            arg_score = eval_res['argument-cls-f1']
            tri_score = eval_res['trigger-cls-f1']
            # cross_score = eval_res['cross-overlap-f1']
            # single_score = eval_res['single-overlap-f1']
            mean_score = (tri_score + arg_score) / 2
            if return_all:
                return mean_score, eval_res
            return mean_score

        results = self.decode(*predictions)
        ori_score = eval_(results)
        for mode in self.thresholds:
            best_score = 0
            best_threshold = 0
            iteration = trange(0, scope, desc=f'search {mode}')
            for threshold in iteration:
                # ratio = round(1 / scope, 2)
                # cur_threshold = round(threshold * ratio, 2)
                # self.thresholds[mode] = cur_threshold
                # # 根据当前阈值重新解码，评估
                # pre_results = self.decode(*predictions)
                # cur_score = eval_(pre_results)
                # iteration.set_postfix(score=cur_score, best_threshold=best_threshold, best_score=best_score)
                # if best_score < cur_score:
                #     best_score = cur_score
                #     best_threshold = cur_threshold
                for direct in [1, -1]:
                    cur_threshold = threshold * direct * ratio
                    cur_threshold = round(cur_threshold, 2)
                    self.thresholds[mode] = cur_threshold
                    # 根据当前阈值重新解码，评估
                    pre_results = self.decode(*predictions)
                    cur_score = eval_(pre_results)
                    iteration.set_postfix(score=cur_score, best_threshold=best_threshold, best_score=best_score)
                    if best_score < cur_score:
                        best_score = cur_score
                        best_threshold = cur_threshold
            self.thresholds[mode] = best_threshold

        pre_results = self.decode(*predictions)
        final_score, eval_res = eval_(pre_results, return_all=True)
        for k, v in self.thresholds.items():
            logger.info(f"  best {k} = {v}")
        logger.info(f"  origin score = {ori_score}")
        logger.info(f"  search score = {final_score}")
        logger.info(f"  search scope = from {0} to {scope}")
        logger.info(f"  search ratio = {ratio}")
        logger.info(eval_res)
        return self.thresholds
