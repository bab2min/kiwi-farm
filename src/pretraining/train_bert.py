import re
import time
import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, RandomSampler, DataLoader


from transformers import RobertaConfig, RobertaModel, RobertaForMaskedLM, BertTokenizer
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from kiwipiepy.sw_tokenizer import SwTokenizer

class MaskLMDataset(Dataset):
    def __init__(self, path, tokenizer, mlm_prob=0.15, max_seq_length=512):
        super().__init__()

        self._input_ids = np.load(path + '/input_ids.npy', mmap_mode='r')
        self._input_ptrs = np.load(path + '/input_ptrs.npy', mmap_mode='r')

        self._vocab_size = len(tokenizer)
        self._is_subtoken = torch.zeros([len(tokenizer)], dtype=torch.bool)
        self._valid_mlm_token = torch.zeros([len(tokenizer)], dtype=torch.bool)
        
        for t, i in tokenizer.vocab.items():
            if t.startswith('##'): self._is_subtoken[i] = 1
            else: self._valid_mlm_token[i] = 1
        
        for i in tokenizer.all_special_ids: self._valid_mlm_token[i] = 0
        self._valid_mlm_token[tokenizer.unk_token_id] = 1

        self._cls_token = np.array([tokenizer.cls_token_id])
        self._sep_token = np.array([tokenizer.sep_token_id])
        self._mask_token_id = tokenizer.mask_token_id
        self._pad_token_id = tokenizer.pad_token_id

        self._mlm_prob = mlm_prob
        self._max_seq_length = max_seq_length
        self._stride = max_seq_length // 2
        self._num_chunks = len(self._input_ids) // self._stride
        self._random = np.random.RandomState(42)

    def __len__(self):
        return self._num_chunks

    def __getitem__(self, idx):
        start = idx * self._stride
        chunk_begin, chunk_end = np.searchsorted(self._input_ptrs, [start, start + self._max_seq_length], 'right') - 1
        chunk_end = self._random.randint(chunk_begin, chunk_end + 1) + 1

        p = self._input_ptrs[chunk_begin:chunk_end + 1].copy()
        p[0] = start

        chunks = [self._cls_token]
        for s, e in zip(p, p[1:]):
            chunks.append(self._input_ids[s:e])
            chunks.append(self._sep_token)
        input_ids = np.concatenate(chunks)
        input_ids = input_ids[:self._max_seq_length]

        num_valid_tokens = len(input_ids)
        if num_valid_tokens < self._max_seq_length:
            input_ids = np.pad(input_ids, (0, self._max_seq_length - num_valid_tokens), constant_values=self._pad_token_id)

        input_ids = torch.from_numpy(input_ids.astype(np.int64))

        probs = torch.full(input_ids.shape, self._mlm_prob)
        probs.masked_fill_(~self._valid_mlm_token[input_ids], 0.)
        masked_indices = torch.bernoulli(probs).bool()
        for p in zip(*torch.where(masked_indices.clone())):
            try:
                e = torch.where(~self._is_subtoken[input_ids[(*p[:-1], slice(p[-1] + 1, None))]])[0][0] + p[-1] + 1
            except:
                e = -1
            masked_indices[(*p[:-1], slice(p[-1], e))] = 1

        labels = input_ids.clone()
        labels[~masked_indices] = -100
        
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self._mask_token_id

        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self._vocab_size, input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        attention_mask = torch.zeros_like(input_ids)
        attention_mask[torch.arange(len(input_ids)) < num_valid_tokens] = 1

        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

def map_old_to_new_tokenizer(old, new):
    old_ids, new_ids = [], []
    old_vocab = old.vocab
    new_vocab = new.vocab

    for form, idx in new_vocab.items():
        form = form.rsplit('/', 1)[0]
        try: 
            old_idx = old_vocab[form]
        except KeyError: 
            try: old_idx = old_vocab['##' + form]
            except KeyError: 
                continue
        old_ids.append(old_idx)
        new_ids.append(idx)
    return old_ids, new_ids

def update_token_embeddings(model, old_ids, new_ids, new_tokenizer):
    old_emb = model.get_input_embeddings()
    old_weights = old_emb.weight.data[old_ids].clone()
    model.resize_token_embeddings(len(new_tokenizer))
    new_emb = model.get_input_embeddings()
    new_emb.padding_idx = new_tokenizer.pad_token_id
    model._init_weights(new_emb)
    new_emb.weight.data[new_ids] = old_weights
    new_emb.weight.data[new_tokenizer.pad_token_id].zero_()

    model.config.pad_token_id = new_tokenizer.pad_token_id
    model.config.bos_token_id = new_tokenizer.bos_token_id
    model.config.eos_token_id = new_tokenizer.eos_token_id

def load_datasets(pathes, tokenizer, mlm_prob, max_seq_length):
    datasets = []
    for path in pathes:
        m = re.search(r'\*([0-9]+)', path)
        if m:
            path = path[:m.span()[0]]
            multiplier = int(m.group(1))
        else:
            multiplier = 1
        d = MaskLMDataset(path, tokenizer, mlm_prob=mlm_prob, max_seq_length=max_seq_length)
        for _ in range(multiplier): datasets.append(d)
    return ConcatDataset(datasets)

def get_optimizer_paramters(model, weight_decay=0):
    no_decay = ["bias", "LayerNorm.weight"]
    inner_model = getattr(model, 'module', model)
    wd, nwd = [], []
    for n, p in inner_model.named_parameters():
        if not p.requires_grad: continue
        if any(nd in n for nd in no_decay): nwd.append(p)
        else: wd.append(p)
    return [
        {
            "params": wd,
            "weight_decay": weight_decay,
        },
        {
            "params": nwd, 
            "weight_decay": 0.0,
        },
    ]

def inf_generator(dataloader, sampler=None):
    import itertools
    for epoch in itertools.count():
        try: sampler.set_epoch(epoch)
        except: pass
        yield from iter(dataloader)

def train_model(args, model, tokenizer, gpu=None):
    train_dataset = load_datasets(args.train_data, tokenizer, mlm_prob=args.mlm_prob, max_seq_length=args.max_seq_length)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.dataloader_num_workers)
    
    total_train_steps = int(len(train_dataloader) * args.num_train_epochs)
    opt_params = get_optimizer_paramters(model, args.weight_decay)

    optimizer = AdamW(opt_params, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
        num_warmup_steps=int(total_train_steps * args.warmup) // (args.grad_acc or 1), 
        num_training_steps=total_train_steps // (args.grad_acc or 1)
    )

    gen = inf_generator(train_dataloader, train_sampler)
    mlm_loss = 0

    model.train()

    t_begin = time.perf_counter()
    for global_step, batch in zip(range(total_train_steps), gen):
        inputs = {k:v.to(model.device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss / (args.grad_acc or 1)
        loss.backward()
        mlm_loss += loss.item()

        if not args.grad_acc or (global_step + 1) % args.grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            real_step = (global_step + 1) // (args.grad_acc or 1)

            if not gpu and args.log_interval and real_step % args.log_interval == 0:
                elapsed = int(time.perf_counter() - t_begin)
                text = f"Step {real_step:05} ({real_step / (total_train_steps / (args.grad_acc or 1)):.2%})  Elapsed {datetime.timedelta(seconds=elapsed):}"
                text += (f"  Mlm Loss {mlm_loss / args.log_interval:.4}")
                print(text, flush=True)
                mlm_loss = 0
            
            if not gpu and args.save_interval and real_step % args.save_interval == 0:
                model.eval()
                model_to_save = getattr(model, 'module', model)
                model_to_save.save_pretrained(args.output_dir)
                #tokenizer.save_pretrained(args.output_dir, False)
                model.train()
    model.eval()
    model_to_save = getattr(model, 'module', model)
    model_to_save.save_pretrained(args.output_dir)

def main(args):
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except:
        pass
    
    model = RobertaForMaskedLM.from_pretrained(args.model_name_or_path).to(args.device)
    if args.new_tokenizer:
        old_tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        new_tokenizer = SwTokenizer(args.new_tokenizer)
        old_ids, new_ids = map_old_to_new_tokenizer(old_tokenizer, new_tokenizer)
        update_token_embeddings(model, old_ids, new_ids, new_tokenizer)
    else:
        new_tokenizer = SwTokenizer(args.model_name_or_path + '/tokenizer.json')
    
    train_model(args, model, new_tokenizer)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default='klue/roberta-base')
    parser.add_argument('--output_dir')
    parser.add_argument('--new_tokenizer')
    parser.add_argument('--train_data', nargs='*')
    parser.add_argument('--mlm_prob', default=0.15, type=float)
    parser.add_argument('--max_seq_length', default=512, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--save_interval', default=1000, type=int)
    parser.add_argument('--dataloader_num_workers', default=0, type=int)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--grad_acc', default=0, type=int)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--warmup", default=0.1, type=float)
    parser.add_argument("--learning_rate", default=5e-5, type=float)

    main(parser.parse_args())
