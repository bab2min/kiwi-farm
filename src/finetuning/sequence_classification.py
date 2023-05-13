from functools import partial
import itertools

import numpy as np

from datasets import load_dataset
import evaluate

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding, 
    TrainingArguments, 
    Trainer
)
import kiwipiepy.transformers_addon

def preprocess_function(args, examples):
    return tokenizer(examples[args.key], truncation=True)

def make_noisy_split(args, examples, method):
    examples = {k:v for k, v in examples.items()}
    if not examples[args.key].strip():
        return examples
    
    if method == "no_space":
        examples[args.key] = examples[args.key].replace(" ", "")
    elif method == "all_space":
        examples[args.key] = " ".join(examples[args.key].replace(" ", ""))
    elif method == "random_space":
        target = examples[args.key].strip()
        target = np.array(list(target))
        spaces = (target == ' ')
        space_pos = np.where(spaces)[0]
        space_pos -= np.arange(len(space_pos))
        target = target[~spaces]
        if len(target) * 0.2 > 1:
            fixed_random = np.random.RandomState(ord(target[0]))
            spaces = np.zeros_like(target, dtype=np.bool_)
            spaces[space_pos] = True
            invert = fixed_random.choice(len(spaces), int(len(spaces) * 0.2), replace=False)
            spaces[invert] = ~spaces[invert]
            target = ''.join(itertools.chain.from_iterable(zip(target, np.where(spaces, ' ', ''))))
            examples[args.key] = target
    else:
        raise ValueError(f"Unknown method: {method}")
    return examples

def compute_metrics(evaluator, eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return evaluator.compute(predictions=predictions, references=labels)

def main(args):
    global tokenizer
    dataset = load_dataset(args.dataset, args.subset)
    if args.test_split is None:
        for c in ["test", "validation"]:
            if c in dataset:
                args.test_split = c
                break
        if args.test_split is None:
            raise ValueError("Cannot find test split.")
        print(f"`args.test_split` is set to {args.test_split!r}")
    
    num_labels = max(dataset["train"]["label"]) + 1

    if args.transform_train:
        dataset["train"] = dataset["train"].map(partial(make_noisy_split, args, method=args.transform_train))
    dataset["test_no_space"] = dataset[args.test_split].map(partial(make_noisy_split, args, method="no_space"))
    dataset["test_all_space"] = dataset[args.test_split].map(partial(make_noisy_split, args, method="all_space"))
    dataset["test_random_space"] = dataset[args.test_split].map(partial(make_noisy_split, args, method="random_space"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    tokenized_dataset = dataset.map(partial(preprocess_function, args), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    evaluator = evaluate.load(args.metric)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        tf32=args.tf32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset[args.test_split],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, evaluator),
    )

    trainer.train()
    print("==== Test Set ====")
    print(trainer.evaluate(tokenized_dataset[args.test_split]))
    print("==== Test Set (No Space) ====")
    print(trainer.evaluate(tokenized_dataset["test_no_space"]))
    print("==== Test Set (All Space) ====")
    print(trainer.evaluate(tokenized_dataset["test_all_space"]))
    print("==== Test Set (Random Space) ====")
    print(trainer.evaluate(tokenized_dataset["test_random_space"]))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default='kiwi-farm/roberta-base-32k')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset', default='nsmc')
    parser.add_argument('--subset')
    parser.add_argument('--key', default='document')
    parser.add_argument('--test_split')
    parser.add_argument('--metric', default='accuracy')
    parser.add_argument('--transform_train', choices=['no_space', 'all_space', 'random_space'])
    parser.add_argument('--num_train_epochs', default=2., type=float)
    
    parser.add_argument('--tf32', default=False, action='store_true')
    parser.add_argument('--warmup_ratio', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--per_device_train_batch_size', default=64, type=int)
    parser.add_argument('--per_device_eval_batch_size', default=64, type=int)
    main(parser.parse_args())

