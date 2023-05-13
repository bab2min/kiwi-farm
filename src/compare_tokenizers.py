import sys
import re
try:
    import readline
except:
    pass

from transformers import AutoTokenizer, PreTrainedTokenizer
import kiwipiepy.transformers_addon

def vocab_stat(tokenizer:PreTrainedTokenizer):
    vocab = tokenizer.vocab
    h = r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002A6DF\U0002A700–\U0002EBEF\U0002F800-\U0002FA1F\U00030000–\U000323AF]"
    num_hangul_tokens = sum(1 for k in vocab if re.search(r'^(##|~~?)?[가-힣]', k))
    num_hangul_words = sum(1 for k in vocab if re.search(r'^[가-힣]', k))
    num_hangul_subwords = sum(1 for k in vocab if re.search(r'^(##|~~?)[가-힣]', k))
    num_alnum_tokens = sum(1 for k in vocab if re.search(r'^(##)?[A-Za-z0-9]', k))
    num_alnum_words = sum(1 for k in vocab if re.search(r'^[A-Za-z0-9]', k))
    num_alnum_subwords = sum(1 for k in vocab if re.search(r'^##[A-Za-z0-9]', k))
    num_hanja_tokens = sum(1 for k in vocab if re.search(r'^(##)?' + h, k))
    num_hanja_words = sum(1 for k in vocab if re.search(r'^' + h, k))
    num_hanja_subwords = sum(1 for k in vocab if re.search(r'^##' + h, k))

    return dict(
        vocab_size=len(vocab),
        num_special_tokens=len(tokenizer.all_special_ids),
        num_hangul_tokens=num_hangul_tokens,
        num_hangul_words=num_hangul_words,
        num_hangul_subwords=num_hangul_subwords,
        num_alnum_tokens=num_alnum_tokens,
        num_alnum_words=num_alnum_words,
        num_alnum_subwords=num_alnum_subwords,
        num_hanja_tokens=num_hanja_tokens,
        num_hanja_words=num_hanja_words,
        num_hanja_subwords=num_hanja_subwords,
    )

def main(args):
    tokenizers = []
    for name in args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizers.append((name, tokenizer))
    
    if args.print_stat:
        for name, tokenizer in tokenizers:
            print(name, vocab_stat(tokenizer))

    while 1:
        line = input()
        for name, tokenizer in tokenizers:
            r = tokenizer.tokenize(line)
            print(name, r, sep='\t')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('tokenizer', nargs='+')
    parser.add_argument('--print_stat', default=False, action='store_true')
    main(parser.parse_args())
