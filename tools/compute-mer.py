#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import re
from itertools import chain

import jieba
import opencc
import editdistance

CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                   "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                   "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ"]

chars_to_ignore_re = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"
def remove_special_characters(text):
    if chars_to_ignore_re is not None:
        return re.sub(chars_to_ignore_re, "", text).lower()
    else:
        return text.lower()

def tokenize_for_mer(text):
    tokens = list(filter(lambda tok: len(tok.strip()) > 0, jieba.lcut(text)))
    tokens = [[tok] if tok.isascii() else list(tok) for tok in tokens]
    return list(chain(*tokens))

def tokenize_for_cer(text):
    tokens = list(filter(lambda tok: len(tok.strip()) > 0, list(text)))
    return tokens

def load_file(filepath):
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2: data[parts[0]] = parts[1]
            elif len(parts) == 1: data[parts[0]] = ""
    return data

def main():
    parser = argparse.ArgumentParser(description="compute mixture error rate (MER)")
    parser.add_argument('ref', help="reference text file")
    parser.add_argument('pred', help="prediction text file")
    args = parser.parse_args()

    refs = load_file(args.ref) 
    preds = load_file(args.pred)

    mixed_distance, mixed_tokens = 0, 0
    char_distance, char_tokens = 0, 0

    converter = opencc.OpenCC('t2s.json')

    for utt_id  in refs:        
        pred = remove_special_characters(converter.convert(preds[utt_id]))
        ref = remove_special_characters(refs[utt_id])

        m_pred = tokenize_for_mer(pred)
        m_ref = tokenize_for_mer(ref)
        mixed_distance += editdistance.distance(m_pred, m_ref)
        mixed_tokens += len(m_ref)

        c_pred = tokenize_for_cer(pred)
        c_ref = tokenize_for_cer(ref)
        char_distance += editdistance.distance(c_pred, c_ref)
        char_tokens += len(c_ref)

    mer = mixed_distance / mixed_tokens
    cer = char_distance / char_tokens
    print(f"MER: {mer:.4f}, CER: {cer:.4f}")

if __name__ == "__main__":
    main()