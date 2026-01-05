import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, GPT2Tokenizer, Adafactor, T5Config, AutoTokenizer, BertTokenizer
from tqdm import tqdm
import random
from aksharamukha import transliterate
import multiprocessing
from functools import partial
import os
import csv
from collections import Counter
import numpy as np
from torch.amp import autocast, GradScaler

# Basic CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

def load_dks_data(filename):
    print(f'Loading G2P dataset from {filename}...')
    df = pd.read_csv(filename, header=None, keep_default_na=False, delimiter="\t")
    # print(df)

    orig = df[0].astype(str).tolist()
    trans = df[1].apply(lambda x: x.replace(" . ", " ")).astype(str).tolist()
    
    print(f"Loaded {len(orig)} orig-trans pairs")
    return orig, trans

class DKSDataset(torch.utils.data.Dataset):
    def __init__(self, orig, trans, tokenizer, max_input_len=128, max_output_len=128, cache_tokenization=True):
        self.orig = orig
        self.trans = trans
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.cache_tokenization = cache_tokenization
        self.cache = {}  # Add caching
    
    def __len__(self):
        return len(self.orig)

    def __getitem__(self, idx):
        if self.cache_tokenization and idx in self.cache:
            return self.cache[idx]
            
        # Format input
        input_text = f"O-{self.orig[idx]} T:"
        target_text = f"{self.trans[idx]} "

        inputs = self.tokenizer(input_text, max_length=self.max_input_len, padding="max_length", truncation=True, return_tensors="pt")
        targets = self.tokenizer(target_text, max_length=self.max_output_len, padding="max_length", truncation=True, return_tensors="pt")

        item = {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": targets.input_ids.squeeze(),
            "orig": self.orig[idx],
            "trans": target_text
        }
        
        if self.cache_tokenization:
            self.cache[idx] = item
            
        return item

def custom_collate_fn(batch):
    """Custom collate function to handle transliteration tuples properly"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    orig = [item["orig"] for item in batch]
    trans = [item["trans"] for item in batch]  # list of lists stays intact

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "orig": orig,
        "trans": trans,
    }


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def is_partial_match(s1, s2):
    s1, s2 = s1.replace('।','').replace('॥',''), s2.replace('।','').replace('॥','')
    return s1 in s2 or s2 in s1


def edit_distance(x, y) -> int:
    #  From https://github.com/sigmorphon/2020/blob/master/task1/evaluation/evallib.py
    idim = len(x) + 1
    jdim = len(y) + 1
    table = np.zeros((idim, jdim), dtype=np.uint8)
    table[1:, 0] = 1
    table[0, 1:] = 1
    for i in range(1, idim):
        for j in range(1, jdim):
            if x[i - 1] == y[j - 1]:
                table[i][j] = table[i - 1][j - 1]
            else:
                c1 = table[i - 1][j]
                c2 = table[i][j - 1]
                c3 = table[i - 1][j - 1]
                table[i][j] = min(c1, c2, c3) + 1
    return int(table[-1][-1])

def test_model(model, tokenizer, test_file, device, batch_size=64, output_csv="./results/results_dks/results.csv", file_loading_function=load_dks_data):
    # Load test data
    test_orig, test_trans = file_loading_function(test_file)

    # Prepare test dataset
    test_dataset = DKSDataset(test_orig, test_trans, tokenizer)

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=5, 
        pin_memory=True, 
        prefetch_factor=10, 
        persistent_workers=True, 
        collate_fn=custom_collate_fn
    )
    
    exact_matches = 0
    total_partial_match = 0
    total_predictions = 0
    total_cer = 0.0   # store cumulative CER
    total_edit_distance = 0
    total_truth_chars = 0


    torch.cuda.empty_cache()
    model.eval()

    with torch.no_grad(), open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['input','real_answer', 'pred_answer', 'exact_match', 'partial_match', 'cer'])
        
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            batch_answers = batch['trans']

            outputs = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                max_length=128,
                use_cache=True,  
                synced_gpus=False
            )

            predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                
            # Process each prediction in the batch
            for idx in range(len(predictions)):
                pred = predictions[idx].strip()
                truth = batch_answers[idx].strip()

                pred = pred.replace(" . "," ")
                truth = truth.replace(" . "," ")

                # Exact Match
                exact_match = pred == truth
                if exact_match:
                    exact_matches += 1

                # Partial Match
                partial_match = is_partial_match(pred, truth)
                if partial_match:
                    total_partial_match += 1

                # Character Error Rate (CER)
                total_edit_distance += edit_distance(pred, truth)
                total_truth_chars += len(truth)

                # Increment total predictions
                total_predictions += 1

                # Write to CSV
                csv_writer.writerow([
                    tokenizer.decode(input_ids[idx], skip_special_tokens=True),
                    truth,
                    pred,
                    int(exact_match),
                    int(partial_match),
                    f"{edit_distance(pred, truth):.4f}"
                ])

    # Calculate final metrics
    exact_match_accuracy = exact_matches / total_predictions
    average_partial_match = total_partial_match / total_predictions
    average_cer = total_edit_distance / total_truth_chars if total_truth_chars > 0 else 0.0

    print(f"\nFinal Metrics:")
    print(f"Total predictions processed: {total_predictions}")
    # print(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")
    # print(f"Partial Match Accuracy: {average_partial_match:.4f}")
    print(f"CER: {(average_cer*100):.4f}%")

    return {
        "exact_match": exact_match_accuracy,
        "partial_match": average_partial_match,
        "cer": average_cer
    }



import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--tok_name", type=str, default="gc", help="Tokenizer name", choices=["gc", "gbpe", "wp", "sp", "itrans", "bpe"])
parser.add_argument("--lang", type=str, default="hin", help="Tokenizer name", choices=["hin", "mr"])
parser.add_argument("--eval_all", type=bool, default=False, help="Evaluate all types")
args = parser.parse_args()

tok_names = [args.tok_name] if not args.eval_all else ["gc", "gbpe", "wp", "sp", "bpe"]

for tok_name in tok_names:

    if(tok_name == 'gc'):
        from training_tokenizers.gc_tokenizer_hf import DevanagariTokenizer, load_vocabulary
        # tokenizer = DevanagariTokenizer(vocab_dict=load_vocabulary(vocab_file_path="vocab_gc.json"))
        tokenizer = DevanagariTokenizer(vocab_dict=load_vocabulary(vocab_file_path="./all_tokenizer_files/gc_tokenizer_files_ehm/vocab.json"))
    elif(tok_name == 'gbpe'):
        from training_tokenizers.gbpe_tokenizer_hf import *
        tokenizer = GBPETokenizer(vocab_file="./all_tokenizer_files/gbpe_tokenizer_files_ehm/vocab.json", merges_file="./all_tokenizer_files/gbpe_tokenizer_files_ehm/merges.json")
    elif(tok_name=='bpe'):
        from training_tokenizers.bpe_tokenizer_hf import *
        tokenizer = BPETokenizer(vocab_file="./all_tokenizer_files/bpe_tokenizer_files_ehm/vocab.json", merges_file="./all_tokenizer_files/bpe_tokenizer_files_ehm/merges.json")
    elif(tok_name == 'wp'):
        # Initialize BERT (wp) tokenizer 
        # tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        from training_tokenizers.wp_tok import WordPieceTokenizer
        tokenizer = WordPieceTokenizer(tokenizer_path="./all_tokenizer_files/wordpiece_tokenizer_files_ehm/wordpiece.json")
    elif(tok_name=='sp'): 
        from training_tokenizers.sent_tok import SentencePieceTokenizer
        tokenizer = SentencePieceTokenizer("./all_tokenizer_files/sentencepiece_tokenizer_files_ehm/sentencepiece.json")
    elif(tok_name == 'gpt' or 'itrans' in tok_name):
        from training_tokenizers.itrans_tokenizer_hf import *
        tokenizer = BPETokenizer(vocab_file="./all_tokenizer_files/itrans_tokenizer_files_ehm/vocab.json", merges_file="./all_tokenizer_files/itrans_tokenizer_files_ehm/merges.json")
    else:
        tokenizer = None

    lang = 'dks'
    file_loading_function = load_dks_data

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = T5ForConditionalGeneration.from_pretrained(f"./models/models_v5/t5-dks-{tok_name}_{lang}-best")
    model.to(device)

    model.gradient_checkpointing_enable()

    test_file = f"./datasets/dks_{args.lang}_test.tsv"

    # print(tok_name, test_file)
    print(f"Tokenizer: {tok_name}\nTest File: {test_file}")
    test_model(model, tokenizer, test_file, device, output_csv=f'./results/results_dks/{tok_name}_{lang}_{test_file.split("/")[-1].split(".")[0]}-results.csv',batch_size=128, file_loading_function=file_loading_function)

    # print(f"Done Results for {tok_name} - {test_file}")
    print("_"*140)


# sample run command: python eval_t5_dks.py --tok_name gc --lang hin