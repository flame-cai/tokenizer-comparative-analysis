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

def load_g2p_data(filename):
    print(f'Loading G2P dataset from {filename}...')
    df = pd.read_csv(filename, delimiter="\t", header=None)
    # df[0] = df[0].astype(str).apply(lambda x: transliterate.process('Devanagari','ITRANS', x))
    # df.to_csv(f'{filename.replace(".tsv","")}_translit.tsv', sep="\t",header=None, index=None)
    # print(df)
    # df[1] = df[1].apply(lambda x: x.replace(" ",""))

    # graphemes = df[0].astype(str).tolist()
    graphemes = df[0].astype(str).tolist()
    phonemes = [str(p).split(",") for p in df[1].tolist()]
    # print(f"Phonemes: {phonemes}")
    
    print(f"Loaded {len(graphemes)} grapheme-phoneme pairs")
    return graphemes, phonemes

class G2PDataset(torch.utils.data.Dataset):
    def __init__(self, graphemes, phonemes, tokenizer, max_input_len=128, max_output_len=128, cache_tokenization=True):
        self.graphemes = graphemes
        self.phonemes = phonemes
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.cache_tokenization = cache_tokenization
        self.cache = {}  # Add caching
    
    def __len__(self):
        return len(self.graphemes)

    def __getitem__(self, idx):
        if self.cache_tokenization and idx in self.cache:
            return self.cache[idx]
            
        # Format input as "grapheme to phoneme: <grapheme>"
        input_text = f"G-{self.graphemes[idx]} P:"
        target_text = f"{self.phonemes[idx][0]} "

        inputs = self.tokenizer(input_text, max_length=self.max_input_len, padding="max_length", truncation=True, return_tensors="pt")
        targets = self.tokenizer(target_text, max_length=self.max_output_len, padding="max_length", truncation=True, return_tensors="pt")

        # print(f"Tuple: {tuple(self.phonemes[idx])}")

        item = {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": targets.input_ids.squeeze(),
            "grapheme": self.graphemes[idx],
            "phoneme": tuple(self.phonemes[idx])
        }
        
        if self.cache_tokenization:
            self.cache[idx] = item
            
        return item

def custom_collate_fn(batch):
    """Custom collate function to handle phoneme tuples properly"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    graphemes = [item["grapheme"] for item in batch]
    phonemes = [item["phoneme"] for item in batch]  # list of lists stays intact

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "grapheme": graphemes,
        "phoneme": phonemes,
    }

# Load dataset
# train_file = "datasets/wiki_squad_hi_train_trunc.jsonl"
# val_file = "datasets/wiki_squad_hi_val_trunc.jsonl"

# train_contexts, train_questions, train_answers = load_squad_hindi(train_file)
# val_contexts, val_questions, val_answers = load_squad_hindi(val_file)

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

def normalized_edit_distance(s1, s2):
    if len(s1) == 0 and len(s2) == 0:
        return 0
    return edit_distance(s1, s2) / max(len(s1), len(s2))

def calculate_wer(hypothesis, reference):
    
    hyp_words = hypothesis.split()
    ref_words = reference.split()
    
    if len(ref_words) == 0:
        return 0, 0
    
    edit_dist = edit_distance(hyp_words, ref_words)
    return edit_dist, len(ref_words)

def calculate_per(hypothesis, reference):
       
    hyp_phones = hypothesis.split()
    ref_phones = reference.split()
    
    if len(ref_phones) == 0:
        return 0, 0
    
    edit_dist = edit_distance(hyp_phones, ref_phones)
    
    # if edit_dist>0:
    #     print(f"Pred: {hypothesis}\nTruth: {reference}\nEdit Distance: {edit_dist}")

    return edit_dist, len(ref_phones)

def test_model(model, tokenizer, test_file, device, batch_size=64, output_csv="./results/results_g2p/results.csv", file_loading_function=load_g2p_data):
    # Load test data
    test_graphemes, test_phonemes = file_loading_function(test_file)
    test_dataset = G2PDataset(test_graphemes, test_phonemes, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True, prefetch_factor=10, persistent_workers=True, collate_fn=custom_collate_fn)
    
    exact_matches = 0
    total_edit_distance = 0
    total_predictions = 0
    total_per_edit_distance = 0
    total_per_ref_length = 0

    torch.cuda.empty_cache()
    model.eval()

    with torch.no_grad(), open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['input','real_answer', 'pred_answer', 'exact_match', 'norm_edit_score','per'])
        
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_answers = batch['phoneme'] 

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128, use_cache=True)
            predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                
            # Process each prediction in the batch
            for idx in range(len(predictions)):
                pred = predictions[idx].strip()
                truth_options = list(batch_answers[idx])

                exact_match = pred in truth_options
                if exact_match:
                    exact_matches += 1

                # normalized edit dist
                ned = min(normalized_edit_distance(pred, t) for t in truth_options)
                total_edit_distance += ned

                # per for min edit distance
                def phone_edit_distance(hyp, ref):
                    hyp_phones = hyp.split()
                    ref_phones = ref.split()
                    return edit_distance(hyp_phones, ref_phones)

                # best truth based on phone level ed
                best_truth_for_per = min(truth_options, key=lambda t: phone_edit_distance(pred, t))
                
                # per calculation
                per_edit_dist, per_ref_len = calculate_per(pred, best_truth_for_per)
                total_per_edit_distance += per_edit_dist
                total_per_ref_length += per_ref_len
                individual_per = (per_edit_dist / per_ref_len * 100) if per_ref_len > 0 else 0.0

                # pred count increase
                total_predictions += 1
                
                csv_writer.writerow([tokenizer.decode(input_ids[idx], skip_special_tokens=True), "|".join(truth_options), pred, int(exact_match), f"{ned:.4f}", f"{individual_per:.4f}"
                ])

    # Calculate final metrics
    exact_match_accuracy = exact_matches / total_predictions
    average_edit_distance = total_edit_distance / total_predictions
    micro_avg_per = (total_per_edit_distance / total_per_ref_length * 100) if total_per_ref_length > 0 else 0.0

    print(f"\nFinal Metrics:")
    print(f"Total predictions processed: {total_predictions}")
    print(f"WER: {((1-exact_match_accuracy)*100):.4}%")
    print(f"Micro-averaged PER: {micro_avg_per:.4f}%")

    return {
        "exact_match": exact_match_accuracy,
        "edit_distance": average_edit_distance,
        "per": micro_avg_per
    }


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--tok_name", type=str, default="gc", help="Tokenizer name", choices=["gc", "gbpe", "wp", "sp", "itrans", "bpe"])
parser.add_argument("--lang", type=str, default="hin", help="Tokenizer name", choices=["hin", "en", "mr"])
parser.add_argument("--eval_all", type=bool, default=False, help="Evaluate all types")
args = parser.parse_args()

tok_names = [args.tok_name] if not args.eval_all else ["gc", "gbpe", "wp", "sp", "bpe"]

for tok_name in tok_names:

    if(tok_name == 'gc'):
        from training_tokenizers.gc_tokenizer_hf import DevanagariTokenizer, load_vocabulary
        # tokenizer = DevanagariTokenizer(vocab_dict=load_vocabulary(vocab_file_path="vocab_gc.json"))
        tokenizer = DevanagariTokenizer(vocab_dict=load_vocabulary(vocab_file_path="./all_tokenizer_files/gc_tokenizer_files_ehm_ipa/vocab.json"))
    elif(tok_name == 'gbpe'):
        from training_tokenizers.gbpe_tokenizer_hf import *
        tokenizer = GBPETokenizer(vocab_file="./all_tokenizer_files/gbpe_tokenizer_files_ehm_ipa/vocab.json", merges_file="./all_tokenizer_files/gbpe_tokenizer_files_ehm_ipa/merges.json")
    elif(tok_name=='bpe'):
        from training_tokenizers.bpe_tokenizer_hf import *
        tokenizer = BPETokenizer(vocab_file="./all_tokenizer_files/bpe_tokenizer_files_ehm_ipa/vocab.json", merges_file="./all_tokenizer_files/bpe_tokenizer_files_ehm_ipa/merges.json")
    elif(tok_name == 'wp'):
        # Initialize BERT (wp) tokenizer 
        # tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        from training_tokenizers.wp_tok import WordPieceTokenizer
        tokenizer = WordPieceTokenizer(tokenizer_path="./all_tokenizer_files/wordpiece_tokenizer_files_ehm_ipa/wordpiece.json")
    elif(tok_name=='sp'): 
        from training_tokenizers.sent_tok import SentencePieceTokenizer
        tokenizer = SentencePieceTokenizer("./all_tokenizer_files/sentencepiece_tokenizer_files_ehm_ipa/sentencepiece.json")
    elif(tok_name == 'gpt' or 'itrans' in tok_name):
        # # Initialize GPT-2 tokenizer (compatible with T5 model)
        # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # tokenizer.pad_token = tokenizer.eos_token  # Ensure padding tokens are set properly
        from training_tokenizers.itrans_tokenizer_hf import *
        # transliterate_input = False if lang == 'en' else True
        tokenizer = BPETokenizer(vocab_file="./all_tokenizer_files/itrans_tokenizer_files_ehm_ipa/vocab.json", merges_file="./all_tokenizer_files/itrans_tokenizer_files_ehm_ipa/merges.json", transliterate_input=True)
    else:
        tokenizer = None

    lang = 'ipa'
    file_loading_function = load_g2p_data

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = T5ForConditionalGeneration.from_pretrained(f"./models/models_v5/t5-g2p-{tok_name}_{lang}-best")
    # model = T5ForConditionalGeneration.from_pretrained(f"./models/models_v5/t5-g2p-model-{tok_name}_{lang}-30")
    model.to(device)

    model.gradient_checkpointing_enable()

    test_file = f"./datasets/g2p_{args.lang}_test.tsv" if tok_name != "itrans" else f"./datasets/g2p_{args.lang}_test_translit.tsv"

    # print(tok_name, test_file)
    print(f"Tokenizer: {tok_name}\nTest File: {test_file}")
    test_model(model, tokenizer, test_file, device, output_csv=f'./results/results_g2p/{tok_name}_{lang}_{test_file.split("/")[-1].split(".")[0]}-results.csv',batch_size=64, file_loading_function=file_loading_function)

    # print(f"Done Results for {tok_name} - {test_file}")
    print("_"*140)


# sample run command: python eval_t5_g2p.py --tok_name gc --lang hin