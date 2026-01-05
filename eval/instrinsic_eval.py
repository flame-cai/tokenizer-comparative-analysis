import argparse
import numpy as np
from collections import Counter
import re
import pandas as pd
import sys
import os
from aksharamukha import transliterate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# clean text and change devanagari numbers to english numbers
def clean_text(text):
    text = re.sub(r"[।!,?.;’:\"'()\[\]{}<>।॥|\-_@%$]", " ", text)
    # text = re.sub(r"[०१२३४५६७८९]", lambda x: str(int(x.group(0))), text)
    # text = text.replace("संवत्","संवत")

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)
    return text

def load_flores_data(language):
    print(f"Debug: Loading Flores data for language: {language}")
    # Map user input codes to Flores dataset codes
    lang_map = {
        'hi': 'hin_Deva',
        'mr': 'mar_Deva',
        'en': 'eng_Latn'
    }
    if language == "mixed":
        all_text = []
        for i in lang_map.keys():
            flores_code = lang_map[i]
            with open(f'intrinsic_eval/floresp-v2.0-rc.3/dev/dev.{flores_code}') as f:
                text_list = f.readlines()
            all_text.extend(text_list)
        text_list = all_text
        print(f"Debug: Creating DataFrame with {len(text_list)} lines")
        pd_dict = {f'{language}_text':text_list}
    else:
        flores_code = lang_map[language]
        print(f"Debug: Using Flores code: {flores_code}")
        with open(f'intrinsic_eval/floresp-v2.0-rc.3/dev/dev.{flores_code}') as f:
            text_list = f.readlines()
        # with open(f'intrinsic_eval/floresp-v2.0-rc.3/devtest/devtest.{flores_code}') as f:
        #     text_list_new = f.readlines()
        # text_list += text_list_new
        print(f"Debug: Creating DataFrame with {len(text_list)} lines")
        pd_dict = {f'{language}_text':text_list}

    # text_list += text_list_new
    df = pd.DataFrame(pd_dict)
    text_list = df[f'{language}_text'].str.cat(sep=' ')
    print(f"Debug: Final text length: {type(text_list)}")
    return text_list

def get_fertility(text, tokenizer):
    # print("Debug: Calculating fertility")
    words = text.split()
    word_count = len(words)
    # print(f"Debug: Word count: {word_count}")
    
    tokens = []
    for word in words:
        # print(tokenizer(word, add_special_tokens=False)['input_ids'])
        tokens.extend(tokenizer(word, add_special_tokens=False)['input_ids'])
    
    return f'{(len(tokens)/word_count):.4f}'

def get_renyi_efficiency(text, tokenizer, alpha=2):
    tokenizer_output = [tokenizer.decode(i) for i in tokenizer(text, add_special_tokens=False)['input_ids']]
    
    # print(f"Debug: Token output length: {len(tokenizer_output)}")
    token_counts = Counter(tokenizer_output)
    total_tokens = len(tokenizer_output)
    probabilities = np.array([count / total_tokens for count in token_counts.values()])
    # Calculate Renyi entropy
    if alpha == 1:
        # For alpha=1, use Shannon entropy formula
        renyi_entropy = -np.sum(probabilities * np.log2(probabilities))
    else:
        # print("Debug: Using Renyi entropy formula")
        renyi_entropy = 1 / (1 - alpha) * np.log2(np.sum(probabilities ** alpha))
    
    # Calculate maximum possible entropy (uniform distribution)
    max_entropy = np.log2(len(token_counts))
    # Calculate Renyi efficiency
    return f'{(renyi_entropy / max_entropy):.4f}'

def get_percentile_frequency(text, tokenizer, gamma1=0.03, gamma2=0.83):
    tokenizer_output = [tokenizer.decode(i) for i in tokenizer(text, add_special_tokens=False)['input_ids']]
    
    # print(f"Debug: Token output length: {len(tokenizer_output)}")
    token_counts = Counter(tokenizer_output)
    total_tokens = len(tokenizer_output)

    # Calculate probabilities
    token_probs = {token: count / total_tokens for token, count in token_counts.items()}
    # print(f"Debug: Number of unique tokens: {len(token_probs)}")
    
    # Sort probabilities in descending order
    sorted_probs = sorted(token_probs.values(), reverse=True)

    # Calculate indices corresponding to the desired percentiles
    start_index = int(gamma1 * len(sorted_probs))
    end_index = int(gamma2 * len(sorted_probs))
    
    # Get the probability thresholds
    start_threshold = sorted_probs[start_index]
    end_threshold = sorted_probs[end_index]
    
    # Sum probabilities within the percentile range and return
    return f'{sum(prob for prob in token_probs.values() if end_threshold <= prob < start_threshold):.4f}'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--language', type=str, required=True, help='Language code (en/hi/mr)', choices = ['en', 'hi', 'mr','mixed'])
    parser.add_argument('-t','--tokenizer', type=str, choices=['gc', 'gbpe', 'bpe', 'wp', 'sp', 'gpt', 'itrans'])
    parser.add_argument('-m','--metric', type=str, choices=['fertility', 'renyi', 'percentile'])
    parser.add_argument('--all', default=False, action='store_true', help='Run all metrics for all tokenizers')
    args = parser.parse_args()

    # Load text
    text = load_flores_data(args.language)

    if args.all:
        # Print markdown table header
        print("\n| Tokenizer | Fertility(low) | Renyi Eff.(high) | Pctile Freq.(high) |")
        print("|-----------|----------------|------------------|-------------------|")
        
        for tok_name in ['gc', 'gbpe', 'wp', 'sp', 'bpe', 'itrans']:
            if(tok_name == 'gc'):
                from training_tokenizers.gc_tokenizer_hf import DevanagariTokenizer, load_vocabulary
                tokenizer = DevanagariTokenizer(vocab_dict=load_vocabulary(vocab_file_path="all_tokenizer_files/gc_tokenizer_files_ehm/vocab.json"))
            elif(tok_name == 'gbpe'):
                from training_tokenizers.gbpe_tokenizer_hf import GBPETokenizer
                tokenizer = GBPETokenizer(vocab_file="all_tokenizer_files/gbpe_tokenizer_files_ehm/vocab.json", merges_file="all_tokenizer_files/gbpe_tokenizer_files_ehm/merges.json")
            elif(tok_name=='bpe'):
                from training_tokenizers.bpe_tokenizer_hf import BPETokenizer
                tokenizer = BPETokenizer(vocab_file="all_tokenizer_files/bpe_tokenizer_files_ehm_ipa/vocab.json", merges_file="all_tokenizer_files/bpe_tokenizer_files_ehm_ipa/merges.json")
            elif(tok_name == 'wp'):
                from training_tokenizers.wp_tok import WordPieceTokenizer
                tokenizer = WordPieceTokenizer(tokenizer_path="all_tokenizer_files/wordpiece_tokenizer_files_ehm/wordpiece.json")
            elif(tok_name=='sp'): 
                from training_tokenizers.sent_tok import SentencePieceTokenizer
                tokenizer = SentencePieceTokenizer("all_tokenizer_files/sentencepiece_tokenizer_files_ehm/sentencepiece.json")
            elif(tok_name == 'gpt' or 'itrans' in tok_name):
                from training_tokenizers.itrans_tokenizer_hf import BPETokenizer
                tokenizer = BPETokenizer(vocab_file="all_tokenizer_files/itrans_tokenizer_files_ehm/vocab.json", merges_file="all_tokenizer_files/itrans_tokenizer_files_ehm/merges.json", transliterate_input=True)

            fertility = get_fertility(text, tokenizer)
            renyi = get_renyi_efficiency(text, tokenizer)
            pctile = get_percentile_frequency(text, tokenizer)
            
            print(f"| {tok_name} | {fertility} | {renyi} | {pctile} |")

    else:
        # Initialize tokenizer
        if args.tokenizer == 'gc':
            from training_tokenizers.gc_tokenizer_hf import DevanagariTokenizer, load_vocabulary
            tokenizer = DevanagariTokenizer(vocab_dict=load_vocabulary(vocab_file_path="all_tokenizer_files/gc_tokenizer_files_ehm/vocab.json"))
            # tokenizer = DevanagariTokenizer(vocab_dict=load_vocabulary(vocab_file_path="vocab_gc.json"))
        elif args.tokenizer == 'gbpe':
            from training_tokenizers.gbpe_tokenizer_hf import GBPETokenizer
            tokenizer = GBPETokenizer(vocab_file="all_tokenizer_files/gbpe_tokenizer_files_ehm/vocab.json", merges_file="all_tokenizer_files/gbpe_tokenizer_files_ehm/merges.json")
        elif args.tokenizer == 'bpe':
            from training_tokenizers.bpe_tokenizer_hf import BPETokenizer
            tokenizer = BPETokenizer(vocab_file="all_tokenizer_files/bpe_tokenizer_files_ehm/vocab.json", merges_file="all_tokenizer_files/bpe_tokenizer_files_ehm/merges.json")
        elif args.tokenizer == 'wp':
            from training_tokenizers.wp_tok import WordPieceTokenizer
            tokenizer = WordPieceTokenizer(tokenizer_path="all_tokenizer_files/wordpiece_tokenizer_files_ehm/wordpiece.json")
        elif args.tokenizer == 'sp':
            from training_tokenizers.sent_tok import SentencePieceTokenizer
            tokenizer = SentencePieceTokenizer("all_tokenizer_files/sentencepiece_tokenizer_files_ehm/sentencepiece.json")
        elif args.tokenizer in ['gpt', 'itrans']:
            from training_tokenizers.itrans_tokenizer_hf import BPETokenizer
            tokenizer = BPETokenizer(vocab_file="all_tokenizer_files/itrans_tokenizer_files_ehm/vocab.json", merges_file="all_tokenizer_files/itrans_tokenizer_files_ehm/merges.json", transliterate_input=True)
            text = transliterate.process("Devanagari", "ITRANS", text)

        # Calculate metric
        if args.metric == 'fertility':
            result = get_fertility(text, tokenizer)
        elif args.metric == 'renyi':
            result = get_renyi_efficiency(text, tokenizer)
        else:  # percentile
            result = get_percentile_frequency(text, tokenizer)

        print(f"{args.metric} score for {args.tokenizer} tokenizer on {args.language}: {result}")

if __name__ == "__main__":
    main()

# Sample Usage: python instrinsic_eval.py -l hi -t gc -m fertility
