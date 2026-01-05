from transformers import PreTrainedTokenizer, BatchEncoding
from typing import List, Optional, Tuple, Dict, Union, Iterator
import json
import torch
import os
from collections import defaultdict
import re
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import argparse



def ipa_tokenize(ipa_string):

    # ranges that IPA characters usually belong to
    IPA_RANGES = [
        (0x0020, 0x007F),   # Basic Latin
        (0x00A0, 0x00FF),   # Latin-1 Supplement
        (0x0100, 0x017F),   # Latin Extended-A
        (0x0180, 0x024F),   # Latin Extended-B
        (0x0250, 0x02AF),   # IPA Extensions
        (0x02B0, 0x02FF),   # Spacing Modifier Letters
        (0x0300, 0x036F),   # Combining Diacritics
        (0x0370, 0x03FF),   # Greek and Coptic (for θ, β, etc.)
        (0x1D00, 0x1D7F),   # Phonetic Extensions
        (0x1D80, 0x1DBF),   # Phonetic Extensions Supplement
    ]

    def is_ipa_char(ch):
        cp = ord(ch)
        return any(start <= cp <= end for start, end in IPA_RANGES)

    if not all(is_ipa_char(ch) or ch.isspace() for ch in ipa_string):
        return ipa_string
    
    combining_marks = frozenset({
        '̪', 'ː', 'ʱ', 'ᵊ','ʰ', '̃', '̩', '̯', '̤', '̥', '̬', '̰', '̱', '̲', '̳', '̴', '̵', '̶',
        '̷', '̸', '̹', '̺', '̻', '̼', '̽', '̾', '̿', '̀', '́', '͆', '͇', '͈', '͉',
        '͊', '͋', '͌', '͍', '͎', '͐', '͑', '͒', '͓', '͔', '͕', '͖', '͗', '͘', '͙',
        '͚', '͛', '͜', '͝', '͞', '͟', '͠', '͡', '͢', 'ͣ', 'ͤ', 'ͥ', 'ͦ', 'ͧ', 'ͨ',
        'ͩ', 'ͪ', 'ͫ', 'ͬ', 'ͭ', 'ͮ', 'ͯ', '̚', '̣', '̇', '̈', '̊', '̋', '̌', '̍', '̎',
        '̏', '̓', '̔', '̕', '̖', '̗', '̘', '̙', '̜', '̝', '̞', '̟', '̠', '̡', '̢',
        '̦', '̨',
    })
    digits = frozenset('')
    end_markers = frozenset(['͡'])
    
    graphemes = []
    temp_grapheme = []

    def is_devanagari(c):
        return '\u0900' <= c <= '\u097F'

    for char in ipa_string:
        if is_devanagari(char):  
            # flush any IPA token first
            if temp_grapheme:
                graphemes.append(''.join(temp_grapheme))
                temp_grapheme = []
            graphemes.append(char)  # keep Devanagari as-is
        elif char not in combining_marks:
            if temp_grapheme and not (temp_grapheme[-1] in end_markers):
                if char in digits:
                    graphemes.append(''.join(temp_grapheme))
                    temp_grapheme = []
                    graphemes.append(char)
                else:
                    graphemes.append(''.join(temp_grapheme))
                    temp_grapheme = [char]
            else:
                temp_grapheme.append(char)
        else:
            temp_grapheme.append(char)

    if temp_grapheme:
        graphemes.append(''.join(temp_grapheme))

    # cleanup
    graphemes = [g.replace('\u200c', '').replace('\u200d', '').replace('\u200b', '') 
                 for g in graphemes]

    return " ".join(graphemes)

def DevaIPAFix(text, lang):
    # hindi
    if lang == 'hi':
        text = re.sub(r'g','ɡ',text)
        text = re.sub(r'(?:b|d|ɖ|d͡ʒ|ɡ|ɽ|d̪|ʋ)ʰ', lambda m: m.group(0)[:-1] + "ʱ", text)
        text = re.sub(r'ẽː([ɡd])(?=(?:[ʰʱ][ .]|[ .]|$))', r'eːŋ\1', text)
        text = re.sub(r'ẽː([ɡd])', r'eːŋ.\1', text)
        text = re.sub("õː","õː", text)
        text = re.sub("ĩː","ĩː", text)
        text = re.sub("ẽː","ẽː",text)
        text = re.sub('ũː', "ũː", text)
        text = re.sub("d͡ʒɲ","ɡ.j", text)
        text = re.sub(r'əũ([d̪ɖɡ])', lambda m: {'d̪':'ɔːn','ɖ':'ɔːɳ','ɡ':'ɔːŋ'}.get(m.group(1), 'ɔ̃ː') + m.group(1), text)
        text = re.sub(r'əũ', 'ɔ̃ː', text)
        text = re.sub("əu","ɔː",text)
        text = re.sub(r'u(?!ː)', 'ʊ', text)
        text = re.sub(r'ũ(?!ː)', 'ʊ̃', text)
        text = re.sub("n̪","n",text)
        text = re.sub("s̪","s", text)
        text = re.sub("əɪ̃","ɛ̃ː", text)
        text = re.sub("əɪj","ə̯i.j", text)
        text = re.sub("əɪ","ɛː",text)
        text = re.sub(r'(?:j|ɾ|ʋ|m|n|l|ɦ|ɳ|ɡ|d̪|s|t̪|ɖ|d͡ʒ)ə(?=[\s.])', lambda m: m.group(0)[0] + "ᵊ", text)

    # marathi
    if lang == 'mr':
        text = text.replace("ː","")
        text = re.sub(r'ɭ', "ɭ̆", text)
        text = re.sub(r"ʃ","ɕ", text)
        text = re.sub(r'g','ɡ',text)
        text = re.sub(r"ɑ","a",text)
        text = re.sub(r"n̪", "n", text)
        text = re.sub(r"s̪", "s", text)
        text = re.sub(r"d͡ʒənm", "d͡ʑənm", text)
        text = re.sub(r'(?:b|d|ɖ|d͡ʒ|ɡ|ɽ|d̪|ʋ|ʑ|d̪)ʰ', lambda m: m.group(0)[:-1] + "ʱ", text)
        text = re.sub(r'd͡ʒʱe', "d͡ʑʱe", text)
        text = re.sub(r"d͡ʒaneʋaɾi", "d͡ʑaneʋaɾi", text)
    
    text = ipa_tokenize(text)

    return text

def clean_text(text, lang):
    text = str(text).strip()  # Remove extra spaces
    text = re.sub(r"[;\"'()\[\]{}<>\@$]", "", text)  # Remove punctuation
    text = text.replace('\u200d', '').replace('\u200c', '').replace('\u200b', '')
    # remove non hindi marathi, non numeric characters
    # text = re.sub(r'[^\u0900-\u097F\u0980-\u09FFa-zA-Z0-9 ,\.।॥%:;?!\'"-]', '', text)
    # remove non devanagari, non numeric characters
    # text = re.sub(r'[^\u0900-\u097Fa-zA-Z0-9 ,\.।॥%:;?!\'"-]', '', text)
    # remove non devanagari, non numeric, non ipa characters
    text = re.sub(r'[^\u0900-\u097F\u0250-\u02AF\u02B0-\u02FF\u0300-\u036Fa-zA-Z0-9 ,\.।॥%:;?!\'"-]', '', text)
    # remove non english, non numeric characters
    # text = re.sub(r'[^a-zA-Z0-9 ,\.।॥%:;?!\'"-]', '', text)
    # too many ., replace with one .
    text = re.sub(r'\.{2,}', '.', text)
    # IPA fixes fro aksharamukha
    text = DevaIPAFix(text, lang)
    # multiple space to be replaced by one
    text = re.sub(r'\s+', ' ', text)
    # print(text)
    return text


class BPETokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_dict=None,
        merges_dict=None,
        vocab_file=None,
        merges_file=None,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        **kwargs
    ):
        if vocab_file:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_dict = json.load(f)
        if merges_file:
            with open(merges_file, 'r', encoding='utf-8') as f:
                merges_dict = json.load(f)

        self.vocab = vocab_dict if vocab_dict else {}
        
        self.merges = {}
        self.merge_ranks = {}
        
        if merges_dict:
            for rank, (k, v) in enumerate(merges_dict.items()):
                if isinstance(k, str) and '|' in k:
                    key = tuple(k.split('|'))
                    self.merges[key] = v  
                    self.merge_ranks[key] = rank
                elif isinstance(k, tuple) and len(k) == 2:
                    self.merges[k] = v
                    self.merge_ranks[k] = rank
                else:
                    print(f"merge key format: {k}")
        
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
        special_tokens = [unk_token, pad_token, eos_token]
        for special_token in special_tokens:
            if special_token not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[special_token] = new_id
                self.ids_to_tokens[new_id] = special_token

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            eos_token=eos_token,
            **kwargs
        )

        self._vocab_size = len(self.vocab)
        self.name_or_path = "bpe-tokenizer"

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def _get_pairs(self, tokens):
        return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}

    def _tokenize(self, text: str) -> List[str]:
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            # start with chars
            word_tokens = list(word)
            
            while True:
                pairs = self._get_pairs(word_tokens)
                if not pairs:
                    break

                valid_pairs = [(pair, self.merge_ranks[pair]) 
                              for pair in pairs if pair in self.merges]
                
                if not valid_pairs:
                    break

                merge_candidate = min(valid_pairs, key=lambda x: x[1])[0]

                first, second = merge_candidate
                merged_token = self.merges[merge_candidate]
                
                new_word = []
                j = 0
                while j < len(word_tokens):
                    if (j < len(word_tokens) - 1 and 
                        word_tokens[j] == first and 
                        word_tokens[j + 1] == second):
                        new_word.append(merged_token)
                        j += 2
                    else:
                        new_word.append(word_tokens[j])
                        j += 1
                
                word_tokens = new_word
            
            if i > 0:
                result.append("Ġ")

            result.extend(word_tokens)
        
        return result

    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() > 1:
                token_ids = token_ids.squeeze(0)
            token_ids = token_ids.tolist()
        elif not isinstance(token_ids, (list, tuple)):
            token_ids = [token_ids]

        tokens = [self._convert_id_to_token(id) for id in token_ids]
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.all_special_tokens]
        
        text = ""
        for token in tokens:
            if token == "Ġ":
                text += " "
            else:
                text += token
        
        if clean_up_tokenization_spaces:
            text = ' '.join(text.split())
        
        return text

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            raise OSError(f"Vocabulary path ({save_directory}) should be a directory")

        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        merges_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + "merges.json"
        )

        # save vocabulary
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

        merges_dict = {}
        sorted_merges = sorted(self.merges.items(), key=lambda x: self.merge_ranks[x[0]])
        for (k, v) in sorted_merges:
            if isinstance(k, tuple) and len(k) == 2:
                key = f"{k[0]}|{k[1]}"
                merges_dict[key] = v
            else:
                print(f"merge key format: {k}")
        
        with open(merges_file, "w", encoding="utf-8") as f:
            json.dump(merges_dict, f, ensure_ascii=False)

        return vocab_file, merges_file
    
    @classmethod
    def from_pretrained(cls, save_directory: str, **kwargs):
        vocab_file = os.path.join(save_directory, "vocab.json")
        merges_file = os.path.join(save_directory, "merges.json")

        if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
            raise OSError(f"Can't find vocab.json and merges.json in {save_directory}")

        # Load vocabulary
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)

        # Load merges
        with open(merges_file, "r", encoding="utf-8") as f:
            merges_dict = json.load(f)

        return cls(vocab_dict=vocab_dict, merges_dict=merges_dict, **kwargs)

    def __call__(
        self,
        text_input,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> BatchEncoding:
        return self.encode(
            text_input,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            **kwargs
        )

    def encode(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> BatchEncoding:
        def encode_single(text: str) -> Dict[str, List[int]]:
            tokens = self._tokenize(text)
            token_ids = [self._convert_token_to_id(token) for token in tokens]
            
            if self.eos_token_id is not None:
                token_ids.append(self.eos_token_id)
                
            if truncation and max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            attention_mask = [1] * len(token_ids)
            return {"input_ids": token_ids, "attention_mask": attention_mask}

        if isinstance(text, str):
            encoding = encode_single(text)
            
            if padding == "max_length" and max_length:
                pad_length = max_length - len(encoding["input_ids"])
                if pad_length > 0:
                    encoding["input_ids"].extend([self.pad_token_id] * pad_length)
                    encoding["attention_mask"].extend([0] * pad_length)
            
            if return_tensors == "pt":
                encoding = {k: torch.tensor(v, dtype=torch.long) for k, v in encoding.items()}
            
            return BatchEncoding(encoding)

        elif isinstance(text, (list, tuple)):
            encodings = [encode_single(t) for t in text]
            
            if padding == "max_length" and max_length:
                for enc in encodings:
                    pad_length = max_length - len(enc["input_ids"])
                    if pad_length > 0:
                        enc["input_ids"].extend([self.pad_token_id] * pad_length)
                        enc["attention_mask"].extend([0] * pad_length)
            
            batch_encoding = {
                "input_ids": [enc["input_ids"] for enc in encodings],
                "attention_mask": [enc["attention_mask"] for enc in encodings]
            }
            
            if return_tensors == "pt":
                batch_encoding = {k: torch.tensor(v, dtype=torch.long) for k, v in batch_encoding.items()}
            
            return BatchEncoding(batch_encoding)
        else:
            raise ValueError("text must be either a string or a list/tuple of strings")

    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs
    ) -> BatchEncoding:
        if add_special_tokens:
            if self.eos_token_id is not None and self.eos_token_id not in ids:
                ids = ids + [self.eos_token_id]
        
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]

        if padding:
            padding_side = kwargs.get("padding_side", "right")
            if max_length and len(ids) < max_length:
                pad_length = max_length - len(ids)
                if padding_side == "right":
                    ids = ids + [self.pad_token_id] * pad_length
                else:
                    ids = [self.pad_token_id] * pad_length + ids

        encoded = {"input_ids": ids}
        
        if return_attention_mask:
            attention_mask = [1] * len(ids)
            if padding and max_length:
                attention_mask.extend([0] * (max_length - len(attention_mask)))
            encoded["attention_mask"] = attention_mask

        if return_tensors == 'pt':
            encoded = {k: torch.tensor([v] if prepend_batch_axis else v) 
                    for k, v in encoded.items()}

        return BatchEncoding(encoded)

    @classmethod
    def train_tokenizer(cls, 
                    corpus_folders: List[str], 
                    vocab_size: int = 30000, 
                    min_frequency: int = 2,
                    special_tokens: List[str] = None) -> "BPETokenizer":
        import tempfile
        
        tokenizer = Tokenizer(models.BPE())
        
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        # Prepare trainer
        special_tokens_list = special_tokens or ["<unk>", "<pad>", "<eos>", "Ġ"]
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens_list,
            show_progress=True
        )
        
        def files_iterator():
            for folder in corpus_folders:
                if not os.path.exists(folder):
                    print(f"Warning: Folder {folder} does not exist")
                    continue
                    
                for filename in os.listdir(folder):
                    if filename.endswith('.txt'):
                        lang = 'hi' if 'hindi-wiki' in folder else 'mr' if 'marathi-wiki' in folder else 'en'
                        file_path = os.path.join(folder, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if line.strip():
                                        yield clean_text(line.strip(), lang)
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")
            
            # Add SQuAD dataset
            print("Loading English SQuAD...")
            from datasets import load_dataset
            ds = load_dataset("rajpurkar/squad")
            for text in ds['train']['context']:
                if text.strip():
                    yield clean_text(text.strip(), 'en')

        tokenizer.train_from_iterator(files_iterator(), trainer=trainer)
        
        vocab = tokenizer.get_vocab()
        
        if "Ġ" not in vocab:
            vocab["Ġ"] = len(vocab)
        
        merges = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            # save the trained tokenizer
            tokenizer.save(os.path.join(tmpdir, "tokenizer.json"))
            with open(os.path.join(tmpdir, "tokenizer.json"), 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
            
            if 'model' in tokenizer_data and 'merges' in tokenizer_data['model']:
                merge_list = tokenizer_data['model']['merges']
                for rank, merge_str in enumerate(merge_list):
                    parts = merge_str.split(' ')
                    if len(parts) == 2:
                        key = (parts[0], parts[1])
                        merged = parts[0] + parts[1]
                        merges[key] = merged
        
        return cls(vocab_dict=vocab, merges_dict=merges)
    
def load_vocabulary(vocab_file_path: str, merges_file_path: str) -> Tuple[Dict[str, int], Dict[Tuple[str, str], str]]:
    with open(vocab_file_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    with open(merges_file_path, 'r', encoding='utf-8') as f:
        merges_dict = json.load(f)
        merges = {tuple(k.split('|')): v for k, v in merges_dict.items()}
    
    return vocab, merges


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("--corpus_folders", type=str, nargs='+', default=["./datasets/hindi-wiki/train/train","./datasets/hindi-wiki-ipa/train","./datasets/marathi-wiki/train/train","./datasets/marathi-wiki-ipa/train"], 
                        help="Paths to folders containing text files for training")
    parser.add_argument("--output_dir", type=str, default = "./all_tokenizer_files/bpe_tokenizer_files_ehm_ipa", help="Directory to save the tokenizer")
    parser.add_argument("--vocab_size", type=int, default=17411, help="Size of the vocabulary")
    parser.add_argument("--min_frequency", type=int, default=1, help="Minimum frequency for a token")
    parser.add_argument("--test_text", type=str, default="शिक्षा द्वारा राष्ट्रों, जातियों अथवा घार्मिक समूहों के बीच आपसी सद्भावना, सहिष्णुता और मंत्री का विकास होगा. मराठी साहित्याचे विविध प्रकार म्हणजेच कथा, कादंबरी, ललित लेख, प्रवास वर्णनं, समीक्षा लेखन, नाटक वगैरे This sentence is a sample sentence. kut͡ʃʰ s̪ɑːl pəɦleː t̪ʰɑː 2012. d͡ʒʱoːl. ʊ t̪ s ʊ k t̪ ɑː', 'ʊ t̪ . s ʊ k . t̪ ɑː", help="Optional text to test the tokenizer")
    
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Training BPE tokenizer on folders: {', '.join(args.corpus_folders)} with vocab size {args.vocab_size}...")
    
    # Count total files to process
    total_files = 0
    for folder in args.corpus_folders:
        txt_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
        total_files += len(txt_files)
    
    print(f"Found {total_files} text files for training")
    
    # Define special tokens
    special_tokens = ["<unk>", "<pad>", "<eos>"]
    
    # Train the tokenizer
    tokenizer = BPETokenizer.train_tokenizer(
        corpus_folders=args.corpus_folders,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens
    )
    
    # Save the tokenizer
    vocab_file, merges_file = tokenizer.save_vocabulary(args.output_dir)
    print(f"Tokenizer saved to {args.output_dir}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Test the tokenizer with test text
    if args.test_text:
        print("\nTesting tokenizer:")
        print(f"Original text: {args.test_text}")
        
        # Tokenize
        tokens = tokenizer._tokenize(args.test_text)
        print(f"Tokens: {tokens}")
        
        # Encode
        encoding = tokenizer.encode(args.test_text)
        token_ids = encoding["input_ids"]
        print(f"Token IDs: {token_ids}")
        
        # Decode
        decoded = tokenizer.decode(token_ids)
        print(f"Decoded text: {decoded}")
        


if __name__ == "__main__":
    main()

