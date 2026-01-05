from transformers import PreTrainedTokenizer, BatchEncoding
from typing import List, Optional, Tuple, Dict, Union
import json
import torch
import os
import unicodedata
from collections import defaultdict
import re
from tqdm import tqdm
from collections import defaultdict, Counter
import unicodedata

# from .custom_tokenizers.gc_tokenizer import *

def get_gc_tokens(devanagari_string):

    # Pre-compile set lookups for better performance
    combining_marks = frozenset({
        '_','्', 'ँ', 'ं', 'ः', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॢ', 'ॣ', 'े', 'ै','ॆ', 'ो', 'ौ','ॊ', '़','ॅ','ॉ'
    })

    # digits = frozenset('०१२३४५६७८९০১২৩৪৫৬৭৮৯0123456789')
    digits = frozenset('०१२३४५६७८९0123456789')
    end_markers = frozenset(['्',])
    
    # Pre-allocate list with estimated capacity
    graphemes = []
    temp_grapheme = []
    
    # Replace spaces once instead of checking each time
    devanagari_string = devanagari_string.replace(" ", "_")
    
    for char in devanagari_string:
        if char not in combining_marks:
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
    
    # Process all cleanups in one pass
    if '\u200c' in devanagari_string or '\u200d' in devanagari_string or '\u200b' in devanagari_string:
        graphemes = [g.replace('\u200c', '').replace('\u200d', '').replace('\u200b', '') for g in graphemes]
    
    return graphemes


class GBPETokenizer(PreTrainedTokenizer):
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
        self.merges = {tuple(k.split('|')): v for k, v in merges_dict.items()} if merges_dict else {}
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
        self.name_or_path = "gbpe-tokenizer"

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary."""
        return self.vocab

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using GBPE with integrated space handling."""
        final_tokens = []
        words = text.split()
        
        for i, word in enumerate(words):
            if i < len(words) - 1:
                word = word + " "
                
            graphemes = get_gc_tokens(word)
            
            while True:
                pairs = []
                for j in range(len(graphemes) - 1):
                    pair = (graphemes[j], graphemes[j + 1])
                    if pair in self.merges:
                        pairs.append((j, pair))
                
                if not pairs:
                    break
                    
                j, pair = pairs[0]
                graphemes = (
                    graphemes[:j] +
                    [self.merges[pair]] +
                    graphemes[j + 2:]
                )
            
            final_tokens.extend(graphemes)
        
        return final_tokens
    
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

        # save vocab
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

        merges_dict = {f"{k[0]}|{k[1]}": v for k, v in self.merges.items()}
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
            merges_dict = {tuple(k.split('|')): v for k, v in merges_dict.items()}

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
        
        text = "".join(tokens)
        
        if clean_up_tokenization_spaces:
            text = ' '.join(text.split('_'))
        
        return text


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
        
        # to preserve word boundaries
        if truncation and max_length and len(ids) > max_length:
            # last underscore before max_length
            tokens = self._convert_id_to_token(ids[:max_length])
            last_underscore = -1
            for i in range(len(tokens)-1, -1, -1):
                if tokens[i] == "_":
                    last_underscore = i
                    break
            
            if last_underscore != -1:
                ids = self.convert_tokens_to_ids(tokens[:last_underscore])
            else:
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

def load_vocabulary(vocab_file_path: str, merges_file_path: str) -> Tuple[Dict[str, int], Dict[Tuple[str, str], str]]:
    with open(vocab_file_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    with open(merges_file_path, 'r', encoding='utf-8') as f:
        merges_dict = json.load(f)
        merges = {tuple(k.split('|')): v for k, v in merges_dict.items()}
    
    return vocab, merges

def load_text_data(folder_path: str) -> List[str]:
    print(f'Loading texts from {folder_path}...')
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if text:
                        texts.append(text)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    print(f"Loaded {len(texts)} texts from {folder_path}")
    return texts

def build_initial_vocabulary(train_folders: list, val_folder: str, min_freq: int = 2, max_initial_tokens: int = 2000) -> Tuple[Dict[str, int], Counter]:
    word_freq = Counter()

    # special tokens
    print("Building vocabulary with special tokens...")
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<eos>": 2,
    }
    # # # Load all texts
    train_texts = []
    for train_folder in train_folders:
        train_texts.extend(load_text_data(train_folder))
    # # val_texts = load_text_data(val_folder)
    # # all_texts = train_texts + val_texts

    all_texts = train_texts

    # all_texts = []
    print("Loading English SQuAD...")
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad")
    all_texts.extend(ds['train']['context'])

    # Collect all the grapheme clusters using the get_gc_tokens function
    print("Processing texts and counting tokens...")
    for i in tqdm(range(len(all_texts))):
        texts = clean_text(all_texts[i], lang='en').split(" ")
        for text in texts:
            for gc in get_gc_tokens(text+" "):
                word_freq[gc] += 1
    
    # Filter tokens by minimum frequency
    if min_freq >= 1:
        word_freq = Counter({token: count for token, count in word_freq.items() if count >= min_freq})
    
    # Limit to top N most common tokens (after special tokens)
    next_id = len(vocab)
    for token, _ in word_freq.most_common():
        if token not in vocab: 
            vocab[token] = next_id
            next_id += 1

    print(f"Initial vocabulary size: {len(vocab)}")
    return vocab, word_freq


def train_bpe(vocab: Dict[str, int], texts: List[str], num_merges: int) -> Dict[Tuple[str, str], str]:

    print("Training BPE merges...")
    
    merges = {}
    
    word_freqs = defaultdict(int)
    print("Counting word frequencies...")
    for text in tqdm(texts):
        words = text.split()
        for i, word in enumerate(words):
            # Add space to all words except the last one
            if i < len(words) - 1:
                word = word + " "
            word_freqs[word] += 1
    
    # Initialize statistics for pairs
    pairs = defaultdict(lambda: defaultdict(int))
    word_tokens = {}
    
    print("Getting initial pairs...")
    for word, freq in tqdm(word_freqs.items()):
        tokens = tuple(get_gc_tokens(word))
        word_tokens[word] = tokens
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair][word] = freq
    
    pair_freqs = {pair: sum(freq.values()) for pair, freq in pairs.items()}
    
    print("Starting merge operations...")
    for merge_num in tqdm(range(num_merges)):
        if not pair_freqs:
            print("No more pairs to merge")
            break
            
        best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
        merge_count = pair_freqs[best_pair]
        
        if best_pair in merges:
            del pair_freqs[best_pair]
            continue
            
        new_token = ''.join(best_pair)
        merges[best_pair] = new_token
        
        # add to vocab
        if new_token not in vocab:
            vocab[new_token] = len(vocab)
        
        del pair_freqs[best_pair]
        
        affected_words = list(pairs[best_pair].keys())
        for word in affected_words:
            freq = word_freqs[word]
            old_tokens = word_tokens[word]
            
            new_tokens = []
            i = 0
            while i < len(old_tokens):
                if i < len(old_tokens) - 1 and old_tokens[i] == best_pair[0] and old_tokens[i + 1] == best_pair[1]:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(old_tokens[i])
                    i += 1
            new_tokens = tuple(new_tokens)
            
            for i in range(len(old_tokens) - 1):
                pair = (old_tokens[i], old_tokens[i + 1])
                if word in pairs[pair]:
                    pairs[pair].pop(word)
                    pair_freqs[pair] = pair_freqs.get(pair, 0) - freq
                    if pair_freqs[pair] <= 0:
                        if pair in pair_freqs:
                            del pair_freqs[pair]
                        if pair in pairs:
                            del pairs[pair]
            
            for i in range(len(new_tokens) - 1):
                pair = (new_tokens[i], new_tokens[i + 1])
                if pair not in merges:
                    pairs.setdefault(pair, {})[word] = freq
                    pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
            
            word_tokens[word] = new_tokens
    
    return merges

def save_vocab_and_merges(vocab: Dict[str, int], merges: Dict[Tuple[str, str], str], 
                         save_dir: str, prefix: str = ""):
    os.makedirs(save_dir, exist_ok=True)
    
    # Save vocabulary
    vocab_file = os.path.join(save_dir, f"{prefix}vocab.json")
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # Save merges with string keys
    merges_file = os.path.join(save_dir, f"{prefix}merges.json")
    merges_dict = {f"{k[0]}|{k[1]}": v for k, v in merges.items()}
    with open(merges_file, 'w', encoding='utf-8') as f:
        json.dump(merges_dict, f, ensure_ascii=False, indent=2)
    
    print(f"Saved vocabulary ({len(vocab)} tokens) to {vocab_file}")
    print(f"Saved merges ({len(merges)} merges) to {merges_file}")

def ipa_tokenize(ipa_string):

    # Ranges that IPA characters usually belong to
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
    # remove non hindi bengali, non numeric characters
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

def create_gbpe_tokenizer(train_folders: list, val_folder: str, save_dir: str, 
                         min_freq: int = 1, max_initial_tokens: int = 2000, 
                         final_vocab_size: int = 7600):
    
    vocab, grapheme_freqs = build_initial_vocabulary(
        train_folders, val_folder, min_freq, max_initial_tokens
    )
    num_merges = final_vocab_size - len(vocab)
    
    print(f"Initial vocabulary size: {len(vocab)}")
    print(f"Number of merges to perform: {num_merges}")
    
    if num_merges <= 0:
        print("initial vocabulary exceeds target size")
        num_merges = 0
    
    # Load texts for BPE training
    print("Loading texts from folders...")
    train_texts = []
    for train_folder in train_folders:
        for filename in os.listdir(train_folder):
            if filename.endswith('.txt'):
                lang = 'hi' if 'hindi-wiki' in train_folder else 'mr' if 'marathi-wiki' in train_folder else 'en'
                with open(os.path.join(train_folder, filename), 'r', encoding='utf-8') as f:
                    train_texts.append(clean_text(f.read().strip(), lang))
                
    # # val_texts = []
    # # for filename in os.listdir(val_folder):
    # #     if filename.endswith('.txt'):
    # #         with open(os.path.join(val_folder, filename), 'r', encoding='utf-8') as f:
    # #             val_texts.append(f.read().strip())
    
    all_texts = train_texts

    # all_texts = []

    print("Loading English SQuAD...")
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad")
    all_texts.extend(ds['train']['context'])
    
    # Train BPE
    print(f"\nTraining BPE with {num_merges} merges...")
    merges = train_bpe(vocab, all_texts, num_merges)
    
    # Convert merges to string format
    merges_dict = {f"{k[0]}|{k[1]}": v for k, v in merges.items()}
    
    # Save files
    save_vocab_and_merges(vocab, merges, save_dir)
    
    print("\nVocabulary Statistics:")
    print(f"Total vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    return vocab, merges_dict


# Example usage
if __name__ == "__main__":

    train_folders = ["./datasets/hindi-wiki/train/train","./datasets/hindi-wiki-ipa/train","./datasets/marathi-wiki/train/train","./datasets/marathi-wiki-ipa/train"]  # folder containing train txt files
    val_folder = "./datasets/hindi-wiki/valid/valid"      # folder containing val txt files
    save_dir = "./all_tokenizer_files/gbpe_tokenizer_files_ehm_ipa"
    
    # Create vocabulary and merges
    vocab, merges = create_gbpe_tokenizer(
        train_folders=train_folders,
        val_folder=val_folder,
        save_dir=save_dir,
        min_freq=2,  # Minimum frequency for initial vocabulary
        max_initial_tokens=17412,  # Limit initial vocab to top 2000 graphemes
        final_vocab_size=17412     # Target total vocabulary size
    )
    
    # Create and test tokenizer
    tokenizer = GBPETokenizer(
        vocab_file=f"{save_dir}/vocab.json",
        merges_file=f"{save_dir}/merges.json",
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>"
    )

    # test_text = "This is a sample sentence for testing the GBPE tokenizer."
    # encoded = tokenizer(test_text)
    # decoded = tokenizer.decode(encoded["input_ids"])
    # print(f"\nTest tokenization:")
    # print(f"Original: {test_text}")
    # print(f"Encoded: {encoded}")
    # print(f"Decoded: {decoded}")
    list_text = ["शिक्षा द्वारा राष्ट्रों, जातियों अथवा घार्मिक समूहों के बीच आपसी सद्भावना, सहिष्णुता और मंत्री का विकास होगा",
                "मराठी साहित्याचे विविध प्रकार म्हणजेच कथा, कादंबरी, ललित लेख, प्रवास वर्णनं, समीक्षा लेखन, नाटक वगैरे",
                "This sentence is a sample sentence.",
                "kut͡ʃʰ s̪ɑːl pəɦleː t̪ʰɑː 2012", "d͡ʒʱoːl", "ʊ t̪ s ʊ k t̪ ɑː', 'ʊ t̪ . s ʊ k . t̪ ɑː"]
    for i in list_text:
        print(tokenizer.encode(i))
        print(tokenizer.decode(tokenizer(i)["input_ids"]))