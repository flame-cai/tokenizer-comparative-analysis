from transformers import PreTrainedTokenizer, BatchEncoding
from typing import List, Optional, Tuple, Dict, Union
# from .custom_tokenizers.gc_tokenizer import *
import json
import re
from tqdm import tqdm
from collections import defaultdict, Counter
import torch
import os
import unicodedata

class DevanagariTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_dict, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>", **kwargs):
        self.vocab = vocab_dict
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        for special_token in [unk_token, pad_token]:
            if special_token not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[special_token] = new_id
                self.ids_to_tokens[new_id] = special_token

        super().__init__(unk_token=unk_token, pad_token=pad_token,eos_token=eos_token, **kwargs)

        self._vocab_size = len(self.vocab)

    def __call__(self, text_input, add_special_tokens=True, padding=False,
                 truncation=False, max_length=None, stride=0,
                 is_split_into_words=False, pad_to_multiple_of=None,
                 return_tensors=None, return_token_type_ids=None,
                 return_attention_mask=None, return_overflowing_tokens=False,
                 return_special_tokens_mask=False, return_offsets_mapping=False,
                 return_length=False, verbose=True, **kwargs):
        
        if isinstance(text_input, str):
            text_input = [text_input]
        
        encoded_inputs = []
        for text in text_input:
            tokens = self._tokenize(text)
            token_ids = self.convert_tokens_to_ids(tokens)
            # print(f"Line 50: {token_ids}")
            if add_special_tokens and self.eos_token_id is not None:
                token_ids.append(self.eos_token_id)
            encoded_inputs.append(token_ids)

        # Find max length for padding if not specified
        if padding == "max_length" and max_length is None:
            max_length = max(len(ids) for ids in encoded_inputs)

        if padding:
            for i in range(len(encoded_inputs)):
                if len(encoded_inputs[i]) < max_length:
                    pad_length = max_length - len(encoded_inputs[i])
                    encoded_inputs[i].extend([self.pad_token_id] * pad_length)
                elif truncation and len(encoded_inputs[i]) > max_length:
                    encoded_inputs[i] = encoded_inputs[i][:max_length]

        attention_masks = None
        if return_attention_mask or padding:
            attention_masks = [
                [1] * min(len(ids), max_length) + [0] * (max_length - len(ids)) if padding
                else [1] * len(ids)
                for ids in encoded_inputs
            ]

        if return_tensors == "pt":
            encoded_inputs = torch.tensor(encoded_inputs)
            if attention_masks is not None:
                attention_masks = torch.tensor(attention_masks)

        # for single sequence, remove the batch dimension
        if len(text_input) == 1:
            encoded_inputs = encoded_inputs.squeeze(0) if return_tensors == "pt" else encoded_inputs[0]
            if attention_masks is not None:
                attention_masks = attention_masks.squeeze(0) if return_tensors == "pt" else attention_masks[0]

        features = {
            "input_ids": encoded_inputs,
        }
        if attention_masks is not None:
            features["attention_mask"] = attention_masks

        return BatchEncoding(features, tensor_type="pt" if return_tensors == "pt" else None)

    def _tokenize(self, text: str) -> List[str]:

        # first get the initial grapheme clusters
        tokens = custom_tokenizer(text)
        final_tokens = []
        
        for token in tokens:
            if token in self.vocab:
                # ff token exists in vocab, keep it as is
                final_tokens.append(token)
            else:
                # find longest valid subsequences
                n = len(token)
                i = 0
                while i < n:
                    # find longest valid sequence starting at position i
                    found_valid = False
                    for j in range(n, i, -1):  # longest sequences first
                        substring = token[i:j]
                        if substring in self.vocab:
                            final_tokens.append(substring)
                            i = j  # move index past the found token
                            found_valid = True
                            break
                    
                    if not found_valid:
                        # if no valid sequence found starting at i, add UNK and move forward
                        final_tokens.append(self.unk_token)
                        i += 1
        
        return final_tokens
    
    def _convert_token_to_id(self, token: str) -> int:
        # if self.vocab.get(token) is None:
        #     with open('log.txt','a') as f:
        #         f.write(token)
        #         f.write("\n")
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)


    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        return token_ids_0

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False) -> List[int]:
        return [0] * len(token_ids_0)

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        return [0] * len(token_ids_0)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        import os
        import json

        if not os.path.isdir(save_directory):
            raise OSError(f"Vocabulary path ({save_directory}) should be a directory")

        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

        return (vocab_file,)

    def encode(self, text: str, max_length: Optional[int] = None, 
               truncation: bool = False, return_tensors: Optional[str] = None) -> Union[List[int], torch.Tensor]:
        full_text = text
        tokens = self._tokenize(full_text)
        token_ids = self.convert_tokens_to_ids(tokens) + [self.eos_token_id]
        # print(f"Line 154: {token_ids}")

        # truncation if specified
        if truncation and max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        # Convert to tensor if specified
        if return_tensors == 'pt':
            return torch.tensor(token_ids)
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True,
               clean_up_tokenization_spaces: bool = True) -> str:
        tokens = self.convert_ids_to_tokens(token_ids)
        # Remove special tokens if requested
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.all_special_tokens]
        return "".join(tokens).replace("_"," ")

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
    ) -> Dict[str, Union[List[int], List[List[int]], torch.Tensor]]:
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
            encoded["attention_mask"] = attention_mask

        if return_tensors == 'pt':
            encoded = {k: torch.tensor(v) for k, v in encoded.items()}

        return encoded

def load_vocabulary(vocab_file_path="vocab.json"):
    with open(vocab_file_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab

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


def build_initial_vocabulary(train_file_paths, val_file_path, min_freq=1, save_dir=None):
    # Initialize a Counter to count word frequencies
    from collections import Counter
    word_freq = Counter()

    # Define special tokens
    print("Building vocabulary with special tokens...")
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<eos>": 2,
    }

    print("Loading texts from folders...")
    train_texts = []
    for train_file_path in train_file_paths:
        lang = 'hi' if 'hindi-wiki' in train_file_path else 'mr' if 'marathi-wiki' in train_file_path else 'en'
        for filename in os.listdir(train_file_path):
            if filename.endswith('.txt'):
                with open(os.path.join(train_file_path, filename), 'r', encoding='utf-8') as f:
                    train_texts.append(clean_text(f.read().strip(), lang=lang))
                

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
                # print(gc)
                word_freq[gc] += 1
    
    # Filter tokens by minimum frequency if needed
    if min_freq > 1:
        word_freq = Counter({token: count for token, count in word_freq.items() if count >= min_freq})
    
    # Add tokens to vocabulary starting from index 3 (after special tokens)
    next_id = len(vocab)
    for token, _ in word_freq.most_common():
        if token not in vocab:  # Skip if token is already a special token
            vocab[token] = next_id
            next_id += 1
    
    # Print length of the vocab
    print(f"Final Vocab size: {len(vocab)}")
    
    # Create directory if it doesn't exist
    os.makedirs(f"{save_dir}", exist_ok=True)
    
    # Save it as a json file
    with open(f"{save_dir}/vocab.json", "w") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    
    return vocab

if __name__ == "__main__":

    from custom_tokenizers.gc_tokenizer import *
    custom_tokenizer = get_gc_tokens

    save_dir = "./all_tokenizer_files/gc_tokenizer_files_ehm_ipa"

    build_initial_vocabulary(train_file_paths=["./datasets/hindi-wiki/train/train","./datasets/hindi-wiki-ipa/train","./datasets/marathi-wiki/train/train","./datasets/marathi-wiki-ipa/train"], val_file_path="./datasets/hindi-wiki/valid/valid", min_freq=1, save_dir=save_dir)

    tokenizer = DevanagariTokenizer(load_vocabulary(vocab_file_path=f"{save_dir}/vocab.json"))
    list_text = ["शिक्षा द्वारा राष्ट्रों, जातियों अथवा घार्मिक समूहों के बीच आपसी सद्भावना, सहिष्णुता और मंत्री का विकास होगा",
                "मराठी साहित्याचे विविध प्रकार म्हणजेच कथा, कादंबरी, ललित लेख, प्रवास वर्णनं, समीक्षा लेखन, नाटक वगैरे",
                "This sentence is a sample sentence.",
                "kʊt͡ʃʰ sɑːl pəɦleː t̪ʰɑː 2012, ","d͡ʒʱoːl", "ʊ t̪ s ʊ k t̪ ɑː', 'ʊ t̪ . s ʊ k . t̪ ɑː",
                "राााााजाा",
                # "मुक्केबाज़ों ",
                "कोफ़ता"
                # "तेेेंदुुुुनी किस मण्डल के अन्तर्गत आता हहै?",
                "रुद्र कककांंंडााली कौन थे?"
                ]
    for i in list_text:
        print(tokenizer.encode(i))
        print([(tokenizer.decode(j),j) for j in tokenizer.encode(i)])

    # print(DevaIPAFix("d͡ʒjoːt̪ɪʂ ʃɑːs̪t̪ɾə keː mɑːd̪ʰjəm s̪eː puːɾeː ʋəɾʂ mẽː ɦoːn̪eː ʋɑːliː mukʰjə gʰəʈən̪ɑːõː kɑː ɑːklən̪ kɪjɑː d͡ʒɑːt̪ɑː ɦəɪ."))

else:
    from .custom_tokenizers.gc_tokenizer import *
    # Custom tokenizer
    custom_tokenizer = get_gc_tokens


