import os
import re
from typing import List
from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import TemplateProcessing
from transformers import BatchEncoding
import unicodedata

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
    digits = frozenset('1234567890')
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
    # print(texts)
    return texts

def create_wordpiece_tokenizer(train_folders: list, val_folder: str, save_dir: str, 
                              vocab_size: int = 10000, min_frequency: int = 2):
    # # Load texts
    train_texts = []
    for train_folder in train_folders:
        print(train_folder)
        print("Loading texts from folders...")
        train_text = load_text_data(train_folder)
        lang = 'hi' if 'hindi-wiki' in train_folder else 'mr' if 'marathi-wiki' in train_folder else 'en'
        train_texts.extend([clean_text(i, lang) for i in train_text])
    # val_texts = load_text_data(val_folder)
    all_texts = train_texts
    # print("all_texts: ", all_texts)
    # all_texts = []

    print("Loading English SQuAD...")
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad")
    all_texts.extend(ds['train']['context'])

    # Create temporary file for training
    os.makedirs(save_dir, exist_ok=True)
    temp_file = os.path.join(save_dir, "temp_corpus.txt")
    
    print("Writing corpus to temporary file...")
    with open(temp_file, 'w', encoding='utf-8') as f:
        for text in tqdm(all_texts):
            cleaned_text = clean_text(text, lang='en')
            if cleaned_text:
                f.write(cleaned_text + '\n')
    
    # Initialize tokenizer
    print(f"Training WordPiece tokenizer with vocab size {vocab_size}...")
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,
    )
    
    # Train tokenizer
    tokenizer.train(
        files=[temp_file],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        limit_alphabet=2000,
        wordpieces_prefix="##",
    )
    
    # Save tokenizer
    model_prefix = os.path.join(save_dir, "wordpiece")
    tokenizer.save(f"{model_prefix}.json")
    
    # Clean up temporary file
    os.remove(temp_file)
    
    print(f"Tokenizer saved to {model_prefix}.json")
    
    return tokenizer

class WordPieceTokenizer:
    def __init__(self, tokenizer_path):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.eos_token = "[SEP]"  # Using SEP as EOS token
        
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.unk_token_id = self.tokenizer.token_to_id("[UNK]")
        self.cls_token_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_token_id = self.tokenizer.token_to_id("[SEP]")
        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")
        self.eos_token_id = self.tokenizer.token_to_id("[SEP]")
        
        self.all_special_tokens = [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token]
        self.all_special_ids = [self.pad_token_id, self.unk_token_id, self.cls_token_id, self.sep_token_id, self.mask_token_id]
    
    def __len__(self):
        return self.tokenizer.get_vocab_size()
    
    def get_vocab(self):
        return self.tokenizer.get_vocab()
    
    def encode(self, text, text_pair=None, add_special_tokens=True, max_length=None, 
               padding=False, truncation=False, return_tensors=None, **kwargs):
        import torch
        
        if isinstance(text, list):
            batch_encoding = {
                "input_ids": [],
                "attention_mask": []
            }
            
            for t in text:
                encoding = self.encode(t, text_pair, add_special_tokens, max_length, 
                                      padding, truncation, None, **kwargs)
                batch_encoding["input_ids"].append(encoding["input_ids"])
                batch_encoding["attention_mask"].append(encoding["attention_mask"])
            
            if return_tensors == "pt":
                batch_encoding = {k: torch.tensor(v) for k, v in batch_encoding.items()}
            
            return batch_encoding
        
        encoding = self.tokenizer.encode(text)
        input_ids = encoding.ids
        
        if add_special_tokens:
            if self.cls_token_id is not None and input_ids[0] != self.cls_token_id:
                input_ids = [self.cls_token_id] + input_ids
            if self.sep_token_id is not None and input_ids[-1] != self.sep_token_id:
                input_ids = input_ids + [self.sep_token_id]
        
        if truncation and max_length and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        attention_mask = [1] * len(input_ids)
        if padding and max_length:
            pad_length = max_length - len(input_ids)
            if pad_length > 0:
                input_ids = input_ids + [self.pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        if return_tensors == "pt":
            result = {k: torch.tensor([v]) for k, v in result.items()}
        
        return BatchEncoding(result)
    
    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        import torch
        
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() > 1:
                token_ids = token_ids.squeeze(0)
            token_ids = token_ids.tolist()
        elif not isinstance(token_ids, (list, tuple)):
            token_ids = [token_ids]
        
        if skip_special_tokens:
            token_ids = [id for id in token_ids if id not in self.all_special_ids]
        
        text = self.tokenizer.decode(token_ids)
        
        if clean_up_tokenization_spaces:
            text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        encoding = self.tokenizer.encode(text)
        return encoding.tokens
    
    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.tokenizer.token_to_id(tokens)
        return [self.tokenizer.token_to_id(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.tokenizer.id_to_token(ids)
        return [self.tokenizer.id_to_token(id) for id in ids]
    
    def prepare_for_model(
        self,
        ids,
        pair_ids=None,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        max_length=None,
        stride=0,
        pad_to_multiple_of=None,
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=True,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        return_length=False,
        verbose=True,
        prepend_batch_axis=False,
        **kwargs
    ):
        import torch
        
        if add_special_tokens:
            if self.cls_token_id is not None and (len(ids) == 0 or ids[0] != self.cls_token_id):
                ids = [self.cls_token_id] + ids
            if self.sep_token_id is not None and (len(ids) == 0 or ids[-1] != self.sep_token_id):
                ids = ids + [self.sep_token_id]
        
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]

        padding_side = kwargs.get("padding_side", "right")
        if padding and max_length and len(ids) < max_length:
            pad_length = max_length - len(ids)
            if padding_side == "right":
                ids = ids + [self.pad_token_id] * pad_length
            else:
                ids = [self.pad_token_id] * pad_length + ids

        encoded = {"input_ids": ids}
        
        if return_attention_mask:
            attention_mask = [1] * len(ids)
            if padding and max_length and len(attention_mask) < max_length:
                pad_length = max_length - len(attention_mask)
                if padding_side == "right":
                    attention_mask = attention_mask + [0] * pad_length
                else:
                    attention_mask = [0] * pad_length + attention_mask
            encoded["attention_mask"] = attention_mask

        if return_tensors == 'pt':
            if prepend_batch_axis:
                encoded = {k: torch.tensor([v]) for k, v in encoded.items()}
            else:
                encoded = {k: torch.tensor(v) for k, v in encoded.items()}

        return encoded
    
    def __call__(self, text, text_pair=None, add_special_tokens=True, padding=False, 
                truncation=False, max_length=None, return_tensors=None, **kwargs):
        return self.encode(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            **kwargs
        )

if __name__ == "__main__":
    train_folder = ["./datasets/hindi-wiki/train/train","./datasets/hindi-wiki-ipa/train","./datasets/marathi-wiki/train/train","./datasets/marathi-wiki-ipa/train"]  # folder containing train txt files
    # train_folder = ["datasets/hindi-wiki/train/train"]  # folder containing train txt files
    val_folder = "hindi-wiki/valid/valid"    # folder containing val txt files
    save_dir = "./all_tokenizer_files/wordpiece_tokenizer_files_ehm_ipa"
    
    # Create and train tokenizer
    tokenizer = create_wordpiece_tokenizer(
        train_folders=train_folder,
        val_folder=val_folder,
        save_dir=save_dir,
        vocab_size=17411,  # Vocabulary size
        min_frequency=1    # Minimum frequency for tokens
    )
    
    wp_tokenizer = WordPieceTokenizer(os.path.join(save_dir, "wordpiece.json"))
    
    # Test tokenization
    list_text = ["शिक्षा द्वारा राष्ट्रों, जातियों अथवा घार्मिक समूहों के बीच आपसी सद्भावना, सहिष्णुता और मंत्री का विकास होगा",
                "मराठी साहित्याचे विविध प्रकार म्हणजेच कथा, कादंबरी, ललित लेख, प्रवास वर्णनं, समीक्षा लेखन, नाटक वगैरे",
                "This sentence is a sample sentence.",
                "kut͡ʃʰ s̪ɑːl pəɦleː t̪ʰɑː 2012", "d͡ʒʱoːl", "ʊ t̪ s ʊ k t̪ ɑː', 'ʊ t̪ . s ʊ k . t̪ ɑː"]
    for i in list_text:
        encoded = wp_tokenizer(i)
        print(encoded)
        print(wp_tokenizer.decode(encoded["input_ids"]))

    # encoded = wp_tokenizer(test_text)
    # decoded = wp_tokenizer.decode(encoded["input_ids"])
    # print(f"\nTest tokenization:")
    # print(f"Original: {test_text}")
    # print(f"Encoded: {encoded}")
    # print(f"Decoded: {decoded}")