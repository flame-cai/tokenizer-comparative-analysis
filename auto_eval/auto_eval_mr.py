from lss_similarity_mr import *
import re
import pandas as pd
import random
from collections import Counter
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import logging
import os

def setup_logging(tok_name, doc, lang, type, doc_type):
    # Create logs directory if it doesn't exist
    os.makedirs('logs_v4', exist_ok=True)

    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging
    log_file = f"logs_v4/log_{tok_name}_{doc}_{lang}_type{type}_{doc_type}_v2.txt"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# for each question in the df, find the type of question, 'कोणत्या, कोण, केव्हा, कुठे, किती, काय, कसा/कसे, का, etc'
def find_question_type(question):
    # question_type = 'unknown'
    if 'केव्हा' in question or 'कधी' in question or 'कोणत्या वर्ष' in question or 'कोणत्या काळ' in question or 'कोणत्या वेळ' in question or 'कोणत्या महिन्य' in question or 'कोणत्या दशक' in question:
        question_type = 2
    elif('कुठे' in question or 'कोठे' in question or 'कुठं' in question or 'कशाबद्दल' in question or 'कशावरून' in question or 'कुठून' in question 
        or 'कोणत्या जिल्ह्य' in question or 'कोणत्या देश' in question or 'कोणत्या राज्य' in question or 'कोणत्या प्रदेशात' in question or 'कोणत्या शहर' in question 
        or 'कोणत्या स्थान' in question or 'कोणत्या जिल्य' in question or 'कोणता देश' in question or 'कोणता राज्य' in question or 'कोणता प्रदेश' in question 
        or 'कोणता शहर' in question or 'कोणता स्थान' in question or 'कोणता जिल्य' in question or 'कोणता जिल्ह्य' in question or 'कोणता जिल्ह'  in question
        or 'स्थान कोणता' in question or 'राजधानी कोणती' in question):
        question_type = 3
    elif ' का ' in question or 'कारण' in question:
        question_type = 7
    elif 'कोणत्या' in question or 'कोण' in question or 'कोणी' in question or 'नाव सांग' in question or 'कुठल्या' in question or 'कुठला' in question or 'कशासाठी' in question or 'कशा ' in question:
        question_type = 1
    elif 'किती' in question or 'कितीवेळा' in question or 'कितव्या' in question or 'कितवा' in question or 'कितवी ' in question or 'कितवे ' in question:
        question_type = 4
    elif 'काय' in question or 'कशाने ' in question or 'कशावर' in question:
        question_type = 5
    elif 'कसा' in question or 'कसे' in question or 'कशी' in question or 'कशामुळे' in question:
        question_type = 6
    else:
        question_type = 0
    return question_type

# %%
# function to test out pos tagging of stanza using random user input words
def pos_tagging(sentence):
    doc = nlp(sentence)
    for sent in doc.sentences:
        for word in sent.words:
            logging.info(f"{word.text} ({word.lemma} - {word.lemma} - {nlp(word.lemma).sentences[0].words[0].upos}) - {word.upos} - {word.upos} - {word.feats}")


# a function to find if the sentence is incomplete
def is_incomplete(sentence, reference=""):
    # if ends with symbol (even after clean_and_split)
    doc = nlp(sentence)
    for sent in doc.sentences:
        # logging.info(sent.words[-1].text)
        if sent.words[-1].upos == "SYM" or sent.words[-1].upos == "PUNCT" or sent.words[-1].text[-1] == '�' or sent.words[-1].text[-1] == '्':
            logging.info(f"incomplete Sentence: {sent}")
            return True
        if reference != "":
            last_word = sent.words[-1].text
            words2 = reference.strip().split()
            for word in words2:
                if word.startswith(last_word) and word != last_word:
                    return True
    # if ends with that question mark symbol �
    return False



# %%
def separate_attached_digits(text):
    # Insert space between digit and letter when directly adjacent
    text = re.sub(r'(?<=\d)(?=[^\d\s])', ' ', text)
    # Insert space between letter and digit when directly adjacent
    text = re.sub(r'(?<=[^\d\s])(?=\d)', ' ', text)
    return text

# clean text and change devanagari numbers to english numbers
def clean_text_and_split(text, split=True):
    text = re.sub(r"[।!,?.;’:\"'()\[\]{}<>।॥|\-_@%$]", " ", text)
    text = re.sub(r"[०१२३४५६७८९]", lambda x: str(int(x.group(0))), text)
    # text = text.replace("संवत्","संवत")
    text = separate_attached_digits(text)
    text = text + " "
    text = text.replace("ही ", " ही ")
    text = text.replace("ांचा "," चा ")
    text = text.replace("ींच्या "," च्या ")
    text = text.replace("चा "," चा ")
    text = text.replace("ेच्या "," च्या ")
    text = text.replace("च्या "," च्या ")
    text = text.replace("ावर ", " वर ")
    text = text.replace("ात ", " त ")
    text = text.replace('ीत  ', ' त ')
    text = text.replace("ेत ", " त ")
    text = text.replace("तर ", " तर ")
    text = text.replace("वर ", " वर ")
    text = text.replace("ानंतर ", " नंतर ")
    text = text.replace("ासाठी ", " साठी ")
    text = text.replace("साठी ", " साठी ")
    text = text.replace("ांनी ", " नी ")
    text = text.replace("ाने ", " ने ")
    text = text.replace("ामध्ये ", " मध्ये ")
    text = text.replace("मध्ये ", " मध्ये ")
    text = text.replace("सोबत ", " सोबत ")
    text = text.replace("ला ", " ला ")
    text = text.replace("ाचे ", " चे ")
    text = text.replace("ांचे ", " चे ")
    text = text.replace("चे ", " चे ")
    text = text.replace("ापासून ", " पासून ")
    text = text.replace("पासून ", " पासून ")
    text = text.replace("ांपासून ", " पासून ")
    text = text.replace("महाराष्ट्र ", "महाराष्ट्रा")
    text = text.replace("ामधील ", " मधील ")
    text = text.replace("मधील ", " मधील ")
    text = text.replace("समोर ", " समोर ")
    text = text.replace("कडे ", " कडे ")
    text = text.replace("पेक्षा ", " पेक्षा ")
    
    text = text.rstrip()

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)
    if split:
        text = text.split()
    return text

# code to check repetition in a sentence: डी अॅलनचे मुख्य कार्य अभिनेते, विनोदवीर, हजरजबाबी, लेखक, नाटककार, चित्रपट निर्माते, दिग्दर्शक, जाझ संगीतकार, विनोदवी
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def is_repeating(text: str, min_len: int = 3) -> bool:
    tokens = tokenize(text)
    n = len(tokens)
    min_len = 5

    # Not enough tokens to repeat
    if n < 8:
        return False

    # # 1) Check adjacent repetition
    # # Look for any block of length L that repeats immediately after itself
    # for length in range(min_len, n // 2 + 1):
    #     for i in range(n - 2 * length + 1):
    #         if tuple(tokens[i:i + length]) == tuple(tokens[i + length:i + 2 * length]):
    #             logging.info(f"Repeating Text: {text}")
    #             return True

    # 2) Check suffix repetition
    # Look for any trailing n-gram of length >= min_len that appears earlier
    for length in range(min_len, n):
        suffix = tuple(tokens[-length:])
        for i in range(n - length):
            if tuple(tokens[i:i + length]) == suffix:
                logging.info(f"Repeating Text: {text}")
                return True

    return False


def extract_date_components(text_list):
    """Extract date components (years, months, days, times) from a list of tokens."""
    # Known Hindi month names and their standardized forms
    hindi_months = {
        'जानेवारी': 'january', 'फेब्रुवारी': 'february', 'मार्च': 'march',
        'एप्रिल': 'april', 'मे': 'may', 'जून': 'june',
        'जुलै': 'july', 'ऑगस्ट': 'august', 'सप्टेंबर': 'september',
        'ऑक्टोबर': 'october', 'नोव्हेंबर': 'november', 'डिसेंबर': 'december'
    }
    
    date_info = {
        'years': set(),
        'months': set(),
        'days': set(),
        'times': set()
    }
    
    # Pattern for 4-digit years
    year_pattern = re.compile(r'^(1\d{3}|20[0-2]\d)$')
    
    # Pattern for times (like 345 बजे)
    time_pattern = re.compile(r'^(\d+)$')

    # logging.info("Text list: ",text_list)
    text_list = [i.rstrip('व्या').rstrip('च्या') if i.isdigit() else i for i in text_list]
    # logging.info(text_list)

    for i, token in enumerate(text_list):
        # Extract years (4-digit numbers that look like years)
        if year_pattern.match(token):
            date_info['years'].add(token)
        
        # Extract months (Hindi month names)
        if token in hindi_months:
            date_info['months'].add(hindi_months[token])
        
        # Extract days (1-31 followed by specific words)
        if token.isdigit() and 1 <= int(token) <= 31:
            if i < len(text_list) - 1 and (text_list[i+1] in hindi_months):
                date_info['days'].add(token)
        
        # Extract times
        if time_pattern.match(token) and i < len(text_list) - 1 and ('वाजता' in text_list[i+1] or 'वर्षी' in text_list[i+1] or 'साली' in text_list[i+1]):
            date_info['times'].add(token)
    
    return date_info

# date based answers
def check_date(question,pred_answer, gold_answer):
    if is_repeating(pred_answer):
        logging.info("Repeating")
        return gold_answer, pred_answer, 0
    # extract the numbers from gold_answer
    gold_answer = clean_text_and_split(gold_answer)
    # logging.info(gold_answer)
    gold_answer_list = [word.rstrip('व्या').rstrip('च्या').lstrip('स').replace('/','') for word in gold_answer if word.rstrip('व्या').isdigit() or word.rstrip('च्या').replace("/","").isdigit() or word.lstrip('स').isdigit()]
    # extract the numbers from pred_answer
    pred_answer = clean_text_and_split(pred_answer)
    # logging.info(f"Pred answer: {pred_answer}")
    pred_answer_list = [word.rstrip('व्या').rstrip('च्या').lstrip('स').replace('/','') for word in pred_answer if word.rstrip('व्या').isdigit() or word.rstrip('च्या').replace("/","").isdigit() or word.lstrip('स').isdigit()]
    # logging.info("pred answer list:")
    # logging.info(pred_answer_list)
    # check if the numbers are in the pred_answer

    if gold_answer_list == pred_answer_list:
        return gold_answer, pred_answer, 1
    
    # if is_incomplete(" ".join(pred_answer), " ".join(gold_answer)) or is_incomplete(" ".join(pred_answer), question):
    #     logging.info("isincomplete")
    #     return gold_answer, pred_answer, 0

    # extract the date components from the answers
    gold_date_info = extract_date_components(gold_answer)
    pred_date_info = extract_date_components(pred_answer)
    # logging.info(gold_date_info, pred_date_info, sep='\n')
    
    # check if year in real_answer matches with year in pred_answer
    # there might be multiple years in real_answer
    # account for multiple years

    for component in ['years', 'months', 'days', 'times']:
        if gold_date_info[component].intersection(pred_date_info[component]):
            
            match_ratio = len(gold_date_info[component].intersection(pred_date_info[component])) / len(gold_date_info[component]) if len(gold_date_info[component]) > 0 else 0
            # logging.info(component, match_ratio) if match_ratio == 0 else None
            if match_ratio < 0.75:
                logging.info(f"{question}\n{gold_answer}\n{pred_answer}\n{'-'*50}")
                logging.info(match_ratio)
                return question, gold_answer, pred_answer, 0
            else:
                return question, gold_answer, pred_answer, 1
        else:
            logging.info(f"{question}\n{gold_answer}\n{pred_answer}\n{'-'*50}")
            return question, gold_answer, pred_answer, 0
            
    # if none of the above conditions return a false, then return true
    return gold_answer, pred_answer, 1


# %%
# get the lemma of a word
def get_lemma_word_mapping(text_set):
    lemma_to_word = {}
    for word in text_set:
        doc = nlp(word)
        lemma = doc.sentences[0].words[0].lemma
        lemma_to_word[lemma] = word
    return lemma_to_word

# extract what's being asked based on the word following the question word 'किस' , 'कौन' , 'किन' 
def extract_relevant_info(question, sentence, custom_anchor=None):
    # --- Step 1: Extract the anchor from the question (unless a custom anchor is provided) ---
    extracted_tokens = []
    
    if not custom_anchor:
        doc_q = nlp(question)
        for sent_q in doc_q.sentences:
            # logging.info(sent_q.words)
            for i, word in enumerate(sent_q.words):
                if word.upos == 'DET':  # Identify 'किस', 'कौन', 'किन'
                    j = i + 1
                    # Skip tokens until a noun/proper noun is found
                    while j < len(sent_q.words) and sent_q.words[j].upos not in ['NOUN', 'PROPN']:
                        j += 1
                    if j < len(sent_q.words) and sent_q.words[j].upos in ['NOUN', 'PROPN']:
                        extracted_tokens.append(sent_q.words[j].text)
                        j += 1
                        # Collect additional contiguous noun/proper noun tokens, if any.
                        while j < len(sent_q.words) and sent_q.words[j].upos in ['NOUN', 'PROPN']:
                            extracted_tokens.append(sent_q.words[j].text)
                            j += 1
                    # If no tokens found, try reverse search
                    if not extracted_tokens:
                        for sent_q in doc_q.sentences:
                            for i, word in enumerate(sent_q.words):
                                if word.upos == 'DET':
                                    j = i - 1
                                    while j >= 0 and sent_q.words[j].upos not in ['NOUN', 'PROPN']:
                                        j -= 1
                                    if j >= 0 and sent_q.words[j].upos in ['NOUN', 'PROPN']:
                                        extracted_tokens.append(sent_q.words[j].text)
                                        j -= 1
                                        while j >= 0 and sent_q.words[j].upos in ['NOUN', 'PROPN']:
                                            extracted_tokens.append(sent_q.words[j].text)
                                            j -= 1
                                        extracted_tokens = extracted_tokens[::-1]
        # Using set() here to avoid duplicates (adjust if order matters)
        extracted_text = " ".join(set(extracted_tokens))
    else:
        extracted_text = custom_anchor
    # logging.info((f"Extracted text: {extracted_text}"))
    # --- Step 2: Extract the relevant noun phrase from the sentence ---
    allowed_nouns = {"NN", "NNC", "NNP", "NNPC", "NOUN", "PROPN", "NUM", "ADV"}
    # Fillers allowed (ADP/PSP) – in forward/backward search, we allow fillers until the first noun cluster is found.
    filler_limit = {"ADP", "PSP", "DET","QF", "QO", "ADJ"}
    # CC is allowed throughout a noun cluster.
    allowed_cc = {"CC", "QO"}

    doc_s = nlp(sentence)
    proper_noun_phrase_tokens = []
    anchor = extracted_text if extracted_text else None
    # logging.info(f"Anchor: {anchor}")

    found_anchor = False
    sentence_tokens = None
    anchor_idx = None

    # Look for the anchor in the sentence.
    for sent_s in doc_s.sentences:
        tokens = sent_s.words
        for idx, token in enumerate(tokens):
            if token.text == anchor and token.upos in allowed_nouns:
                anchor_idx = idx
                sentence_tokens = tokens
                found_anchor = True
                break
        if found_anchor:
            break

    # If we found the anchor, try first the backward (before anchor) extraction.
    if found_anchor and anchor_idx is not None:
        # Case A: Look backward (before anchor)
        if anchor_idx > 0 and sentence_tokens[anchor_idx - 1].upos in allowed_nouns:
            # Immediate noun cluster exists: collect contiguous noun tokens going backward.
            collected = []
            i = anchor_idx - 1
            while i >= 0 and sentence_tokens[i].upos in allowed_nouns:
                collected.append(sentence_tokens[i])
                i -= 1
            collected = list(reversed(collected))
        else:
            # Case B: No immediate noun before anchor. Apply filler logic backward:
            # First, skip any ADP/PSP fillers (allowing multiple fillers before any noun is found).
            i = anchor_idx - 1
            while i >= 0 and sentence_tokens[i].upos in filler_limit:
                i -= 1
            # Now, if a noun is found, collect the noun cluster (allowing CC between nouns)
            if i >= 0 and sentence_tokens[i].upos in allowed_nouns:
                collected = []
                while i >= 0 and (sentence_tokens[i].upos in allowed_nouns or sentence_tokens[i].upos in allowed_cc):
                    collected.append(sentence_tokens[i])
                    i -= 1
                collected = list(reversed(collected))
            else:
                collected = []
        proper_noun_phrase_tokens = [tok for tok in collected if tok.upos in allowed_nouns]
        # logging.info(f"Proper Noun Phrases: {proper_noun_phrase_tokens}")

        # --- Fallback: If no noun tokens were found backward, try scanning forward ---
        if not proper_noun_phrase_tokens:
            # Forward extraction: starting right after the anchor.
            i = anchor_idx + 1
            # First, skip fillers if any (ADP/PSP allowed in sequence until first noun)
            while i < len(sentence_tokens) and sentence_tokens[i].upos in filler_limit:
                i += 1
            # Now, if a noun is found, collect the noun cluster (allow CC between nouns)
            if i < len(sentence_tokens) and sentence_tokens[i].upos in allowed_nouns:
                collected_forward = []
                while i < len(sentence_tokens) and (sentence_tokens[i].upos in allowed_nouns or sentence_tokens[i].upos in allowed_cc):
                    collected_forward.append(sentence_tokens[i])
                    i += 1
                proper_noun_phrase_tokens = [tok for tok in collected_forward if tok.upos in allowed_nouns]
            # Else, leave proper_noun_phrase_tokens empty.
    # --- Edge Case: If no anchor is found, extract the first contiguous noun group in the sentence ---
    else:
        for sent_s in doc_s.sentences:
            # logging.info(f"Sent S: {sent_s}")
            tokens = sent_s.words
            collected = []
            i = 0
            psp_count = 0

            # logging.info(f"Token Length: {len(tokens)}")
            # logging.info("\nStarting token search:")
            # Find first noun
            while i < len(tokens) and tokens[i].upos not in allowed_nouns:
                # logging.info(f"Skipping token: {tokens[i].text} ({tokens[i].upos})")
                i += 1
                
            # logging.info("\nStarting noun cluster collection:")
            while i < len(tokens):
                # logging.info(f"Checking token: {tokens[i].text} ({tokens[i].upos})")
                if tokens[i].upos in allowed_nouns:
                    # logging.info(f"Adding noun: {tokens[i].text} ({tokens[i].upos})")
                    collected.append(tokens[i])
                    i += 1
                elif tokens[i].upos in filler_limit:
                    psp_count += 1
                    # logging.info(f"Found PSP ({psp_count}): {tokens[i].text}")
                    if psp_count > 2:
                        # logging.info(f"Breaking at third PSP: {tokens[i].text}")
                        break
                    collected.append(tokens[i])
                    i += 1
                elif tokens[i].upos in {*filler_limit, *allowed_cc}:
                    # logging.info(f"Found filler: {tokens[i].text} ({tokens[i].upos})")
                    collected.append(tokens[i])
                    i += 1
                else:
                    # logging.info(f"Breaking at: {tokens[i].text} ({tokens[i].upos})")
                    break
                    
            # logging.info("\nFinal collected tokens:")
            # for token in collected:
            #     logging.info(f"{token.text} ({token.upos})")
                
            proper_noun_phrase_tokens = [tok for tok in collected if tok.upos in allowed_nouns]
            break

    proper_noun_phrase_text = " ".join(tok.text for tok in proper_noun_phrase_tokens)
    return extracted_text, set(proper_noun_phrase_text.split())


def check_which_questions(question, pred_answer, gold_answer):
    if is_repeating(pred_answer):
        logging.info("Repeating")
        return gold_answer, pred_answer, 0
    question = clean_text_and_split(question, split=False)
    gold_answer = clean_text_and_split(gold_answer, split=False)
    anchor, gold_info = extract_relevant_info(question, gold_answer)

    pred_answer = clean_text_and_split(pred_answer, split=False)
    _, pred_info = extract_relevant_info(question, pred_answer, anchor)

    # logging.info(f"Gold Info: {gold_info}")
    # logging.info(f"Pred Info: {pred_info}")

    if gold_answer == pred_answer:
        return gold_answer, pred_answer, 1
    
    if gold_answer in pred_answer:
        return gold_answer, pred_answer, 1
    
    # if is_incomplete(pred_answer, gold_answer) or is_incomplete(pred_answer, question):
    #     return gold_answer, pred_answer, 0

    # do a match_ratio based comparison
    
    # Create mappings
    question_mapping = get_lemma_word_mapping(set(question.split()))
    gold_mapping = get_lemma_word_mapping(gold_info)
    pred_mapping = get_lemma_word_mapping(pred_info)
    # logging.info(f"Gold Mapping: {gold_mapping}")
    # logging.info(f"Pred Mapping: {pred_mapping}")

    # Get differences using lemmas
    gold_lemma_diff = set(gold_mapping.keys()).difference(set(question_mapping.keys()))
    pred_lemma_diff = set(pred_mapping.keys()).difference(set(question_mapping.keys()))

    # Convert back to original words
    gold_info_diff = {gold_mapping[lemma] for lemma in gold_lemma_diff}
    pred_info_diff = {pred_mapping[lemma] for lemma in pred_lemma_diff}


    if gold_info_diff.intersection(pred_info):
        # logging.info(gold_info)
        match_ratio = len(gold_info_diff.intersection(pred_info_diff)) / len(gold_info_diff) if len(gold_info_diff) > 0 else 0
        if match_ratio < 0.67:
            logging.info(f"Question: {question}")
            logging.info(f"Anchor: {anchor}")
            logging.info(f"Gold Answer: {gold_answer}")
            logging.info(f"Gold Difference: {gold_info_diff}")
            logging.info(f"Pred Answer: {pred_answer}")
            logging.info(f"Pred Difference: {pred_info_diff}")
            logging.info("_"*100)
            # logging.info(question, anchor, gold_answer, gold_info, pred_answer, pred_info, "_"*100, sep="\n")
            return gold_answer, pred_answer, 0
        else:
            return gold_answer, pred_answer, 1
    else:
        logging.info(f"{question}\n{anchor}\n{gold_answer}\n{gold_info}\n{pred_answer}\n{pred_info}\n{'_'*100}")
        return gold_answer, pred_answer, 0

# %%
# Marathi number word mappings
UNITS = {
    'शून्य': 0, 'एक': 1, 'दोन': 2, 'तीन': 3, 'चार': 4,
    'पाच': 5, 'सहा': 6, 'सात': 7, 'आठ': 8, 'नऊ': 9,
    'दहा': 10, 'अकरा': 11, 'बारा': 12, 'तेरा': 13, 'चौदा': 14,
    'पंधरा': 15, 'सोळा': 16, 'सतरा': 17, 'अठरा': 18, 'एकोणीस': 19,
    'एका': 1, 'दोन्': 2
}
TENS = {
    'वीस': 20, 'तीस': 30, 'चाळीस': 40, 'पन्नास': 50,
    'साठ': 60, 'सत्तर': 70, 'ऐंशी': 80, 'नव्वद': 90
}
SPECIALS = {
    'दोनशे': 200, 'तीनशे': 300, 'चारशे': 400, 'पाचशे': 500,
    'सहाशे': 600, 'सातशे': 700, 'आठशे': 800, 'नऊशे': 900,
    'हजार': 1000, 'लाख': 100000, 'कोटी': 10000000, 'शंभर': 100,
}
FRACTIONS = {
    'दीड': 1.5, 'अडीच': 2.5
}

# Indefinite quantifiers (no fixed numeric value)
QUANTIFIERS = {'अनेक', 'काही', 'बरेच', 'जास्त', 'किती', 'थोडे', 'अधिक', 'सर्वांत'}


# Helper: merge sequences of space-separated digits into single numbers
def merge_spaced_digits(text):
    # Finds groups like '1 9 9 1' and collapses them to '1991'
    return re.sub(r'((?:\d\s+)+\d)', lambda m: m.group(1).replace(' ', ''), text)

def separate_attached_digits(text):
    # Add space between a digit and a non-digit boundary in both directions
    return re.sub(r'(?<=\d)(?=\D)|(?<=\D)(?=\d)', ' ', text)

# Parse a single Marathi number word into a numeric value (int/float)
def parse_marathi_number(word):
    # Normalize plurals (e.g., हजारो, हजारों)
    word = re.sub(r'(ो|ों)$', '', word)
    # Strip emphasis/case suffixes for units
    word = re.sub(r'(चा|च्या|ची|च|दा)$', '', word)
    if word in FRACTIONS:
        return FRACTIONS[word]
    if word in UNITS:
        return UNITS[word]
    if word in TENS:
        return TENS[word]
    if word in SPECIALS:
        return SPECIALS[word]
    # Compound tens + units (e.g., 'पंचेचाळीस') splits at 'े'
    for ten_word, ten_val in TENS.items():
        if word.endswith(ten_word):
            prefix = word[:-len(ten_word)].rstrip('े')
            if prefix in UNITS:
                return ten_val + UNITS[prefix]
    return None

# check if a word's upos is NUM and xpos is QC, and return those words as a set
def is_count(text):
    text = merge_spaced_digits(text)
    count_words = set()

    # If entire text is digits
    if text.isdigit():
        return {text}

    # Token-based extraction
    for token in text.split():
        # Skip indefinite quantifiers
        if token in QUANTIFIERS:
            continue
        # Direct digits
        if token.isdigit():
            count_words.add(token)
            continue
        # Marathi number words
        numeric = parse_marathi_number(token)
        if numeric is not None:
            num_str = str(int(numeric)) if numeric == int(numeric) else str(numeric)
            count_words.add(num_str)
            continue

    # If any numbers found, return
    if count_words:
        return count_words

    # NLP-based POS fallback
    doc = nlp(text)
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos == 'NUM':
                w = word.text
                # Skip quantifiers in POS fallback
                if w in QUANTIFIERS:
                    continue
                if w.isdigit():
                    count_words.add(w)
                else:
                    numeric = parse_marathi_number(w)
                    if numeric is not None:
                        num_str = str(int(numeric)) if numeric == int(numeric) else str(numeric)
                        count_words.add(num_str)
    return count_words

# Evaluation of count-based answers
def check_count(question, pred_answer, gold_answer):
    if is_repeating(pred_answer):
        logging.info("Repeating")
        return gold_answer, pred_answer, 0

    # extract numbers from the real answer
    gold_answer = clean_text_and_split(gold_answer, split=False)
    # create a golden answer set, if the numbers are of upos NUM and the xpos is QC
    gold_answer_set = is_count(gold_answer)

    pred_answer = clean_text_and_split(pred_answer, split=False)
    pred_answer_set = is_count(pred_answer)

    if gold_answer == pred_answer:
        return gold_answer, pred_answer, 1

    # logging.info(question, gold_answer, gold_answer_set, pred_answer, pred_answer_set, "_"*100, sep='\n')

    # check if the number of numbers in the pred_answer is equal to the number of numbers in the gold_answer
    # match_ratio based eval
    if gold_answer_set.intersection(pred_answer_set):
        match_ratio = (
            len(gold_answer_set.intersection(pred_answer_set)) /
            len(gold_answer_set)
        ) if len(gold_answer_set) != 0 else 0
        if match_ratio < 1:
            logging.info(f"{question}\n{gold_answer}\n{pred_answer}\n{'-'*50}")
            return question, gold_answer, pred_answer, 0
        else:
            return question, gold_answer, pred_answer, 1
    else:
        logging.info(f"{question}\n{gold_answer}\n{pred_answer}\n{'-'*50}")
        return question, gold_answer, pred_answer, 0
# %%
# extract the location based on a proper noun followed by a pnoun or just noun
def extract_location(text):
    doc = nlp(text)
    locations = set()
    filler_words = {"ADP", "PSP", "DET","QF", "QO", "ADJ"}
    
    for sent in doc.sentences:
        # logging.info(sent)
        current_sequence = []
        for i, token in enumerate(sent.words):
            if token.upos in ["PROPN", "NOUN"]:
                # logging.info(f"Noun appended: {token.text}")
                current_sequence.append(token.text)
            elif token.upos == "NOUN" and current_sequence:
                # Extend the current sequence with a noun following a PROPN
                current_sequence.append(token.text)
            elif token.upos in filler_words and current_sequence:
                continue
            else:
                if current_sequence:
                    # Add individual words from the current sequence into the set
                    for word in current_sequence:
                        locations.add(word)
                    # logging.info(f"Current Sequence Emptied: {current_sequence}")
                    current_sequence = []
        if current_sequence:
            for word in current_sequence:
                locations.add(word)
        # logging.info(current_sequence)
        # logging.info(locations)
    
    return locations

# check whether the location is correct
def check_location(question, pred_answer, gold_answer):
    if is_repeating(pred_answer):
        logging.info("Repeating")
        return gold_answer, pred_answer, 0
    # extract location from the gold_answer
    gold_answer = clean_text_and_split(gold_answer, split=False)
    gold_answer_set = extract_location(gold_answer)

    # extract location from the pred_answer
    pred_answer = clean_text_and_split(pred_answer, split=False)
    pred_answer_set = extract_location(pred_answer)

    # logging.info(f"Gold Answer Set: {gold_answer_set}")
    # logging.info(f"Pred Answer Set: {pred_answer_set}")


    if gold_answer == pred_answer:
        return gold_answer, pred_answer, 1
    
    # Create mappings
    question_mapping = get_lemma_word_mapping(set(clean_text_and_split(question)))
    gold_mapping = get_lemma_word_mapping(gold_answer_set)
    pred_mapping = get_lemma_word_mapping(pred_answer_set)

    # logging.info(f"Gold Mapping: {gold_mapping}")
    # logging.info(f"Pred Mapping: {pred_mapping}")
    
    # Get differences using lemmas
    gold_lemma_diff = set(gold_mapping.keys()).difference(set(question_mapping.keys()))
    pred_lemma_diff = set(pred_mapping.keys()).difference(set(question_mapping.keys()))

    # Convert back to original words
    gold_answer_set_diff = {gold_mapping[lemma] for lemma in gold_lemma_diff}
    pred_answer_set_diff = {pred_mapping[lemma] for lemma in pred_lemma_diff}

    # logging.info(f"Gold Lemma Diff: {gold_answer_set_diff}")
    # logging.info(f"Pred Lemma Diff: {pred_answer_set_diff}")

    # check if the location is correct
    if gold_answer_set_diff.intersection(pred_answer_set_diff) or gold_answer_set.intersection(pred_answer_set):
        match_ratio = len(gold_answer_set_diff.intersection(pred_answer_set_diff)) / len(gold_answer_set_diff) if len(gold_answer_set_diff) > 0 else 0
        match_ratio = max(match_ratio,len(gold_answer_set.intersection(pred_answer_set)) / len(gold_answer_set)) if len(gold_answer_set) > 0 else 0
        logging.info(f"Match Ratio: {match_ratio}")
        if match_ratio < 0.55:
            logging.info(f"{question}\n{gold_answer}\n{gold_answer_set}\n{pred_answer}\n{pred_answer_set}\n{'-'*100}")
            return gold_answer_set, pred_answer_set, 0
        else:
            # logging.info("True condition",question, gold_answer_set, pred_answer_set, "-"*100,sep="\n")   
            return gold_answer_set, pred_answer_set, 1
    else:
        logging.info(f"Last Else Condition: {question}\n{gold_answer}\n{gold_answer_set}\n{pred_answer}\n{pred_answer_set}\n{'-'*100}")
        return gold_answer_set, pred_answer_set, 0

# %%

def is_partial_match(s1, s2):
    if s1 == '' or s2 == '':
        return 0
    return 1 if s1 in s2 or s2 in s1 else 0

def edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def normalized_edit_distance(s1, s2):
    if len(s1) == 0 and len(s2) == 0:
        return 0
    return edit_distance(s1, s2) / max(len(s1), len(s2))

def lcs_length(X, Y):
    m, n = len(X), len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    
    return L[m][n]

def rouge_l(prediction, ground_truth):
    if not prediction or not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    pred_tokens = list(prediction)
    truth_tokens = list(ground_truth)
    
    lcs = lcs_length(pred_tokens, truth_tokens)
    
    if lcs == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    precision = lcs / len(pred_tokens)
    recall = lcs / len(truth_tokens)
    
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def three_level_similarity(gold_answer, pred_answer):
    sem_sim = sentence_semantic_similarity(gold_answer, pred_answer)
    lex_sim = lexical_similarity(gold_answer, pred_answer)
    sen_sim = syntactic_similarity(gold_answer, pred_answer)

    if lex_sim == 0:
        return 0
    elif sem_sim == 0:
        return 0
    elif sen_sim == 0:
        return 0
    elif sem_sim > 0.5 and lex_sim > 0.5 and sen_sim > 0.5:
        return 1
    else:
        return 0

def check_answer(question, pred_answer, gold_answer, context):
    if is_repeating(pred_answer):
        logging.info("Repeating")
        return gold_answer, pred_answer, 0
    # if is_incomplete(pred_answer.replace(".",""), gold_answer) or is_incomplete(pred_answer.replace(".",""), question) or is_incomplete(pred_answer.replace(".",""), context):
    #     logging.info("Incomplete")
    #     return gold_answer, pred_answer, 0
    
    # clean the text
    gold_answer = clean_text_and_split(gold_answer, split=False)
    pred_answer = clean_text_and_split(pred_answer, split=False)
    context = clean_text_and_split(context, split=False)

    if gold_answer == pred_answer:
        return gold_answer, pred_answer, 1
    
    # Create mappings
    # Create mappings
    question_mapping = get_lemma_word_mapping(question.split())
    gold_mapping = get_lemma_word_mapping(gold_answer.split())
    pred_mapping = get_lemma_word_mapping(pred_answer.split())

    # Get differences using lemmas while preserving order
    gold_lemma_diff = [lemma for lemma in gold_mapping.keys() 
                    if lemma not in question_mapping.keys()]
    pred_lemma_diff = [lemma for lemma in pred_mapping.keys() 
                    if lemma not in question_mapping.keys()]

    # Convert back to original words while maintaining order
    gold_answer = ' '.join([gold_mapping[lemma] for lemma in gold_lemma_diff])
    pred_answer = ' '.join([pred_mapping[lemma] for lemma in pred_lemma_diff])

    
    # check if the answer is in the question
    if is_partial_match(gold_answer, pred_answer):
        # logging.info(question, gold_answer, pred_answer, "_"*100, sep="\n")
        return gold_answer, pred_answer, 1
    elif rouge_l(pred_answer, gold_answer)["recall"] > 0.8:
        # logging.info(question, gold_answer, pred_answer, "_"*100, sep="\n")
        return gold_answer, pred_answer, 1
    elif three_level_similarity(gold_answer, pred_answer):
        # logging.info(question, gold_answer, pred_answer, "_"*100, sep="\n")
        return gold_answer, pred_answer, 1
    else:
        logging.info(f"False Case: {question}\n{gold_answer}\n{pred_answer}\n{'-'*50}")
        return gold_answer, pred_answer, 0
   

def eval(csv_file, type):

    # function to calculate and print the accuracy of auto_eval for each model in a df, columns: model, auto_eval == 1/0
    def print_acuracy(df):
        # check if model column exists in df
        if 'model' not in df.columns:
            auto_eval = df['auto_eval'].values
            accuracy = np.mean(auto_eval)
            logging.info(f"Accuracy: {accuracy}")
            print(f"Accuracy: {accuracy}")
        # for model in df['model'].unique():
        #     df_model = df[df['model'] == model]
        #     auto_eval = df_model['auto_eval'].values
        #     accuracy = np.mean(auto_eval)
        #     logging.info(f"Model: {model}, Accuracy: {accuracy}")

    def print_accuracy_eval(df):
        # check if model column exists in df
        if 'model' not in df.columns:
            auto_eval = df['auto_eval'].values
            accuracy = np.mean(auto_eval)
            logging.info(f"Accuracy: {accuracy}")
            print(f"Accuracy: {accuracy}")
        # for model in df['model'].unique():
        #     df_model = df[df['model'] == model]
        #     auto_eval = df_model['evaluation'].values
        #     accuracy = np.mean(auto_eval)
        #     logging.info(f"Model: {model}, Accuracy: {accuracy}")

    # load csv_file
    df = pd.read_csv(csv_file)
    df['question_type'] = df['question'].progress_apply(find_question_type)
    # convert all question, context, pred_answer and real_answer column to string
    df['question'] = df['question'].astype(str)
    df['context'] = df['context'].astype(str)
    df['pred_answer'] = df['pred_answer'].astype(str)
    df['real_answer'] = df['real_answer'].astype(str)


    # find for each row, find question_type from the df column filter out baed on type
    df = df[df['question_type'] == type]
    logging.info(f"Lenght of DF: {len(df)}")

    # 7 question_types
    
    # type 1: 'किस' , 'कौन' , 'किन' , नाम बता' 
    if type == 1:
        # random_indices = np.random.choice(len(df), 10, replace=False)
        # for i in random_indices:
        #     logging.info(df.iloc[i]['question'], df.iloc[i]['real_answer'], df.iloc[i]['pred_answer'], "-"*100, sep='\n')
        # pass
        df['auto_eval'] = df.progress_apply(lambda x: check_which_questions(x['question'],x['pred_answer'], x['real_answer'])[-1], axis=1)
        print_acuracy(df)
        print_accuracy_eval(df)
        # total_alignment = (df['auto_eval'] == df['evaluation']).mean()
        # logging.info(f"\nOverall alignment accuracy: {total_alignment:.3f}")
        return df

    # type 2: 'कब'
    elif type == 2:
        # check if the real_answer has a number in it, df.apply check_date
        df['auto_eval'] = df.progress_apply(lambda x: check_date(x['question'],x['pred_answer'], x['real_answer'])[-1], axis=1)
        print_acuracy(df)
        print_accuracy_eval(df)
        # total_alignment = (df['auto_eval'] == df['evaluation']).mean()
        # logging.info(f"\nOverall alignment accuracy: {total_alignment:.3f}")
        return df

    # type 3: 'कहाँ' , 'किधर' , 'कहां'
    elif type == 3:
        df['auto_eval'] = df.progress_apply(lambda x: check_location(x['question'],x['pred_answer'], x['real_answer'])[-1], axis=1)
        print_acuracy(df)
        print_accuracy_eval(df)
        # total_alignment = (df['auto_eval'] == df['evaluation']).mean()
        # logging.info(f"\nOverall alignment accuracy: {total_alignment:.3f}")
        return df

    # type 4: 'कितना' , 'कितने' , 'कितनी'
    elif type == 4:
        df['auto_eval'] = df.progress_apply(lambda x: check_count(x['question'],x['pred_answer'], x['real_answer'])[-1], axis=1)
        print_acuracy(df)
        print_accuracy_eval(df)
        # total_alignment = (df['auto_eval'] == df['evaluation']).mean()
        # logging.info(f"\nOverall alignment accuracy: {total_alignment:.3f}")
        return df

    # type 5: 'क्या'
    elif type == 5:
        # random_indices = np.random.choice(len(df), 20, replace=False)
        # for i in random_indices:
        #     logging.info(df.iloc[i]['question'], df.iloc[i]['real_answer'], df.iloc[i]['pred_answer'], "-"*100, sep='\n')
        df['auto_eval'] = df.progress_apply(lambda x: check_answer(x['question'],x['pred_answer'], x['real_answer'], x['context'])[-1], axis=1)
        print_acuracy(df)
        print_accuracy_eval(df)
        # total_alignment = (df['auto_eval'] == df['evaluation']).mean()
        # logging.info(f"\nOverall alignment accuracy: {total_alignment:.3f}")
        return df

    # type 6: 'कैसा' , 'कैसे' , 'कैसी'
    elif type == 6:

        # random_indices = np.random.choice(len(df), 15, replace=False)
        # for i in random_indices:
        #     logging.info(df.iloc[i]['question'], df.iloc[i]['real_answer'], df.iloc[i]['pred_answer'], "-"*100, sep='\n')
        # check using similarity, or if pred is a subset of real, or if real is a subset of pred
        df['auto_eval'] = df.progress_apply(lambda x: check_answer(str(x['question']),str(x['pred_answer']), str(x['real_answer']), str(x['context']))[-1], axis=1)
        print_acuracy(df)
        print_accuracy_eval(df)
        # total_alignment = (df['auto_eval'] == df['evaluation']).mean()
        # logging.info(f"\nOverall alignment accuracy: {total_alignment:.3f}")
        return df

    # type 7: 'क्यों' , 'कारण'
    elif type == 7:
        # random_indices = np.random.choice(len(df), 20, replace=False)
        # for i in random_indices:
        #     logging.info(df.iloc[i]['question'], df.iloc[i]['real_answer'], df.iloc[i]['pred_answer'], "-"*100, sep='\n')
        # similar to type 6 in evaluation
        df['auto_eval'] = df.progress_apply(lambda x: check_answer(x['question'],x['pred_answer'], x['real_answer'], x['context'])[-1], axis=1)
        print_acuracy(df) 
        print_accuracy_eval(df)
        # total_alignment = (df['auto_eval'] == df['evaluation']).mean()
        # logging.info(f"\nOverall alignment accuracy: {total_alignment:.3f}")
        return df

    # type 0: otherwise 
    else:
        pass

def eval_all_types(csv_file):
    # Load and process the CSV file once
    df = pd.read_csv(csv_file)
    df['question_type'] = df['question'].progress_apply(find_question_type)
    df = df[df['question_type'] != 0]

    # Truncate to at most 100 rows per question type
    question_types = sorted(df['question_type'].unique())
    print(f"Question Types: {question_types}")
    # show th eocunt for each type
    for type_num in question_types:
        print(f"Type {type_num}: {len(df[df['question_type'] == type_num])}")
    truncated_dfs = []
    
    for type_num in question_types:
        type_df = df[df['question_type'] == type_num]
        # if len(type_df) > 100:
        #     # Sample 100 rows randomly if more than 100 rows exist
        #     type_df = type_df.sample(n=100, random_state=42)
        truncated_dfs.append(type_df)
    
    # Combine the truncated dataframes
    df = pd.concat(truncated_dfs)

    
    # Evaluate each question type
    for type_num in range(1, 8):  # Types 1-7
        # Filter for current question type
        type_df = df[df['question_type'] == type_num]
        
        if len(type_df) == 0:
            continue
        
        df['pred_answer'] = df['pred_answer'].progress_apply(lambda x: str(x))
        # Apply appropriate evaluation function based on type
        if type_num == 1:
            df.loc[df['question_type'] == type_num, 'auto_eval'] = df[df['question_type'] == type_num].progress_apply(
                lambda x: check_which_questions(x['question'], x['pred_answer'], x['real_answer'])[-1], axis=1)
        elif type_num == 2:
            df.loc[df['question_type'] == type_num, 'auto_eval'] = df[df['question_type'] == type_num].progress_apply(
                lambda x: check_date(x['question'], x['pred_answer'], x['real_answer'])[-1], axis=1)
        elif type_num == 3:
            df.loc[df['question_type'] == type_num, 'auto_eval'] = df[df['question_type'] == type_num].progress_apply(
                lambda x: check_location(x['question'], x['pred_answer'], x['real_answer'])[-1], axis=1)
        elif type_num == 4:
            df.loc[df['question_type'] == type_num, 'auto_eval'] = df[df['question_type'] == type_num].progress_apply(
                lambda x: check_count(x['question'], x['pred_answer'], x['real_answer'])[-1], axis=1)
        else:  # Types 5, 6, 7
            df.loc[df['question_type'] == type_num, 'auto_eval'] = df[df['question_type'] == type_num].progress_apply(
                lambda x: check_answer(x['question'], x['pred_answer'], x['real_answer'], x['context'])[-1], axis=1)
    
    # Print overall results for each model
    logging.info(f"Overall Auto Accuracy: {df['auto_eval'].mean():.4f}")
    print(f"Overall Auto Accuracy: {df['auto_eval'].mean():.4f}")

    print("\nType-wise Accuracy:")
    for type_num in sorted(df['question_type'].unique()):
        type_df = df[df['question_type'] == type_num]
        accuracy = type_df['auto_eval'].mean()
        count = len(type_df)
        print(f"Type {type_num}: {accuracy:.4f} (n={count})")
        logging.info(f"Type {type_num}: {accuracy:.4f} (n={count})")
    # df.to_csv("./for_evaluators_v3/gc-amp-mahasquad-results-mr.csv", index=False)
    return df

if __name__ == "__main__":
    # command line arg parser for tok_name and type
    import argparse
    parser = argparse.ArgumentParser()
    # accepted values: gc, gbpe, wp, sp, itrans, bpe

    parser.add_argument("-t", "--tok_name", type=str, default="gc", help="Tokenizer name", choices=["gc", "gbpe", "wp", "sp", "itrans", "bpe"])
    parser.add_argument("--type", type=int, default=1, help="Question type", choices=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--eval_all", type=bool, default=False, help="Evaluate all types")
    args = parser.parse_args()
    # logging.info(args.tok_name, args.type)

    docs = ['mahasquad']
    type_doc = ''
    result_folder = './results/results_v6'

    for doc in docs:
        if args.eval_all:
            csv_file = f"./{result_folder}/{type_doc}-{args.tok_name}-{doc}-results-ehm.csv" if type_doc != "" else f"./{result_folder}/-{args.tok_name}-{doc}-results-ehm.csv"
            setup_logging(args.tok_name, doc, "mr", 'all', type_doc) 
            print(f"File: {csv_file}")
            print("Starting evaluation for all types...")
            logging.info(f"Evaluating {args.tok_name} on {doc} dataset for all types...")
            eval_all_types(csv_file)
        else:
            csv_file = f"./{result_folder}/{type_doc}-{args.tok_name}-{doc}-results-mr.csv" if type_doc != "" else f"./{result_folder}/-{args.tok_name}-{doc}-results-mr.csv"
            setup_logging(args.tok_name, doc, "mr", args.type, type_doc) 
            logging.info(f"Evaluating {args.tok_name} on {doc} dataset for {args.type} type questions...")
            eval(csv_file, args.type)

    # sample usage code: 
    # python auto_eval_mr.py --tok_name gpt2 --type 1
    # sample if you want to evaluate all types:
    # python auto_eval_mr.py --tok_name gpt2 --eval_all True