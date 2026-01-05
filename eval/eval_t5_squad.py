import json
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

# Import torch.cuda.amp for mixed precision training instead of Apex
from torch.amp import autocast, GradScaler

# Basic CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# Custom dataset loader
def load_squad_hindi(filename):
    print(f'loading dataset from {filename}...')
    with open(filename, 'r', encoding='utf-8') as f:
        squad_dict = json.load(f)
    contexts = []
    questions = []
    answers = []

    for group in tqdm(squad_dict['data']):
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                if qa['answers'][0]['text'] != '':
                    question = qa['question']
                    answer =  qa['answers'][0]['text']
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    return contexts, questions, answers

def load_maha_squad(filename):
    print(f'loading dataset from {filename}...')
    with open(filename, 'r', encoding='utf-8') as f:
        squad_dict = json.load(f)
    contexts = []
    questions = []
    answers = []

    for entry in tqdm(squad_dict['data']):
        context = entry['context']
        question = entry['question']
        
        # Skip if answers list is empty
        if not entry['answers']['text']:
            continue
            
        answer = entry['answers']['text'][0]
        if answer != '':
            contexts.append(context)
            questions.append(question)
            answers.append(answer)
    # print(len(contexts))
    return contexts, questions, answers

def process_passage(passage):
    context = passage['context']
    qa_list = []
    for qa in passage['qas']:
        if qa['answers'][0]['text'] != '':
            question = qa['question']
            answer = qa['answers'][0]['text']
            qa_list.append((context, question, answer))
    return qa_list

def load_squad_hindi_itrans(filename, save_file=None, num_processes=4):

    # Check if a preprocessed file exists
    if filename and os.path.exists(filename):
        print(f'Loading preprocessed data from {filename}')
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # print(len(data['contexts']), len(data['questions']), len(data['answers']))
        return data['contexts'], data['questions'], data['answers']

    filename = filename.replace("_translit","")
    with open(filename, 'r', encoding='utf-8') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []

    # Flatten the data structure
    all_passages = [passage for group in squad_dict['data'] for passage in group['paragraphs']]

    # Process passages in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_passage, all_passages), total=len(all_passages), desc="Processing data"))

    # Flatten results and separate into contexts, questions, and answers
    for result in results:
        for context, question, answer in result:
            contexts.append(context)
            questions.append(question)
            answers.append(answer)

    print(len(contexts), len(questions),len(answers))

    return contexts, questions, answers


# Dataset class for Hindi SQuAD
class HindiSQuAD(torch.utils.data.Dataset):
    def __init__(self, contexts, questions, answers, tokenizer, max_input_len=512, max_output_len=50, cache_tokenization=True):
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.cache_tokenization = cache_tokenization
        self.cache = {}  # Add caching
    
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        if self.cache_tokenization and idx in self.cache:
            return self.cache[idx]
            
        input_text = f"question: {self.questions[idx]}  context: {self.contexts[idx]}"
        target_text = self.answers[idx]

        inputs = self.tokenizer(input_text, max_length=self.max_input_len, padding="max_length", truncation=True, return_tensors="pt")
        targets = self.tokenizer(target_text, max_length=self.max_output_len, padding="max_length", truncation=True, return_tensors="pt")

        item = {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": targets.input_ids.squeeze(),
            "answers": target_text
        }
        
        if self.cache_tokenization:
            self.cache[idx] = item
            
        return item

# Load dataset
# train_file = "datasets/wiki_squad_hi_train_trunc.jsonl"
# val_file = "datasets/wiki_squad_hi_val_trunc.jsonl"

# train_contexts, train_questions, train_answers = load_squad_hindi(train_file)
# val_contexts, val_questions, val_answers = load_squad_hindi(val_file)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from collections import Counter
import numpy as np

def is_partial_match(s1, s2):
    s1, s2 = s1.replace('।','').replace('॥',''), s2.replace('।','').replace('॥','')
    return s1 in s2 or s2 in s1


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


def calculate_token_f1(prediction, ground_truth):

    if not prediction or not ground_truth:
        return 0.0
        
    prediction_tokens = tokenizer.tokenize(prediction)
    ground_truth_tokens = tokenizer.tokenize(ground_truth)
    
    # Get token counts
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    # If no tokens match, return 0
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

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

def test_model(model, tokenizer, test_file, device, batch_size=64, output_csv="results/results_gp_hi/results.csv", file_loading_function=load_squad_hindi):
    # Load test data
    # if('translit' in test_file):
    #     contexts, questions, answers = load_and_transliterate_squad_hindi(save_file=test_file, filename=test_file.replace("_translit",""))
    # else:
    contexts, questions, answers = file_loading_function(test_file)
    # _, questions, _ = file_loading_function(test_file.replace("_drop","").replace("_reps","").replace("jsonl","json"))

    # Prepare test dataset
    test_dataset = HindiSQuAD(
        contexts,
        questions,
        answers,
        tokenizer=tokenizer,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, prefetch_factor=10, persistent_workers=True)
    
    exact_matches = 0
    total_edit_distance = 0
    total_partial_match = 0
    total_token_f1 = 0
    total_rouge_l_f1 = 0
    total_rouge_l_recall = 0
    total_predictions = 0

    # torch.cuda.empty_cache()
    model.eval()

    with torch.no_grad(), open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['context', 'question', 'real_answer', 'pred_answer', 'exact_match', 'partial_match', 'norm_edit_score', 'ROUGE-L recall','Token F1'])
        
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            batch_answers = batch['answers']  # This should be a batch-sized list of answers
            # print("inputs: ",input_ids)

            outputs = model.generate(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        max_length=50,
                    )
            # print("outputs: ",outputs)
            predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            # Print example for first item in batch

            # if('translit' in test_file):
            #     print("\n")
            #     print("-"*150)
            #     print(f"Question: {transliterate.process('ITRANS','Devanagari',questions[batch_idx * batch_size])}")
            #     print(f"Context: {transliterate.process('ITRANS','Devanagari',contexts[batch_idx * batch_size])}")
            #     print(f"Predicted Answer: {predictions[0]}")
            #     print(f"Predicted Answer: {transliterate.process('ITRANS','Devanagari',predictions[0])}")
            #     print(f"Ground Truth: {transliterate.process('ITRANS','Devanagari',batch_answers[0])}")
            #     print(f"Norm ED: {normalized_edit_distance(predictions[0], batch_answers[0])}",
            #         f"Exact Match: {predictions[0]==batch_answers[0]}", 
            #         f"Partial Match: {is_partial_match(predictions[0], batch_answers[0])}",
            #         f"Token F1: {calculate_token_f1(predictions[0], batch_answers[0])}", 
            #         f"ROUGE-L recall: {rouge_l(predictions[0], batch_answers[0])['recall']}", 
            #         sep="\t")
                
            # else:
            #     print("\n")
            #     print("-"*150)
            #     print(f"Question: {questions[batch_idx * batch_size]}")
            #     print(f"Context: {contexts[batch_idx * batch_size]}")
            #     print(f"Predicted Answer: {predictions[0]}")
            #     print(f"Ground Truth: {batch_answers[0]}")
            #     print(f"Norm ED: {normalized_edit_distance(predictions[0], batch_answers[0])}",
            #         f"Exact Match: {predictions[0]==batch_answers[0]}", 
            #         f"Partial Match: {is_partial_match(predictions[0], batch_answers[0])}",
            #         f"Token F1: {calculate_token_f1(predictions[0], batch_answers[0])}", 
            #         f"ROUGE-L recall: {rouge_l(predictions[0], batch_answers[0])['recall']}", 
            #         sep="\t")

            # Process each prediction in the batch
            for idx in range(len(predictions)):
                pred = predictions[idx].replace(' ।','।')
                truth = batch_answers[idx]
                
                # Skip if we've reached the end of the dataset
                # if (batch_idx * batch_size + idx) >= len(contexts):
                #     continue

                # Exact Match
                if pred == truth:
                    exact_matches += 1

                # Normalized Edit Distance
                ned = normalized_edit_distance(pred, truth)
                total_edit_distance += ned

                # Partial Match
                partial_match = is_partial_match(pred, truth)
                total_partial_match += 1 if partial_match else 0
                
                # Token F1
                token_f1 = calculate_token_f1(pred, truth)
                total_token_f1 += token_f1
                
                # ROUGE-L
                rouge_scores = rouge_l(pred, truth)
                total_rouge_l_f1 += rouge_scores['f1']
                total_rouge_l_recall += rouge_scores['recall']

                # Increment total predictions
                total_predictions += 1

                # Get actual index for the current prediction
                actual_idx = batch_idx * batch_size + idx
                
                # Write to CSV
                # if batch_idx * batch_size <= 2000:
                # if('translit' in test_file):
                #     csv_writer.writerow([
                #         transliterate.process('ITRANS','Devanagari',contexts[actual_idx]),
                #         transliterate.process('ITRANS','Devanagari',questions[actual_idx]),
                #         transliterate.process('ITRANS','Devanagari',truth),
                #         transliterate.process('ITRANS','Devanagari',pred),
                #         int(pred == truth),
                #         int(partial_match),
                #         f"{ned:.4f}",
                #         f"{rouge_scores['recall']:.4f}",
                #         f"{token_f1:.4f}"
                #     ])

                # else:
                csv_writer.writerow([
                    contexts[actual_idx],
                    questions[actual_idx],
                    truth,
                    pred,
                    # transliterate.process('ITRANS','Devanagari',contexts[actual_idx]),
                    # transliterate.process('ITRANS','Devanagari',questions[actual_idx]),
                    # transliterate.process('ITRANS','Devanagari',truth),
                    # transliterate.process('ITRANS','Devanagari',pred),
                    int(pred == truth),
                    int(partial_match),
                    f"{ned:.4f}",
                    f"{rouge_scores['recall']:.4f}",
                    f"{token_f1:.4f}"
                ])

                # else:
                #     break

            # if batch_idx % 10 == 0:  # Print progress every 10 batches
            #     print(f"Processed predictions so far: {total_predictions}")

    # Calculate final metrics
    exact_match_accuracy = exact_matches / total_predictions
    average_edit_distance = total_edit_distance / total_predictions
    average_partial_match = total_partial_match / total_predictions
    average_token_f1 = total_token_f1 / total_predictions
    average_rouge_l_f1 = total_rouge_l_f1 / total_predictions
    average_rouge_l_recall = total_rouge_l_recall / total_predictions
    print(total_predictions)

    print(f"\nFinal Metrics:")
    print(f"Total predictions processed: {total_predictions}")
    print(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")
    print(f"Average Normalized Edit Distance: {average_edit_distance:.4f}")
    print(f"Average Partial Match Accuracy: {average_partial_match:.4f}")
    print(f"Average Token F1: {average_token_f1:.4f}")
    print(f"Average ROUGE-L F1: {average_rouge_l_f1:.4f}")

    print(f"File saved to {output_csv}")

    return {
        "exact_match": exact_match_accuracy,
        "edit_distance": average_edit_distance,
        "partial_match": average_partial_match,
        "token_f1": average_token_f1,
        "rouge_l_f1": average_rouge_l_f1,
        "rouge_l_recall": average_rouge_l_recall
    }

# Usage
# tok_name = 'gc'
# for tok_name in ['itrans']:
# for tok_name in ['itrans']:
# for tok_name in ['sp']:
# for tok_name in ['sp', 'bpe']:
    # doc = 'seen'
type_doc = ""
# for tok_name in ['wp','sp','bpe']:
# for tok_name in ['itrans']:
# for tok_name in ['gc','wp','sp','bpe']:
# for tok_name in ['gc','sp','wp','bpe','gbpe']:
for tok_name in ['wp']:
    if(tok_name == 'gc'):
        from training_tokenizers.gc_tokenizer_hf import DevanagariTokenizer, load_vocabulary
        # tokenizer = DevanagariTokenizer(vocab_dict=load_vocabulary(vocab_file_path="vocab_gc.json"))
        tokenizer = DevanagariTokenizer(vocab_dict=load_vocabulary(vocab_file_path="all_tokenizer_files/gc_tokenizer_files/vocab.json"))
    elif(tok_name == 'gbpe'):
        from training_tokenizers.gbpe_tokenizer_hf import *
        tokenizer = GBPETokenizer(vocab_file="all_tokenizer_files/gbpe_tokenizer_files_ehm/vocab.json", merges_file="all_tokenizer_files/gbpe_tokenizer_files_ehm/merges.json")
    elif(tok_name=='bpe'):
        from training_tokenizers.bpe_tokenizer_hf import *
        tokenizer = BPETokenizer(vocab_file="all_tokenizer_files/bpe_tokenizer_files_ehm/vocab.json", merges_file="all_tokenizer_files/bpe_tokenizer_files_ehm/merges.json")
    elif(tok_name == 'wp'):
        # Initialize BERT (wp) tokenizer 
        # tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        from training_tokenizers.wp_tok import WordPieceTokenizer
        tokenizer = WordPieceTokenizer(tokenizer_path="all_tokenizer_files/wordpiece_tokenizer_files/wordpiece.json")
    elif(tok_name=='sp'): 
        from training_tokenizers.sent_tok import SentencePieceTokenizer
        tokenizer = SentencePieceTokenizer("all_tokenizer_files/sentencepiece_tokenizer_files_en/sentencepiece.json")
    elif(tok_name == 'gpt' or 'itrans' in tok_name):
        # # Initialize GPT-2 tokenizer (compatible with T5 model)
        # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # tokenizer.pad_token = tokenizer.eos_token  # Ensure padding tokens are set properly
        from training_tokenizers.itrans_tokenizer_hf import *
        # transliterate_input = False if lang == 'en' else True
        tokenizer = BPETokenizer(vocab_file="all_tokenizer_files/itrans_tokenizer_files_ehm/vocab.json", merges_file="all_tokenizer_files/itrans_tokenizer_files_ehm/merges.json", transliterate_input=True)
    else:
        tokenizer = None

    lang = ''
    file_loading_function = load_squad_hindi

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # model_path = f"models/models_v5/t5-squad-model-{tok_name}_{lang}-best"
    # model_path = f"models/models_v5/t5-squad-model-{tok_name}-best"
    model_path = f"models/models_v5/t5-squad-model-{tok_name}_{type_doc}_{lang}-best" if (type_doc!= '' and lang!= 'hi') else f"models/models_v5/t5-squad-model-{tok_name}_{lang}-best" if lang != 'hi' else f"models/models_v5/t5-squad-model-{tok_name}_{type_doc}-best" if type_doc != '' else f"models/models_v5/t5-squad-model-{tok_name}-best"
    # model = T5ForConditionalGeneration.from_pretrained(f"models_v4/t5-hindi-squad-model-{tok_name}_ehm-5")
    # model = T5ForConditionalGeneration.from_pretrained(f"models_v4/t5-marathi-squad-model-{tok_name}_optim_mr-6")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    print(f"Model config vocab size {model.config.vocab_size}")
    print(f"Model Path: {model_path}")

    model.to(device)

    model.gradient_checkpointing_enable()

    # for doc in ['unseen']:
    # for doc in ['seen','unseen']:
    for doc in ['xquad']:
    # for doc in ['indicqa']:
    # for doc in ['tydiqa']:
    # for doc in ['mahasquad']:
        # test_file = "datasets/new_clean/wiki_squad_hi_unseen_contexts_trunc_translit.jsonl"
    # test_file = "datasets/new_clean/wiki_squad_hi_seen_contexts_trunc_translit.jsonl"
        # test_file = f"datasets/wiki_squad_{lang}_{doc}_contexts_{type_doc}.jsonl" if lang!= 'hi' else f"datasets/new_clean/wiki_squad_hi_{doc}_contexts_trunc_{type_doc}.jsonl" if type_doc != "" else f"datasets/new_clean/wiki_squad_hi_{doc}_contexts_trunc.jsonl"
        if(doc=='xquad'):
            # test_file = f"datasets/xquad.hi_{type_doc}.jsonl"
            test_file = f"datasets/xquad.hi.json" if type_doc == "" else f"datasets/xquad.hi_{type_doc}.jsonl"
            # test_file = f"datasets/xquad.hi_reps.jsonl"
            # test_file = f"datasets/xquad.hi_translit.jsonl"
            # test_file = f"datasets/xquad.hi_{type_doc}_translit.jsonl"
            # test_file = f"datasets/xquad.hi_drop_translit.jsonl"
            # file_loading_function = load_squad_hindi_itrans
        if(doc=='indicqa'):
            test_file = "datasets/indicqa.hi.json"
        if(doc=='tydiqa'):
            test_file = "datasets/stdised/tydiqa-goldp-v1.1-dev/tydiqa-goldp-dev-english.json"
            # test_file = "datasets/stdised/tydiqa-goldp-v1.1-dev/tydiqa-goldp-dev-english_drop.jsonl"
        if(doc=='mahasquad'):
            # test_file = f"datasets/stdised/maha_squad_test_{type_doc}.jsonl"
            test_file = f"datasets/stdised/maha_squad_test_{type_doc}.jsonl" if type_doc != "" else f"datasets/stdised/maha_squad_test.json"
            # test_file = f"./datasets/stdised/maha_squad_test_translit.jsonl"
            # test_file = f"datasets/stdised/maha_squad_test_reps.jsonl"
            file_loading_function = load_maha_squad
            # test_file = "datasets/indicqa.hi.json"
        # if('itrans' in tok_name):
        #     test_file = f"datasets/new_clean/wiki_squad_hi_{doc}_contexts_trunc_translit.jsonl"
        #     if(doc=='xquad'):
        #         test_file = "datasets/xquad.hi_translit.json"
        print(tok_name, doc, test_file)
        # type_doc = type_doc if type_doc != '' else 'plain'
        test_model(model, tokenizer, test_file, device, output_csv=f'results/results_v6/{type_doc}-{tok_name}-{doc}-results-{lang}.csv',batch_size=512, file_loading_function=file_loading_function)

    print(f"Done Results for {tok_name} - {doc}")