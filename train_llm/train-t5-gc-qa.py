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
import concurrent.futures
from torch.amp import autocast, GradScaler

# CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# init with a single worker thread
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

def async_save_checkpoint(checkpoint_path, checkpoint_dict):
    torch.save(checkpoint_dict, checkpoint_path)
    print(f"Checkpoint saved at step {checkpoint_dict['global_step']}.")

# Load Tokenizer
from training_tokenizers.gc_tokenizer_hf import DevanagariTokenizer, load_vocabulary
tokenizer = DevanagariTokenizer(vocab_dict=load_vocabulary(vocab_file_path="./all_tokenizer_files/gc_tokenizer_files_ehm/vocab.json"))
model_name = "gc_ehm"
print(f"Len: {len(tokenizer)}")

# from training_tokenizers.bpe_tokenizer_hf import BPETokenizer
# tokenizer = BPETokenizer(vocab_file="all_tokenizer_files/bpe_tokenizer_files_ehm/vocab.json", merges_file="all_tokenizer_files/bpe_tokenizer_files_ehm/merges.json")
# model_name = "bpe_ehm"
# print(f"Len: {len(tokenizer)}")

# from training_tokenizers.sent_tok import SentencePieceTokenizer
# tokenizer = SentencePieceTokenizer("all_tokenizer_files/sentencepiece_tokenizer_files_ehm/sentencepiece.json")
# model_name = "sp_ehm"
# print(f"Len: {len(tokenizer)}")

# from training_tokenizers.wp_tok import WordPieceTokenizer
# tokenizer = WordPieceTokenizer(tokenizer_path="all_tokenizer_files/wordpiece_tokenizer_files_ehm/wordpiece.json")
# model_name = "wp_ehm"

# from training_tokenizers.gbpe_tokenizer_hf import *
# tokenizer = GBPETokenizer(vocab_file="all_tokenizer_files/gbpe_tokenizer_files_ehm/vocab.json", merges_file="all_tokenizer_files/gbpe_tokenizer_files_ehm/merges.json")
# model_name = "gbpe_ehm"
# print(f"Len: {len(tokenizer)}")



# data loading and processing
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
                    answer = qa['answers'][0]['text']
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
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

# load datasets
# train_file = "datasets/wiki_squad_mr_train.json"
# val_file = "datasets/wiki_squad_mr_val.json"

train_file = "datasets/wiki_squad_hi_train_trunc.jsonl"
val_file = "datasets/wiki_squad_hi_val_trunc.jsonl"

train_contexts, train_questions, train_answers = load_squad_hindi(train_file)
val_contexts, val_questions, val_answers = load_squad_hindi(val_file)

train_file = "datasets/wiki_squad_mr_train.json"
val_file = "datasets/wiki_squad_mr_val.json"
train_contexts_mr, train_questions_mr, train_answers_mr = load_squad_hindi(train_file)
val_contexts_mr, val_questions_mr, val_answers_mr = load_squad_hindi(val_file)

from datasets import load_dataset

ds = load_dataset("rajpurkar/squad")
train_file = "datasets/wiki_squad_en_train.json"
train_contexts_en, train_questions_en, train_answers_en = load_squad_hindi(train_file)
val_contexts_en = list(ds['validation']['context'])
val_questions_en = list(ds['validation']['question'])
val_answers_en = [i['text'][0] for i in ds['validation']['answers']]

# combine datasets
train_contexts = train_contexts + train_contexts_mr + train_contexts_en
train_questions = train_questions + train_questions_mr + train_questions_en
train_answers = train_answers + train_answers_mr + train_answers_en
val_contexts = val_contexts + val_contexts_mr + val_contexts_en
val_questions = val_questions + val_questions_mr + val_questions_en
val_answers = val_answers + val_answers_mr + val_answers_en

# create datasets
train_dataset = HindiSQuAD(train_contexts, train_questions, train_answers, tokenizer)
val_dataset = HindiSQuAD(val_contexts, val_questions, val_answers, tokenizer)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=20, pin_memory=True, prefetch_factor=10, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=128, num_workers=10, pin_memory=True)

# model config
config = T5Config(
    vocab_size=len(tokenizer),
    d_model=512,
    d_kv=32,
    d_ff=2048,
    num_layers=6,
    num_heads=12,
    relative_attention_bias=False,
    dropout_rate=0.1,
    layer_norm_epsilon=1e-6,
    initializer_factor=1.0,
    feed_forward_proj="relu",
    is_encoder_decoder=True,
    use_cache=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=tokenizer.pad_token_id,
    tie_word_embeddings=False,
)
config.attention_implementation = "flash_attention_2"
model = T5ForConditionalGeneration(config)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model.to(device)


# gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# adafactor optimizer
optimizer = Adafactor(
    model.parameters(),
    lr=None,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.01,
    relative_step=True,
    scale_parameter=True,
    warmup_init=True
)

# GradScaler for AMP
scaler = GradScaler(enabled=False)

# training parameters
num_epochs = 12
best_val_loss = float('inf')
eval_every = 100
save_steps = 1000
global_step = 0
checkpoint_dir = f"./chkpts/chkpts_ehm/{model_name}"

os.makedirs(checkpoint_dir, exist_ok=True)

# if resuming from a checkpoint
resume_from_checkpoint = False
if resume_from_checkpoint:
    checkpoint = torch.load(f'chkpts_v4/{model_name}/model_step_6369.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    global_step = checkpoint['global_step']
    best_val_loss = checkpoint['loss']
    print(f"Resuming from step {global_step}, with best validation loss {best_val_loss:.4f}")

# training with AMP
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    step_counter = 0

    # skip epochs already completed
    if epoch < global_step // len(train_loader):
        continue

    print(f"Starting Epoch {epoch + 1}/{num_epochs}")

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch + 1}")):
        # skip steps already completed
        if((step_counter<=global_step and epoch<=global_step//len(train_loader) and epoch==0 )or (step_counter<=global_step-((epoch)*len(train_loader)) and epoch<=global_step//len(train_loader))):
          step_counter += 1
          continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # clear gradients
        optimizer.zero_grad()

        # forward pass
        with autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        # backward pass with scaler
        scaler.scale(loss).backward()
        
        # uppdate weights
        scaler.step(optimizer)
        scaler.update()

        # total_loss += loss.item()
        total_loss = torch.add(total_loss, loss.detach(), alpha=1.0)
        step_counter += 1
        global_step += 1

        # Evaluation step with random 5-sample printing
        if step_counter % eval_every == 0:
            avg_train_loss = total_loss / eval_every
            print(f"Step {step_counter}, Average Training Loss: {avg_train_loss:.4f}")
            total_loss = 0

            # evaluation 
            model.eval()

            # sample 5 random examples from val set
            random_idxs = random.sample(range(len(val_dataset)), 5)
            random_samples = [val_dataset[i] for i in random_idxs]
            for i, sample in enumerate(random_samples):
                input_ids = sample["input_ids"].unsqueeze(0).to(device)
                attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
                labels = sample["labels"].unsqueeze(0).to(device)

                # Generate predictions
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        max_length=64,
                        use_cache=True,
                        synced_gpus=False
                    )

                input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                target_text = tokenizer.decode(labels[0], skip_special_tokens=True)
                pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                print(f"\n--- Sample {i + 1} from Evaluation ---")
                print(f"Input Question and Context: {input_text}")
                print(f"Ground Truth Answer: {target_text}")
                print(f"Predicted Answer: {pred_text}")
                print("--- End of Sample ---\n")

            # end of eval
            model.train()

        # save checkpoint every save_steps steps
        if step_counter % save_steps == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{global_step}.pt")
            checkpoint_dict = {
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': best_val_loss,
            }
            
            # submit the checkpoint saving task to run asynchronously
            executor.submit(async_save_checkpoint, checkpoint_path, checkpoint_dict)

    # validation loop
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_loader)):
            with autocast('cuda',dtype=torch.bfloat16):
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device)
                )
                loss = outputs.loss
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch + 1}/{num_epochs}, Average Validation Loss: {avg_val_loss:.4f}")

    # save best model based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss

        print(f"New best validation loss: {best_val_loss:.4f}. Saving best model...")

        model.save_pretrained(f"./models/models_v5/t5-squad-model-{model_name}-best")
        
        # Optionally save a checkpoint of the best model too
        best_checkpoint_path = os.path.join(checkpoint_dir, f"best_model_step_{global_step}.pt")
        best_checkpoint_dict = {
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': best_val_loss,
            'epoch': epoch + 1
        }
        torch.save(best_checkpoint_dict, best_checkpoint_path)

    # save the model after training
    model.save_pretrained(f"./models/models_v5/t5-squad-model-{model_name}-{epoch+1}")
    
    # clear CUDA cache after each epoch
    torch.cuda.empty_cache()

executor.shutdown()
print("Training completed!")