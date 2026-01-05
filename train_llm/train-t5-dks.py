import json
import pandas as pd
import torch
import torch.nn.functional as F
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
# from training_tokenizers.gc_tokenizer_hf import DevanagariTokenizer, load_vocabulary
# tokenizer = DevanagariTokenizer(vocab_dict=load_vocabulary(vocab_file_path="./all_tokenizer_files/gc_tokenizer_files_ehm/vocab.json"))
# model_name = "gc_dks"

# from training_tokenizers.sent_tok import SentencePieceTokenizer
# tokenizer = SentencePieceTokenizer("./all_tokenizer_files/sentencepiece_tokenizer_files_ehm/sentencepiece.json")
# model_name = "sp_dks"

# from training_tokenizers.bpe_tokenizer_hf import BPETokenizer
# tokenizer = BPETokenizer(vocab_file="./all_tokenizer_files/bpe_tokenizer_files_ehm/vocab.json", merges_file="./all_tokenizer_files/bpe_tokenizer_files_ehm/merges.json")
# model_name = "bpe_dks"

# from training_tokenizers.gbpe_tokenizer_hf import *
# tokenizer = GBPETokenizer(vocab_file="./all_tokenizer_files/gbpe_tokenizer_files_ehm/vocab.json", merges_file="./all_tokenizer_files/gbpe_tokenizer_files_ehm/merges.json")
# model_name = "gbpe_dks"

from training_tokenizers.wp_tok import WordPieceTokenizer
tokenizer = WordPieceTokenizer(tokenizer_path="./all_tokenizer_files/wordpiece_tokenizer_files_ehm/wordpiece.json")
model_name = "wp_dks"

def load_dks_data(filename):
    print(f'Loading G2P dataset from {filename}...')
    df = pd.read_csv(filename, header=None, keep_default_na=False)
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
        self.cache = {}
    
    def __len__(self):
        return len(self.orig)

    def __getitem__(self, idx):
        if self.cache_tokenization and idx in self.cache:
            return self.cache[idx]
            
        # format input as "O-Mr: <original text> T:"
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

# load G2P datasets
train_file = "./datasets/dks_hm_train.csv"
val_file = "./datasets/dks_hm_val.csv"

train_orig, train_trans = load_dks_data(train_file)
val_orig, val_trans = load_dks_data(val_file)

# create datasets
train_dataset = DKSDataset(train_orig, train_trans, tokenizer)
val_dataset = DKSDataset(val_orig, val_trans, tokenizer)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=10, pin_memory=True, prefetch_factor=10, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=128, num_workers=5, pin_memory=True)

# model config
config = T5Config(
    vocab_size=len(tokenizer),
    d_model=512,
    d_kv=32,
    d_ff=2048,
    num_layers=4,
    num_heads=8,
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

torch.manual_seed(42)
torch.cuda.manual_seed(42)

model = T5ForConditionalGeneration(config)

device = torch.device('cuda:2')
model.to(device)

model.gradient_checkpointing_enable()

# adafactor optim
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

# init GradScaler for AMP
scaler = GradScaler(enabled=False)

num_epochs = 15
best_val_loss = float('inf')
eval_every = 100
save_steps = 100
global_step = 0
checkpoint_dir = f"./chkpts/chkpts_dks/{model_name}"

os.makedirs(checkpoint_dir, exist_ok=True)

# ff resuming from a checkpoint
resume_from_checkpoint = False
if resume_from_checkpoint:
    checkpoint = torch.load(f'./chkpts/chkpts_v5/{model_name}/model_step_7799.pt')
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

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch+1}")):
        # skip steps already completed
        if((step_counter<=global_step and epoch<=global_step//len(train_loader) and epoch==0 )or (step_counter<=global_step-((epoch)*len(train_loader)) and epoch<=global_step//len(train_loader))):
          step_counter += 1
          continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # forward pass
        with autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            # logits = outputs.logits
            # loss = focal_loss(
            #     logits, 
            #     labels, 
            #     alpha=1.0,      
            #     gamma=2.0,      
            #     ignore_index=tokenizer.pad_token_id
            # )


        # backward pass
        scaler.scale(loss).backward()
        
        # uppdate weights
        scaler.step(optimizer)
        scaler.update()

        # total_loss += loss.item()
        total_loss = torch.add(total_loss, loss.detach(), alpha=1.0)
        step_counter += 1
        global_step += 1

        # eval
        if step_counter % eval_every == 0:
            avg_train_loss = total_loss / eval_every
            print(f"Step {step_counter}, Average Training Loss: {avg_train_loss:.4f}")
            total_loss = 0

            model.eval()

            # sample 5 random examples from val set
            random_idxs = random.sample(range(len(val_dataset)), 5)
            random_samples = [val_dataset[i] for i in random_idxs]
            for i, sample in enumerate(random_samples):
                input_ids = sample["input_ids"].unsqueeze(0).to(device)
                attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
                labels = sample["labels"].unsqueeze(0).to(device)

                # predictions
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
                print(f"Ground Truth Answer:\t {target_text}")
                print(f"Predicted Answer:\t {pred_text}")
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
            
            executor.submit(async_save_checkpoint, checkpoint_path, checkpoint_dict)

    # val loop
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
                # logits = outputs.logits
                # loss = focal_loss(
                #     logits, 
                #     labels, 
                #     alpha=1.0, 
                #     gamma=2.0, 
                #     ignore_index=tokenizer.pad_token_id
                # )

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Last Average Training Loss: {avg_train_loss:.4f}")

    # save best model based on validation loss
    if avg_val_loss <= best_val_loss:
        best_val_loss = avg_val_loss
        print(f"New best validation loss: {best_val_loss:.4f}. Saving best model...")

        model.save_pretrained(f"./models/models_v5/t5-dks-{model_name}-best")
        
        # save a checkpoint of the best model
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

    # save model after training
    model.save_pretrained(f"./models/models_v5/t5-dks-model-{model_name}-{epoch+1}")
    
    # clear CUDA cache after each epoch
    torch.cuda.empty_cache()

executor.shutdown()
print("Training completed!")
