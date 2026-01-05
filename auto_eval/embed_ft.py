import re
from gensim.models import FastText  # Switched from Word2Vec
from tqdm import tqdm

# Cleaning function
def clean_text(text):
    text = text.strip()  # Remove extra spaces
    text = re.sub(r"[।!,?;:\"'()\[\]{}<>।॥\-_@%$]", "", text)  # Remove punctuation
    return text

filename = 'mr.txt'

# Corpus Iterator
class CorpusIterator:
    def __iter__(self):
        with open(f"{filename}", "r", encoding="utf-8") as f:
            for line in f:
                cleaned_line = clean_text(line)
                if cleaned_line:  # Ignore empty lines
                    yield cleaned_line.split()  # Tokenize into words

sentences = CorpusIterator()

def count_sentences():
    count = 0
    with open(f"{filename}", "r", encoding="utf-8") as f:
        for line in tqdm(f):
            cleaned_line = clean_text(line)
            if cleaned_line:  # Ignore empty lines
                count += 1
    return count

# Define constants
TOTAL_SENTENCES = count_sentences()
EPOCHS = 5  # Number of training epochs

# Progress bar setup
total_work = TOTAL_SENTENCES * EPOCHS
progress_bar = tqdm(total=total_work, desc="Training FastText")

class TrackedCorpus:
    def __init__(self, corpus_iterator):
        self.corpus_iterator = corpus_iterator
        
    def __iter__(self):
        for sentence in self.corpus_iterator:
            progress_bar.update(1)  # Update progress bar per sentence
            yield sentence

tracked_sentences = TrackedCorpus(sentences)

# Train FastText model
model = FastText(tracked_sentences, vector_size=300, window=10, min_count=5, sample=1e-5,
                 workers=8, hs=0, negative=10, epochs=EPOCHS)

# Save the trained model
model.save("marathi_embeddings_ft.model")

progress_bar.close()
