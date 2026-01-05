# %%
import numpy as np
from numpy.linalg import norm
from numpy import dot
import stanza
from gensim.models import FastText
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics.pairwise import cosine_similarity

# Load Hindi language models
# stanza.download("mr")
nlp = stanza.Pipeline('mr', download_method=None,device='cuda:2')
ft_model = FastText.load('./embeddings/marathi_embeddings_ft.model')

# %%
# Function to calculate Levenshtein similarity
def levenshtein_similarity(word1, word2):
    max_len = max(len(word1), len(word2))
    if max_len == 0:
        return 0.0  # Prevent division by zero
    sim = 1.0 - (levenshtein_distance(word1, word2) / max_len)
    # print(f"Levenshtein Similarity({word1}, {word2}) = {sim}")
    return sim

x= 2

# Function to calculate embedding similarity using FastText
def path_measure(word1, word2):
    try:
        vec1 = ft_model.wv[word1]
        vec2 = ft_model.wv[word2]
        sim = cosine_similarity([vec1], [vec2])[0][0]
        # print(f"Path measure ({word1}, {word2}) = {sim}")
        return sim  # Ensure non-negative similarity
    except KeyError:
        # print(f"Word '{word1}' or '{word2}' not in FastText vocabulary")
        return 0.0


# Combined similarity measure (Algorithm 1)
def word_similarity(word1, word2):
    emb_sim = path_measure(word1, word2)
    if emb_sim < 0.1:
        return levenshtein_similarity(word1, word2)
    else:
        return emb_sim

# Algorithm 2: Proposed similarity algorithm
def sentence_similarity(sentence_a, sentence_b):
    # Use Stanza for Hindi tokenization
    doc_a = nlp(sentence_a)
    doc_b = nlp(sentence_b)
    
    # Get lemmatized tokens
    tokens_a = [word.lemma for sent in doc_a.sentences for word in sent.words]
    tokens_b = [word.lemma for sent in doc_b.sentences for word in sent.words]
    
    # Step 1: Initialize matrix and variables
    n, m = len(tokens_a), len(tokens_b)
    similarity_matrix = np.zeros((n, m))
    total_similarity = 0
    iterations = 0
    
    # Step 2: Calculate similarities for each token pair and populate the matrix
    for i, token_a in enumerate(tokens_a):
        for j, token_b in enumerate(tokens_b):
            similarity_matrix[i][j] = word_similarity(token_a, token_b)
    
    # Step 3: Process the matrix to calculate total similarity
    while similarity_matrix.size > 0:
        max_sim_index = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        total_similarity += similarity_matrix[max_sim_index]
        similarity_matrix = np.delete(similarity_matrix, max_sim_index[0], axis=0)
        similarity_matrix = np.delete(similarity_matrix, max_sim_index[1], axis=1)
        iterations += 1
        
        if similarity_matrix.size == 0:
            break
    
    partial_similarity = total_similarity / iterations if iterations > 0 else 0
    return partial_similarity

# Function to calculate numerical similarity (remains unchanged as numbers are universal)
def numerical_similarity(numbers_a, numbers_b):
    if not numbers_a or not numbers_b:
        return 0.0
    
    num_matches = sum(1 for num in numbers_a if num in numbers_b)
    return num_matches / max(len(numbers_a), len(numbers_b))

def calculate_sdpc(n, m, partial_similarity):
    if n == m:
        return 0
    elif n > m:
        return (abs(n - m) * partial_similarity) / n
    else:
        return (abs(n - m) * partial_similarity) / m

def lexical_similarity(sentence_a, sentence_b):
    # Process Hindi sentences with Stanza
    doc_a = nlp(sentence_a)
    doc_b = nlp(sentence_b)
    
    # Extract words and numerical tokens from sentences
    tokens_a = [word.text for sent in doc_a.sentences for word in sent.words if not word.text.isdigit()]
    tokens_b = [word.text for sent in doc_b.sentences for word in sent.words if not word.text.isdigit()]
    
    numbers_a = [int(word.text) for sent in doc_a.sentences for word in sent.words if word.text.isdigit()]
    numbers_b = [int(word.text) for sent in doc_b.sentences for word in sent.words if word.text.isdigit()]
    
    word_partial_similarity = sentence_similarity(" ".join(tokens_a), " ".join(tokens_b))
    num_partial_similarity = numerical_similarity(numbers_a, numbers_b)
    
    word_sdpc = calculate_sdpc(len(tokens_a), len(tokens_b), word_partial_similarity)
    num_sdpc = calculate_sdpc(len(numbers_a), len(numbers_b), num_partial_similarity)
    
    word_sim = max(0, word_partial_similarity - word_sdpc)
    number_sim = max(0, num_partial_similarity - num_sdpc)
    
    n_word = len(tokens_a) + len(tokens_b)
    n_number = len(numbers_a) + len(numbers_b)
    
    if n_number == 0:
        final_sim = word_sim
    elif n_word == 0:
        final_sim = number_sim
    else:
        if number_sim == 1.0:
            final_sim = ((n_word * word_sim) + (n_number * number_sim)) / (n_word + n_number)
        else:
            final_sim = max(0, word_sim - (1 - number_sim))
    
    return final_sim


# %%
def get_word_embedding(word):
    """Get FastText embedding for a Hindi word"""
    try:
        return ft_model.wv[word].reshape(1, -1)
    except KeyError:
        return np.zeros((1, ft_model.vector_size))
    
def sentence_to_rdf_triples(sentence):
    """
    Converts a Hindi sentence into RDF-like triples using dependency parsing.
    Each triple is of the form (head, dependency, token).
    """
    doc = nlp(sentence)
    triples = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.head > 0:
                head_word = sent.words[word.head - 1].text
                triple = (head_word, word.deprel, word.text)
                triples.append(triple)
    return triples  # Return the full list of triples instead of just 'triple'


def vertex_similarity(v1, v2):
    """
    Enhanced vertex similarity using FastText embeddings and multiple metrics
    """
    # Get embeddings
    emb1 = get_word_embedding(v1)
    emb2 = get_word_embedding(v2)
    
    # Exact match weight
    exact_match = 1.0 if v1 == v2 else 0.0
    
    # Cosine similarity between embeddings
    cos_sim = 1 - cosine_similarity(emb1, emb2)[0][0]
    
    # Weighted combination
    return 0.3 * exact_match + 0.7 * cos_sim

def calculate_triple_similarity(triple1, triple2):
    """
    Enhanced triple similarity with relation weighting
    """
    head1, rel1, dep1 = triple1
    head2, rel2, dep2 = triple2
    
    # Weight relation matching more heavily
    relation_match = 1.0 if rel1 == rel2 else 0.0
    
    # Calculate head and dependent similarities
    head_sim = vertex_similarity(head1, head2)
    dep_sim = vertex_similarity(dep1, dep2)
    
    # Weighted combination
    return 0.4 * head_sim + 0.4 * dep_sim + 0.2 * relation_match

def syntactic_similarity(sentence_1, sentence_2):
    """
    Improved syntactic similarity with normalized scoring
    """
    triples1 = sentence_to_rdf_triples(sentence_1)
    triples2 = sentence_to_rdf_triples(sentence_2)
    
    if not triples1 or not triples2:
        return 0.0
        
    max_similarities = []
    for triple1 in triples1:
        similarities = [calculate_triple_similarity(triple1, triple2) 
                       for triple2 in triples2]
        max_similarities.append(max(similarities))
    
    # Normalize by the number of triples
    return sum(max_similarities) / len(triples1)


# %%
# -----------------------------
# 4. Similarity Functions Using FastText Embeddings
# -----------------------------

def triple_similarity(triple1, triple2):
    """
    Computes similarity between two RDF-like triples.
    Each triple is a tuple: (head, dependency, token).
    The similarity is the average of the cosine similarities between:
      - the head words,
      - the dependency labels, and
      - the dependent tokens.
    """
    head1, rel1, token1 = triple1
    head2, rel2, token2 = triple2
    sim_head = path_measure(head1, head2)
    sim_rel = path_measure(rel1, rel2)
    sim_token = path_measure(token1, token2)
    return (sim_head + sim_rel + sim_token) / 3

def semantic_similarity(triples_a, triples_b):
    """
    Computes semantic similarity between two lists of RDF-like triples using a greedy matching strategy.
    A similarity matrix is built from pairwise triple similarities, then the maximum matching pairs' average is returned.
    """
    n = len(triples_a)
    m = len(triples_b)
    if n == 0 or m == 0:
        return 0.0

    similarity_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            similarity_matrix[i, j] = triple_similarity(triples_a[i], triples_b[j])
    
    total_similarity = 0.0
    iterations = 0
    # Greedy matching: iteratively pick highest similarity and remove corresponding row and column.
    while similarity_matrix.size > 0 and similarity_matrix.shape[0] > 0 and similarity_matrix.shape[1] > 0:
        idx = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)
        total_similarity += similarity_matrix[idx]
        similarity_matrix = np.delete(similarity_matrix, idx[0], axis=0)
        similarity_matrix = np.delete(similarity_matrix, idx[1], axis=1)
        iterations += 1

    partial_similarity = total_similarity / iterations if iterations > 0 else 0.0
    return partial_similarity

# -----------------------------
# 5. Overall Sentence Semantic Similarity Function
# -----------------------------
def sentence_semantic_similarity(sentence1, sentence2):
    """
    Computes semantic similarity between two plain Hindi sentences.
    The process:
      1. Convert sentences to RDF-like triples using dependency parsing.
      2. Compute semantic similarity between the two sets of triples.
    """
    triples1 = sentence_to_rdf_triples(sentence1)
    triples2 = sentence_to_rdf_triples(sentence2)
    return semantic_similarity(triples1, triples2)


# %%
def overall_sentence_similarity(sentence1, sentence2):
    # Compute individual similarity measures
    lexical_sim = lexical_similarity(sentence1, sentence2)
    syntactic_sim = syntactic_similarity(sentence1, sentence2)
    semantic_sim = sentence_semantic_similarity(sentence1, sentence2)

    print(f"Lexical Similarity: {lexical_sim}")
    print(f"Syntactic Similarity: {syntactic_sim}")
    print(f"Semantic Similarity: {semantic_sim}")

    # Compute the number of elements contributing to each similarity measure
    lexn = len(sentence1.split()) + len(sentence2.split())  # Total word count (excluding numbers)
    synn = len(sentence_to_rdf_triples(sentence1)) + len(sentence_to_rdf_triples(sentence2))  # Total syntactic triples
    semn = synn  # Assuming semantic triples are the same as syntactic triples in count

    # Calculate the overall similarity score
    if lexn + synn + semn == 0:  # Avoid division by zero
        return 0.0
    
    overall_similarity = ((lexn * lexical_sim) + (synn * syntactic_sim) + (semn * semantic_sim)) / (lexn + synn + semn)
    
    return overall_similarity


# %%

# # Test sentences
# sent1 = "राम ने सीता को फूल दिया।"
# sent2 = "सीता ने राम से फूल लिया।"

# similarity_score = lexical_similarity(sent1, sent2)
# print(f"Lexical Score: {similarity_score:.2f}")

# # Calculate similarity
# sim = syntactic_similarity(sent1, sent2)
# print(f"Syntactic similarity score: {sim:.4f}")

# sim_score = sentence_semantic_similarity(sent1, sent2)
# print("Semantic Similarity Score:", sim_score)

# final_similarity = overall_sentence_similarity(sent1, sent2)
# print(f"Overall Similarity Score: {final_similarity:.4f}")



