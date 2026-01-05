Code to automatically evaluate the question answers.

- `embed_ft.py` was used to train word embedding models for Hindi and Marathi. (Using C4 data)
- `lss_similarity_*.py` is based on Ferreira et al.'s methodology.
- `auto_eval_*.py` evaluated different types of questions based on predictions generated from the `eval/eval_t5_squad.py` file. 