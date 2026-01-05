Code to train T5 for different tasks

These files need to be run from the root directory.

- `train-t5-dks.py` trains the model on transliteration task.
- `train-t5-gc-gtp.py` trains the model on grapheme-to-phoneme task.
- `train-t5-gc-qa.py` trains the model on multilingual question answering task. We can change the datasets being loaded to train them on other variations of the QA task. 