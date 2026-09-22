# Corpus

`input.txt` is an original, short synthetic text supplied only to exercise the
training and validation pipeline without downloading anything. It is too small
to provide a meaningful language-model evaluation or generate convincing text.

For a substantive experiment, place a local UTF-8 text file here, for example
`data/tiny_shakespeare.txt`, and run:

```sh
Rscript experiments/run.R --input=data/tiny_shakespeare.txt --iterations=2000
```

Tiny Shakespeare can be obtained from the public Tiny Shakespeare corpus used
by Karpathy's `char-rnn` project. Review the corpus source and applicable terms
before distributing a copy. Training reads bytes as-is, splits the file
contiguously into 80% train, 10% validation and 10% untouched test, and creates
input/next-token pairs independently within each split. Never train on the
validation or test portions. The initial tokenizer maps raw bytes directly to
1-based IDs; byte-level BPE is a later standalone extension.
