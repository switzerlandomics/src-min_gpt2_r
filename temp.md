
## Tokenizer
To understand the tokenizer see `experiments/inspect_tokenizer.R`. It uses the same `R/tokenizer.R` and `R/data.R` with no additional packages, model training or changes to the main experiment runner.

```sh
Rscript experiments/inspect_tokenizer.R \
  --input=data/tiny_shakespeare.txt \
  --text="you are," \
  --context-length=8 \
  --start=1 \
  --output=output/tokenization.md
```

The script prints its report to the console and saves the same report to `output/tokenization.md`, ready to copy into a blog draft.

### What the report demonstrates

The report follows three stages, using the actual project functions:

1. Text → bytes → token IDs

For the familiar prompt `you are,`, it prints the UTF-8 bytes, corresponding integer IDs and a position-by-position table.

```
Text:       you are,
Hex bytes:  79 6F 75 20 61 72 65 2C
Token IDs:  122 112 118 33 98 115 102 45
```

2. Read and partition the actual corpus

It shows the number of byte tokens in the supplied file, the training/validation/test split sizes, and the first eight bytes of the validation split—the input used by your experiment's inspection figures.

3. Construct a real training batch

It takes a window from the training split and shows each input byte beside its next-token target, including the hexadecimal values and numerical IDs.

The third stage is particularly useful for the article: it shows that the model is trained to predict the next byte at every position, whereas the generation examples show the model sampling new bytes from its learned probability distributions.

The script is independent of Shakespeare-specific assumptions. You can supply another corpus with `--input`, choose a different readable example with `--text`, or inspect a different training window with `--start`.

Verification: I created the complete file and checked its dependencies against the tokenizer and batching interfaces you supplied. I could not execute it here because `Rscript` is unavailable in this environment, so the generated report should be checked with a local run before publication.

