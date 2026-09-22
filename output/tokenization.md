# From text to next-token training data

This report uses the project's **actual byte tokenizer and batching functions**.
Each token represents one raw byte; token ID = byte value + 1 for R's one-based indexing.
This is **not yet GPT-2's byte-pair encoding (BPE)**.

## 1. Encode a readable input

Input text: "you are,"  
Vocabulary: 256 possible byte tokens.  
UTF-8 bytes (hex): `79 6F 75 20 61 72 65 2C`  
Token IDs (R): `122 112 118 33 98 115 102 45`  
Decoded round-trip: "you are,"

| Position | Byte shown | Hex byte | Token ID |
|---:|:---|:---:|---:|
| 1 | y | 79 | 122 |
| 2 | o | 6F | 112 |
| 3 | u | 75 | 118 |
| 4 | SPACE | 20 | 33 |
| 5 | a | 61 | 98 |
| 6 | r | 72 | 115 |
| 7 | e | 65 | 102 |
| 8 | , | 2C | 45 |

## 2. Read the real corpus

Corpus file: `tiny_shakespeare.txt`  
Total tokens (bytes): 1,115,394  
Contiguous split sizes — train: 892,315; validation: 111,539; test: 111,540.

`read_corpus()` reads the original file as raw bytes without normalisation.
`split_corpus()` holds out the later 10% for validation and the final 10% for testing.

The first eight bytes of this corpus's validation split (the experiment's inspection prompt):  
Decoded: "you are,"  
Token IDs: `122 112 118 33 98 115 102 45`

## 3. Create one real training window

Training split, starting byte position: 1  
Context length: 8  
Model input: "First Ci"  
Next-token targets: "irst Cit"

| Position | Input byte | Input hex | Input ID | Next byte (target) | Target hex | Target ID |
|---:|:---|:---:|---:|:---|:---:|---:|
| 1 | F | 46 | 71 | i | 69 | 106 |
| 2 | i | 69 | 106 | r | 72 | 115 |
| 3 | r | 72 | 115 | s | 73 | 116 |
| 4 | s | 73 | 116 | t | 74 | 117 |
| 5 | t | 74 | 117 | SPACE | 20 | 33 |
| 6 | SPACE | 20 | 33 | C | 43 | 68 |
| 7 | C | 43 | 68 | i | 69 | 106 |
| 8 | i | 69 | 106 | t | 74 | 117 |

**What the model learns:** at each input position it predicts a probability
distribution over the 256 possible *next-byte tokens*. The target column is
the observed next byte used to compute training loss; it is not a generated sample.
A UTF-8 character can occupy multiple bytes, so one token need not equal one character.

