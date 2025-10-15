
import tensorflow_datasets as tfds

# List all available configs for imdb_reviews
print(tfds.list_builders())
import tensorflow_datasets as tfds

# Load plain text IMDB dataset
(train_data, test_data), info = tfds.load(
    "imdb_reviews/plain_text",
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True,
    as_supervised=True
)

print(info)

import tensorflow_datasets as tfds

# Load plain text dataset
train_data, test_data = tfds.load(
    "imdb_reviews/plain_text",
    split=["train", "test"],
    as_supervised=True
)

# Build a subword tokenizer
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (text.numpy() for text, _ in train_data),
    target_vocab_size=2**13
)

# Test encoding-decoding
sample = "I loved the movie!"
encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded)

print("Encoded:", encoded)
print("Decoded:", decoded)

