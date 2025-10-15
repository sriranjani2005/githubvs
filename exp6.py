import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Downloads
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")   # <-- NEW fix

# Define stop words
stop_words = set(stopwords.words("english"))

# Dummy text
txt = "Wow, you and Tom are really smart to save money in the bank."

# Sentence tokenization
tokenized = sent_tokenize(txt)

for i in tokenized:
    wordsList = word_tokenize(i)
    wordsList = [w for w in wordsList if w.lower() not in stop_words]
    tagged = nltk.pos_tag(wordsList)   # POS tagging
    print(tagged)
 