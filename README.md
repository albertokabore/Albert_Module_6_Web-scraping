# Web Scraping and NLP with Requests, BeautifulSoup, and spaCy

### Student Name: Albert Kabore

### Link to GitHub: https://github.com/albertokabore/Albert_Module_6_Web-scraping

Complete the tasks in the Python Notebook in this repository.
Make sure to add and push the pkl or text file of your scraped html (this is specified in the notebook)

## Rubric

* (Question 1) Article html stored in separate file that is committed and pushed: 1 pt
* (Question 2) Article text is correct: 1 pt
* (Question 3) Correct (or equivalent in the case of multiple tokens with same frequency) tokens printed: 1 pt
* (Question 4) Correct (or equivalent in the case of multiple lemmas with same frequency) lemmas printed: 1 pt
* (Question 5) Correct scores for first sentence printed: 2 pts (1 / function)
* (Question 6) Histogram shown with appropriate labelling: 1 pt
* (Question 7) Histogram shown with appropriate labelling: 1 pt
* (Question 8) Thoughtful answer provided: 1 pt


```python
from collections import Counter
import pickle
import requests
import spacy
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
!pip list

print('All prereqs installed.')
```

Question 1. Write code that extracts the article html from https://web.archive.org/web/20210327165005/https://hackaday.com/2021/03/22/how-laser-headlights-work/ and dumps it to a .pkl (or other appropriate file)

Question 2. Read in your article's html source from the file you created in question 1 and print it's text (use .get_text())

```python
import pickle
from bs4 import BeautifulSoup

# Loading the HTML content from the .pkl file
with open("parsed_article.pkl", "rb") as file:
    soup = pickle.load(file)

# Trying to get only the article we are after from this webpage
onlyArticle = soup.find('article')

# Check if the article was found, then extract and print the text
if onlyArticle:
    articleText = onlyArticle.get_text()
    print(articleText)
else:
    print("The <article> tag was not found in the HTML content.")
```

Question 3. Load the article text into a trained spaCy pipeline, and determine the 5 most frequent tokens (converted to lower case). Print the common tokens with an appropriate label. Additionally, print the tokens their frequencies (with appropriate labels). Make sure to remove things we don't care about (punctuation, stopwords, whitespace).

```python
# Load the spacy language model
nlp = spacy.load("en_core_web_sm")

# Load the article text from the pickle file
with open("parsed_article.pkl", "rb") as file:
    soup = pickle.load(file)

# Extract and clean the article text
onlyArticle = soup.find('article')
if onlyArticle:
    articleText = onlyArticle.get_text()
else:
    print("The <article> tag was not found in the HTML content.")
    articleText = ""

# Process the article text with spaCy
doc = nlp(articleText)

# Filter tokens: remove punctuation, stopwords, and whitespace
filtered_tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]

# Count the frequency of each token
token_freq = Counter(filtered_tokens)

# Get the 5 most common tokens
most_common_tokens = token_freq.most_common(5)

# Print the results
print("Top 5 Most Frequent Tokens:")
for token, freq in most_common_tokens:
    print(f"Token: {token}, Frequency: {freq}")
```

Question 4. Load the article text into a trained spaCy pipeline, and determine the 5 most frequent lemmas (converted to lower case). Print the common lemmas with an appropriate label. Additionally, print the lemmas with their frequencies (with appropriate labels). Make sure to remove things we don't care about (punctuation, stopwords, whitespace).

```python
# Load the spacy language model
nlp = spacy.load("en_core_web_sm")

# Load the article text from the pickle file
with open("parsed_article.pkl", "rb") as file:
    soup = pickle.load(file)

# Extract and clean the article text
onlyArticle = soup.find('article')
if onlyArticle:
    articleText = onlyArticle.get_text()
else:
    print("The <article> tag was not found in the HTML content.")
    articleText = ""

# Process the article text with spaCy
doc = nlp(articleText)

# Filter lemmas: remove punctuation, stopwords, and whitespace
filtered_lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]

# Count the frequency of each lemma
lemma_freq = Counter(filtered_lemmas)

# Get the 5 most common lemmas
most_common_lemmas = lemma_freq.most_common(5)

# Print the results
print("Top 5 Most Frequent Lemmas:")
for lemma, freq in most_common_lemmas:
    print(f"Lemma: {lemma}, Frequency: {freq}")
```

Question 5: Define the following methods:
score_sentence_by_token(sentence, interesting_token) that takes a sentence and a list of interesting token and returns the number of times that any of the interesting words appear in the sentence divided by the number of words in the sentence
score_sentence_by_lemma(sentence, interesting_lemmas) that takes a sentence and a list of interesting lemmas and returns the number of times that any of the interesting lemmas appear in the sentence divided by the number of words in the sentence
You may find some of the code from the in class notes useful; feel free to use methods (rewrite them in this cell as well). Test them by showing the score of the first sentence in your article using the frequent tokens and frequent lemmas identified in question 3.

```python
# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

def score_sentence_by_token(sentence, interesting_tokens):
    """
    Scores a sentence based on the frequency of interesting tokens.
    
    Args:
        sentence (str): The sentence to score.
        interesting_tokens (list of str): List of interesting tokens.
    
    Returns:
        float: The score (frequency of interesting tokens / total words).
    """
    doc = nlp(sentence)
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    token_count = sum(1 for token in tokens if token in interesting_tokens)
    return token_count / len(tokens) if tokens else 0

def score_sentence_by_lemma(sentence, interesting_lemmas):
    """
    Scores a sentence based on the frequency of interesting lemmas.
    
    Args:
        sentence (str): The sentence to score.
        interesting_lemmas (list of str): List of interesting lemmas.
    
    Returns:
        float: The score (frequency of interesting lemmas / total words).
    """
    doc = nlp(sentence)
    lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    lemma_count = sum(1 for lemma in lemmas if lemma in interesting_lemmas)
    return lemma_count / len(lemmas) if lemmas else 0

# Example to test the functions
# Use the first sentence of the article and the frequent tokens and lemmas identified in Question 3 and 4
with open("parsed_article.pkl", "rb") as file:
    soup = pickle.load(file)

# Extract the article text and split into sentences
onlyArticle = soup.find('article')
if onlyArticle:
    articleText = onlyArticle.get_text()
    first_sentence = list(nlp(articleText).sents)[0].text  # First sentence
else:
    print("The <article> tag was not found in the HTML content.")
    first_sentence = ""

# Frequent tokens and lemmas from earlier questions
frequent_tokens = ["light", "laser", "system", "beam", "headlights"]  # Example tokens
frequent_lemmas = ["light", "laser", "system", "beam", "headlight"]  # Example lemmas

# Calculate and print scores
token_score = score_sentence_by_token(first_sentence, frequent_tokens)
lemma_score = score_sentence_by_lemma(first_sentence, frequent_lemmas)

print(f"First Sentence: {first_sentence}")
print(f"Token Score: {token_score}")
print(f"Lemma Score: {lemma_score}")
```

Question 6. Make a list containing the scores (using tokens) of every sentence in the article, and plot a histogram with appropriate titles and axis labels of the scores. From your histogram, what seems to be the most common range of scores (put the answer in a comment after your code)?

```python
# Define the URL and fetch the article
url = "https://web.archive.org/web/20210327165005/https://hackaday.com/2021/03/22/how-laser-headlights-work/"
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    article = soup.find('article').get_text()
else:
    raise Exception(f"Failed to fetch the article: {response.status_code}")

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Token Scoring Function
def score_sentence_by_token(sentence, interesting_tokens):
    doc = nlp(sentence)
    words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    if len(words) == 0:
        return 0.0
    return sum(1 for word in words if word in interesting_tokens) / len(words)

# Extract sentences and calculate scores
sentences = [sent.text for sent in nlp(article).sents]
interesting_tokens = ["light", "laser", "system", "beam", "headlights"]
token_scores = [score_sentence_by_token(sentence, interesting_tokens) for sentence in sentences]

# Plot the histogram
plt.hist(token_scores, bins=np.arange(0, 1.1, 0.1), edgecolor="black")
plt.title("Histogram of Sentence Scores (Using Tokens)")
plt.xlabel("Score Range")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Comment on the most common range
# The most common range of scores appears to be around [0.0, 0.1] based on the histogram.
```

Question 7. Make a list containing the scores (using lemmas) of every sentence in the article, and plot a histogram with appropriate titles and axis labels of the scores. From your histogram, what seems to be the most common range of scores (put the answer in a comment after your code)?

```python
# Fetch the article
url = "https://web.archive.org/web/20210327165005/https://hackaday.com/2021/03/22/how-laser-headlights-work/"
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    article_text = soup.find('article').get_text()
else:
    raise Exception(f"Failed to fetch the article: {response.status_code}")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define scoring function for lemmas
def score_sentence_by_lemma(sentence, interesting_lemmas):
    doc = nlp(sentence)
    lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    if len(lemmas) == 0:
        return 0.0
    return sum(1 for lemma in lemmas if lemma in interesting_lemmas) / len(lemmas)

# Extract sentences and calculate scores
sentences = [sent.text for sent in nlp(article_text).sents]
interesting_lemmas = ["light", "laser", "system", "beam", "headlight"]  # Example lemmas from Q4
lemma_scores = [score_sentence_by_lemma(sentence, interesting_lemmas) for sentence in sentences]

# Plot the histogram of lemma scores
plt.hist(lemma_scores, bins=np.arange(0, 1.1, 0.1), edgecolor="black")
plt.title("Histogram of Sentence Scores (Using Lemmas)")
plt.xlabel("Score Range")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Comment on the most common range of scores
# The most common range of scores appears to be around [0.0, 0.1] based on the histogram.
```

Which tokens and lexems would be ommitted from the lists generated in questions 3 and 4 if we only wanted to consider nouns as interesting words? How might we change the code to only consider nouns? Put your answer in this Markdown cell (you can edit it by double clicking it).

