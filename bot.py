import nltk
from nltk.stem.snowball import RussianStemmer

import dataset

data = dataset.get_data()

print ("%s sentences of training data" % len(data))

corpus_words = {}
class_words = {}

classes = list(set([a['class'] for a in data]))
for c in classes:
    class_words[c] = []

for item in data:
    for word in nltk.word_tokenize(item['sentence']):
        stemmed_word = RussianStemmer().stem(word)

        if stemmed_word not in corpus_words:
            corpus_words[stemmed_word] = 1
        else:
            corpus_words[stemmed_word] += 1

        class_words[item['class']].extend([stemmed_word])


# we now have each stemmed word and the number of occurances of the word in our training corpus (the word's commonality)
# print (corpus_words)
# also we have all words in each class
# print (class_words)


def calculate_class_score(sentence, class_name, show_details=True):
    score = 0
    # tokenize each word in our new sentence
    for word in nltk.word_tokenize(sentence):
        # check to see if the stem of the word is in any of our classes
        if RussianStemmer().stem(word.lower()) in class_words[class_name]:
            # treat each word with same weight
            score += 1

            if show_details:
                print("   match: %s" % RussianStemmer().stem(word.lower()))
    return score

def get_answer(sentence):
    answers = {"greetings": "приветули", "life": "крутотенюшка", "bot": "яжбот"}
    scores = {}
    for c in class_words.keys():
        score = calculate_class_score(sentence, c)
        print("Class: %s  Score: %s \n" % (c, score))

    max_index = scores.index(max(scores))
    return answers[classes[max_index]]


# print("Привет \n")
# print("Ты можешь со мной поговорить, если хочешь \n")
# print("Просто напиши мне  \n")

while True:
    text = input("- ")
    print(get_answer(text) + '\n')

