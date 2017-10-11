import nltk
from nltk.stem.snowball import RussianStemmer

import dataset
import sys
from random import randint

print_mode = False
if len(sys.argv) > 1:
    print_mode = bool(sys.argv[1])

data = dataset.get_data()

if print_mode:
    print("%s sentences of training data" % len(data))

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

if print_mode:
    print(corpus_words)
    print(class_words)


def calculate_class_score(sentence, class_name, show_details=False):
    score = 0
    # tokenize each word in our new sentence
    for word in nltk.word_tokenize(sentence):
        # check to see if the stem of the word is in any of our classes
        if RussianStemmer().stem(word.lower()) in class_words[class_name]:
            # treat each word with same weight
            score += (1 / corpus_words[RussianStemmer().stem(word.lower())])

            if show_details:
                print("   match: %s" % RussianStemmer().stem(word.lower()))
    return score


def get_answer(sentence, show_details=False):
    answers = [
        {"greeting": "приветули", "life": "крутотенюшка", "bot": "яжбот"},
        {"greeting": "Здравствуйте. Чем могу помочь?", "life": "Я мыслю, значит существую?", "bot": "Я всего лишь искуствееный интеллект. Пока что..."},
    ]
    scores = []
    for c in class_words.keys():
        score = calculate_class_score(sentence, c, print_mode)
        scores.append(score)

        if show_details:
            print("Class: %s  Score: %s \n" % (c, score))

    max_score = max(scores)
    if max_score == 0:
        return answers[randint(0, len(answers)-1)]['bot']

    max_index = scores.index(max_score)
    return answers[randint(0, len(answers)-1)][classes[max_index]]


print("Привет")
print("Ты можешь со мной поговорить, если хочешь")
print("Просто напиши мне  \n")

text = ''
while text != 'пока':
    text = input("- ")
    print("- " + get_answer(text, print_mode) + '\n')

