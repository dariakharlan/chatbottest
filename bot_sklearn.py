from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

import dataset
import sys
from random import randint

print_mode = True
if len(sys.argv) > 1:
    print_mode = bool(sys.argv[1])

data = dataset.get_data()

if print_mode:
    print("%s sentences of training data" % len(data))

class_labels = {}

classes = list(set([a['class'] for a in data]))
for idx, c in enumerate(classes):
    class_labels[c] = idx

X = []
Y = []
for item in data:
    X.append(item['sentence'])
    Y.append(class_labels[item['class']])

if print_mode:
    print(class_labels)

count_vect = CountVectorizer(analyzer='char_wb', stop_words=["как", "что", "че", "где", "в"])
X_vect = count_vect.fit_transform(X)

if print_mode:
    print(X_vect.shape)


clf = MultinomialNB()
clf.fit(X_vect, Y)

answers = [
    {"greeting": "приветули", "life": "крутотенюшка", "bot": "яжбот", "price": "цена"},
    {"price": "стоимость", "greeting": "Здравствуйте. Чем могу помочь?", "life": "Я мыслю, значит существую?", "bot": "Я всего лишь искуствееный интеллект. Пока что..."},
]


def get_answer(sentence, show_details=False):
    sentence_vect = count_vect.transform([sentence])
    predicted = clf.predict(sentence_vect)

    probabilities = clf.predict_proba(sentence_vect)

    if show_details:
        print(predicted[0])
        print(classes[predicted[0]])
        print(probabilities)

    if probabilities[0][predicted[0]] < 0.5:
        return 'ne ponimat'

    return answers[randint(0, len(answers)-1)][classes[predicted[0]]]


print("Привет")
print("Ты можешь со мной поговорить, если хочешь")
print("Просто напиши мне  \n")

while True:
    text = input("- ")
    if text == 'пока':
        print('-  пакеда')
        break
    print("- " + get_answer(text, print_mode) + '\n')

