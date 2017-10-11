# This is bot

He speaks only russian

launch `bot.py` to start a conversation

to end conversation - say "пока"


To be improved:
- consider grammar mistakes, misspellings (привит, здрастуйте), mistypings (привеет, првиет)
- make training dataset larger
- maybe create more categories which bot can understand
and also separate existing categories for more detailed sets


This version of chatbot solves pretty simple task and has
only 42 sentences in training dataset so there is really no
point of using logistic regression or neural networks algorithm.
Multinomial Naive Bayes is good enough for this task and works fast.


For text preprocessing stemming was used, it works best
for simplifying words which is enough for this task 