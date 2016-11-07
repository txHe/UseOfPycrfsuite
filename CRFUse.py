#coding=utf-8
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite

#Corpus Read
print(nltk.corpus.conll2002.fileids())

train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))

#Features
"""
define some features,this example, use word identity,word suffix,word shape and
word pos tag; also some informaion from nearby words is used.

this makes a simple baseline.
"""

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=%s' % word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.issupper=%s' % word1.isupper(),
            '-1:postag=%s' % postag1,
            '-1:postag[:2]=%s' % postag1[:2],
        ])
    else:
        features.append('BOS')

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=%s' % word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.issupper=%s' % word1.isupper(),
            '+1:postag=%s' % postag1,
            '+1:postag[:2]=%s' % postag1[:2],
        ])
    else:
        features.append('EOS')

    return features

def sent2features(sent):
    return [word2features(sent,i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token,postag,label in sent]

def sent2tokens(sent):
    return [token for token,postag,label in sent]

print(sent2features(train_sents[0])[0])

#  Extract the features from the data

X_train = [sent2features(s) for s in train_sents]
Y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
Y_test = [sent2labels(s) for s in test_sents]

# Train the model
"""
Train the model, create pycrfsuite. Trainer, load the training data and call
'train' method.
"""

trainer = pycrfsuite.Trainer(verbose=False)

for xseq,yseq in zip(X_train,Y_train):
    trainer.append(xseq,yseq)

"""
Set training parameters. We will use L-BFGS training algorithms(default) with
Elastic Net(L1 + L2) regularization
"""

trainer.set_params({
    'c1' : 1.0, #coefficient for L1 penalty
    'c2' : 1e-3, #coefficient for L2 penalty
    'max_iterations':50, #stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions':True
})

print(trainer.params())

#Train the model
trainer.train('conll2002-esp.crfsuite')

#Make predictions
"""
To use the trained model,crate pycrfsuite.Tagger, open the model and use "tag" method.
"""
tagger = pycrfsuite.Tagger()
tagger.open('conll2002-esp.crfsuite')

#tag a sentence to see how it works

example_sent = test_sents[0]
print(''.join(sent2tokens(example_sent)))
print('\n\n')

print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
print("Correct: ", ' '.join(sent2labels(example_sent)))

#Evaluate the model
"""
Classification report for a list of BIO-encoded sequences.
It computes token-level metrics and discards "O" labels.

Note that it requires scikit-learn 0.15+(or a version from github master)
to calculate averages properly!
"""

def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

# Predict entity labels for all sequences in our testing set

Y_pred = [tagger.tag(xseq) for xseq in X_test]

print(bio_classification_report(Y_test, Y_pred))
