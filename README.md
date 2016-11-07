# UseOfPycrfsuite
通过示例阐述如何使用pycrfsuite

###How to use pycrfsuite

本文翻译自：http://nbviewer.jupyter.org/github/tpeng/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb
这篇文章将通过一个命名实体识别的例子来讲解下CRFSuite在Python上的使用。
首先需要先安装几个第三方包： nltk、sklearn和pycrfsuite。
安装成功后，就进行下面的操作：

####1、语料库读取

选择nltk内建立的Coll2002语料库。可在python shell里通过如下命令查看：

```
nltk.corpus.conll2002.fileids()
```
输出：

```
[u'esp.testa',
 u'esp.testb',
 u'esp.train',
 u'ned.testa',
 u'ned.testb',
 u'ned.train']
```

语料库内标注了词性，和对应的命名实体信息。

读取测试集和训练集：

```
train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
```

####2、特征处理
特征处理流程，主要选择处理了如下几个特征：

 - 当前词的小写格式 
 - 当前词的后缀
 - 当前词是否全大写 isupper
 - 当前词的首字母大写，其他字母小写判断 istitle
 - 当前词是否为数字 isdigit
 - 当前词的词性
 - 当前词的词性前缀
 - 还有就是与之前后相关联的词的上述特征（类似于特征模板的定义）

```
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
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
                
    return features

#完成特征转化
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

#获取类别，即标签
def sent2labels(sent):
    return [label for token, postag, label in sent]

#获取词
def sent2tokens(sent):
    return [token for token, postag, label in sent]    
```
 
特征如上转化完成后，可以查看下一行特征内容：

输入命令：
```
sent2features(train_sents[0])[0]Out[6]:
```
输出：
```
 ['bias',
 u'word.lower=melbourne',
 u'word[-3:]=rne',
 u'word[-2:]=ne',
 'word.isupper=False',
 'word.istitle=True',
 'word.isdigit=False',
 u'postag=NP',
 u'postag[:2]=NP',
 'BOS',
 u'+1:word.lower=(',
 '+1:word.istitle=False',
 '+1:word.isupper=False',
 u'+1:postag=Fpa',
 u'+1:postag[:2]=Fp']
```
  
构造特征训练集和测试集：
```
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]
```

####3、训练

创建pycrfsuite.Trainer，加载训练数据，然后开始训练。

**1）创建Trainer并加载训练集**
```
trainer = pycrfsuite.Trainer(verbose=False)

#加载训练特征和分类的类别（label)
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)
```

**2）设置训练参数，包括算法选择**。
这里选择L-BFGS训练算法和Elastic Net回归模型

```
trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})
```

**3）开始训练**
```
#含义是训练出的模型名为：conll2002-esp.crfsuite
trainer.train('conll2002-esp.crfsuite')
```

####4、测试

使用训练后的模型，创建用于测试的标注器。
```
tagger = pycrfsuite.Tagger()
tagger.open('conll2002-esp.crfsuite')
```

然后标注一个句子试试：
```
example_sent = test_sents[0]
print(' '.join(sent2tokens(example_sent)), end='\n\n')

print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
print("Correct:  ", ' '.join(sent2labels(example_sent)))
```

输出如下：
```
La Coruña , 23 may ( EFECOM ) .

Predicted: B-LOC I-LOC O O O O B-ORG O O
Correct:   B-LOC I-LOC O O O O B-ORG O O
```

####5、评测
使用sklearn的classification_report来做评测：
```
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
```

标注所有信息：
```
y_pred = [tagger.tag(xseq) for xseq in X_test]
```

打印出评测报告：
```
print(bio_classification_report(y_test, y_pred))

#输出
             precision    recall  f1-score   support

      B-LOC       0.78      0.75      0.76      1084
      I-LOC       0.87      0.93      0.90       634
     B-MISC       0.69      0.47      0.56       339
     I-MISC       0.87      0.93      0.90       634
      B-ORG       0.82      0.87      0.84       735
      I-ORG       0.87      0.93      0.90       634
      B-PER       0.61      0.49      0.54       557
      I-PER       0.87      0.93      0.90       634
      avg / total       0.81      0.81      0.80      5251
```

**参考文献**

http://nbviewer.jupyter.org/github/tpeng/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb