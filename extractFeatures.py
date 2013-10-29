import json
import re
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import cross_validation

# Convert the entire dataset into a list of python dicts
data = []
f = open('training.json')
n = f.readline()
for line in f:
    data.append(json.loads(line))

# Get the heading, cities, sections and categories
heading = []
cities = set()
sections = set()
Y = []
for x in data:
    heading.append(x['heading'].lower().strip())
    cities.add(x['city'])
    sections.add(x['section'])
    Y.append(x['category'])
cities = list(cities)
sections = list(sections)
    
# Remove special characters, numbers, letters
remove_chars = [';', '-', '?', '#', '*', '/', '_', '(', ')', '&', ':', '<', '>',
                '{', '}', '.', '!', '@', '\\', '$', '%', '~', '`', '^', '-', '+'
                , '=', '[', ']', '\'', '"', ',']
temp = []
for text in heading:
    for sym in remove_chars:
        text = text.replace(sym, ' ')
    for sym in re.findall(r'\d+|\d+[.,]\d+', text):
        text = text.replace(sym, ' numbr ')
    t = " ".join([x for x in text.split() if len(x)>1])
    temp.append(" ".join(t.split()))
heading = temp

vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1,3),max_features=50000)
X = vectorizer.fit_transform(heading)

clf = svm.SVC(kernel='linear', C=1)
clf.fit(X, Y)

test = []
ftest = open('sample-test.in.json')
ntest = ftest.readline()
for line in ftest:
    test.append(json.loads(line))

headings_test = []   
for x in test:
    headings_test.append(x['heading'].lower().strip())

temp = []
for text in headings_test:
    for sym in remove_chars:
        text = text.replace(sym, ' ')
    for sym in re.findall(r'\d+|\d+[.,]\d+', text):
        text = text.replace(sym, ' numbr ')
    t = " ".join([x for x in text.split() if len(x)>1])
    temp.append(" ".join(t.split()))
headings_test = temp

Y_test = []
fytest = open('sample-test.out.json')
for line in fytest:
    Y_test.append(line)

trans = vectorizer.transform(numpy.array(headings_test))
print cross_validation.cross_val_score(clf, trans, numpy.array(Y_test)).mean()
