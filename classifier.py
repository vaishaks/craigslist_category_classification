import json
import re
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import cross_validation

def clean_data(heading, cities, sections):
    # Remove special characters, numbers, letters
    remove_chars = [';', '-', '?', '#', '*', '/', '_', '(', ')', '&', ':', '<', '>',
                '{', '}', '.', '!', '@', '\\', '$', '%', '~', '`', '^', '-', '+'
                , '=', '[', ']', '\'', '"', ',']
    temp = []
    # Count special characters, numbers and single characters
    spl = []
    numbr = []
    single_chars = []
    for text in heading:
        c = 0
        for sym in remove_chars:
            if sym in text:
                c += 1
            text = text.replace(sym, ' ')
        spl.append(c)
        if(len(re.findall(r'\d+|\d+[.,]\d+', text)) > 0):
            numbr.append(1)
        else:
            numbr.append(0)
        for sym in re.findall(r'\d+|\d+[.,]\d+', text):
            text = text.replace(sym, ' numbr ')
        c = 0
        for x in text.split():
            if len(x) == 1 and x != ' ':
                c += 1
        single_chars.append(c)
        t = " ".join([x for x in text.split() if len(x)>1])
        temp.append(" ".join(t.split()))
    heading = temp
    # Joining to form the feature vector
    temp = []
    for x in zip(heading, spl, numbr, single_chars, cities, sections):
        y = x[0] + " " + str(x[1]) + " " + str(x[2]) + " " + str(x[3]) + " " + x[4] + " " + x[5]
        temp.append(y)
    heading = temp
    return heading



# Convert the entire dataset into a list of python dicts
data = []
f = open('training.json')
n = f.readline()
for line in f:
    data.append(json.loads(line))

# Get the heading, cities, sections and categories
heading = []
cities = []
sections = []
Y = []
upper_case_count = []
for x in data:
    upper_case_count.append(sum(x.isupper() for x in x['heading']))
    heading.append(x['heading'].lower().strip())
    cities.append(x['city'])
    sections.append(x['section'])
    Y.append(x['category'])

heading  = clean_data(heading, cities, sections)

vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1,3),max_features=50000)
X = vectorizer.fit_transform(heading)

clf = LinearSVC()
clf.fit(X, Y)

test = []
ftest = open('sample-test.in.json')
ntest = ftest.readline()
for line in ftest:
    test.append(json.loads(line))

headings_test = []
cities_test = []
sections_test = []  
for x in test:
    headings_test.append(x['heading'].lower().strip())
    cities_test.append(x['city'])
    sections_test.append(x['section'])

headings_test = clean_data(headings_test, cities_test, sections_test)

Y_test = []
fytest = open('sample-test.out.json')
for line in fytest:
    Y_test.append(line)

trans = vectorizer.transform(numpy.array(headings_test))
print cross_validation.cross_val_score(clf, trans, numpy.array(Y_test)).mean()
