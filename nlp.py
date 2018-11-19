
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from functools import reduce
import pickle
import tarfile
import numpy as np
import re
import os
import time
import flask
from flask import request
from flask import render_template_string, render_template
from flask import Flask, jsonify

app = Flask(__name__, static_url_path='')
app.config["DEBUG"] = True

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences
    that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data):
    inputs, queries, answers = [], [], []
    for story, query, answer in data:
        inputs.append([word_idx[w] for w in story])
        queries.append([word_idx[w] for w in query])
        answers.append(word_idx[answer])
    return (pad_sequences(inputs, maxlen=story_maxlen),
            pad_sequences(queries, maxlen=query_maxlen),
            np.array(answers))

# ----------- PATHS ---------------------
filepath = os.path.dirname(__file__)
try:
  filepath = os.environ["PROJECT_DIR"] 
except KeyError:
  pass

print("Current Directory: " + filepath)
# Load the model, if it exists, load vocab too
modelpath = os.path.join(filepath,"chatbot.h5")
print("Model Path: " + modelpath)
model = load_model(modelpath)

pickle_vocabpath = os.path.join(filepath,"vocab.pkl") 
print("Pickle  Vocab  Path: " + pickle_vocabpath)
vocab = pickle.load( open(pickle_vocabpath, "rb" ))
try:
    path =  os.path.join(filepath, 'babi-tasks-v1-2.tar10.gz')
    print("Babi  Tasks  Path: " +path)
except:
   # print('Error downloading dataset, please download it manually:\n'
    #      '$ wget http://www.thespermwhale.com/jaseweston/babi/babi-tasks-v1-2.tar8.gz\n'
     #     '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
tar = tarfile.open(path)

challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
}



challenge_type = 'single_supporting_fact_10k'
challenge = challenges[challenge_type]

#print('Extracting stories for the challenge:', challenge_type)
train_stories = get_stories(tar.extractfile(challenge.format('train')))
test_stories = get_stories(tar.extractfile(challenge.format('test')))

vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)


# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories)

pred = model.predict([inputs_test, queries_test])
# See what the predictions look like, they are just probabilities of each class.
#print(pred)

pred = np.argmax(pred,axis=1)
#print(pred)

score = metrics.accuracy_score(answers_test, pred)
#print("Final accuracy: {}".format(score))


#print("Remember, I only know these words: {}".format(vocab))
#print()
story = "Task today is tasktoday. Task for yesterday was taskyesterday.  Task for tomorrow is tasktomorrow."




@app.route('/api', methods=['GET'])
def home():
   
    query = request.args['query']
    #query = "Where is Daniel?"
    adhoc_stories = (tokenize(story), tokenize(query), '?')

    adhoc_train, adhoc_query, adhoc_answer = vectorize_stories([adhoc_stories])
    pred = model.predict([adhoc_train, adhoc_query])
    pred = np.argmax(pred,axis=1)
    answer = "{}".format(vocab[pred[0]-1])
    return jsonify({'tasks': answer}) 

	
@app.route('/' , methods=['GET'])
def render_static():
    return render_template("NLP_API_CALL.html", title = '')
    #return render_template('%NLP_API_CALL.html' % page_name)

app_port = 5000
try:
    app_port = os.environ["PORT"]
except KeyError:
    pass
app.run(port=app_port, host='0.0.0.0')