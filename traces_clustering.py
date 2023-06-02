import re
from functools import partial

import nltk
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk import word_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

nltk.download('punkt')


class ActivityEventClusteringModel:
    def __init__(self, name, vec_size=10, traces=None):
        self.__name = name
        self.__vec_size = vec_size
        self.__vectorizer = None
        if traces is not None:
            self.train(traces)

    def train(self, logs):
        traces = logs.groupby(['traceId'])
        traces = [y for x, y in traces]

        messages = []
        for trace in traces:
            trace['cleaned_message'] = trace.apply(lambda log: clean_message(log['message']), axis=1)
            messages.extend(trace['cleaned_message'].values.tolist())
        self.__update_vectorizer(messages)

        traces_vectors = []
        for trace in traces:
            trace['vector'] = trace.apply(lambda log: self.__vectorize(log['cleaned_message']), axis=1)
            traces_vectors.extend(trace['vector'].values.tolist())

        min_trace_length = min(len(x) for x in traces)
        max_trace_length = max(len(x) for x in traces)
        if min_trace_length == max_trace_length == 1:
            return
        elif min_trace_length == 1:
            min_trace_length += 1
        self.__update_model(traces_vectors, min_trace_length, max_trace_length)

        self.__predict(traces)
        self.__save_train_data(traces)

    def __update_vectorizer(self, messages):
        data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(messages)]

        if self.__vectorizer is None:
            self.__vectorizer = Doc2Vec(vector_size=self.__vec_size, alpha=0.1, dm=1, min_count=1)
            self.__vectorizer.build_vocab(data)
        else:
            self.__vectorizer.build_vocab(data, update=True)

        if self.__vectorizer.corpus_count == 0:
            print(self.__name)
            return

        self.__vectorizer.train(data, total_examples=self.__vectorizer.corpus_count, epochs=50)

    def __vectorize(self, message):
        data = word_tokenize(message)
        return self.__vectorizer.infer_vector(data)

    def __update_model(self, vectors, min_trace_length, max_trace_length):
        sil = {}
        for k in range(min_trace_length, max_trace_length + 1, 1):
            model = KMeans(n_clusters=k, max_iter=100)
            model.fit_transform(vectors)
            sil[k] = silhouette_score(vectors, model.labels_, metric='euclidean')
        k = max(sil, key=sil.get)

        print('activity = ' + self.__name + ', best n_clusters = ' + str(k))
        self.__model = KMeans(n_clusters=k, max_iter=300)
        self.__model.fit(vectors)
        self.__num_classes = k

    def __predict(self, traces):
        for trace in traces:
            trace['label'] = trace.apply(lambda log: self.__model.predict([log['vector']])[0] + 1, axis=1)

    def __save_train_data(self, traces):
        self.__train_data_path = 'train/' + self.__name
        with open(self.__train_data_path, 'w') as fp:
            for trace in traces:
                for index, log in trace.iterrows():
                    fp.write("%s " % log['label'])
                fp.write("\n")
            fp.flush()

    def predict(self, logs):
        traces = logs.groupby(['traceId'])
        traces = [y for x, y in traces]
        for trace in traces:
            trace['cleaned_message'] = trace.apply(lambda log: clean_message(log['message']), axis=1)
            trace['vector'] = trace.apply(lambda log: self.__vectorize(log['cleaned_message']), axis=1)
        self.__predict(traces)
        self.__save_test_data(traces)

    def __save_test_data(self, traces):
        self.__test_data_path = 'test/' + self.__name
        with open(self.__test_data_path, 'w') as fp:
            for trace in traces:
                for index, log in trace.iterrows():
                    fp.write("%s " % log['label'])
                fp.write("\n")
            fp.flush()

        self.__test_data_labels_path = 'test/' + self.__name + "_labels"
        with open(self.__test_data_labels_path, 'w') as fp:
            for trace in traces:
                for index, log in trace.iterrows():
                    fp.write("%s " % str(1 if log['is_anomaly'] else 0))
                fp.write("\n")
            fp.flush()

    @property
    def name(self):
        return self.__name

    @property
    def num_classes(self):
        return self.__num_classes


def clean_message(message):
    message = re.sub('\n', ' ', message)
    message = replace_body(message)
    message = replace_other(message)
    return message


def replace_other(message):
    patterns = [
        r"[-()\"#/@;:<>{}`+=~|.!?,']",
        '\\d+'
    ]

    for pattern in patterns:
        message = re.sub(pattern, '', message)

    return message


def replace_body(message):
    message = re.sub('Method .+ (?P<body>{(.+)})', partial(replace_closure, 'body', ''), message)
    message = re.sub('Method .+ (?P<body>\[(.+)\])', partial(replace_closure, 'body', ''), message)
    return message


def replace_closure(subgroup, replacement, message):
    group = message.group(subgroup)

    if group not in [None, '']:
        start = message.start(subgroup)
        end = message.end(subgroup)
        temp = message.group()[:start] + replacement + message.group()[end:]
        return temp

    return message
