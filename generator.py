import enum
import itertools
import random
import uuid
from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import pandas as pd
from transformers import pipeline

random.seed(50)

INDETERMINACY_PROBABILITY = 0

logs_dict = {}


class SpanDeviation(float, enum.Enum):
    reorder = 100
    cycle = 0

    @classmethod
    def make_choice(cls):
        weights: [float] = [e.value for e in SpanDeviation]
        return random.choices(list(SpanDeviation), weights=weights, k=1)[0]


class Log:
    def __init__(self, key: str, span_id: str):
        self.__key = key
        self.__span_id = span_id

    @property
    def key(self):
        return self.__key

    @property
    def span_id(self):
        return self.__span_id


class Span:
    def __init__(self, id: str, method: str, logs: [str]):
        self.__id = id
        self.__logs = logs
        self.__method = method

    @property
    def id(self):
        return self.__id

    @property
    def method(self):
        return self.__method

    @property
    def logs(self):
        return self.__logs


def uuid_generator():
    while True:
        yield uuid.uuid4().hex


def generate_sequence(length: int) -> [str]:
    gen = uuid_generator()
    return list(itertools.islice(gen, length))


def split_to_spans(sequence: [str], min_span_length: int, max_span_length: int) -> [Span]:
    if min_span_length < 2:
        raise ValueError("Min length of span must be > 1")

    sequence_length = len(sequence)

    if sequence_length < min_span_length * 2:
        return Span(uuid.uuid4().hex, sequence)

    pos: int = 0
    sections: [int] = []

    while sequence_length - pos > max_span_length:
        pos += random.randint(min_span_length, max_span_length)
        sections.append(pos)
    split = np.split(sequence, sections)

    spans = []
    for i in range(len(split)):
        elem = split[i]
        spans.append(Span(uuid.uuid4().hex, 'M' + str(i), elem.tolist()))

    return spans


def generate_spans(base_logs_sequence_length: int, min_span_length: int, max_span_length: int) -> [Span]:
    sequence_length = generate_sequence(base_logs_sequence_length)
    return split_to_spans(sequence_length, min_span_length, max_span_length)


def generate_trace_tree(spans: [Span]):
    graph = nx.DiGraph()

    for span in spans:
        span_logs = span.logs
        for i in range(len(span_logs)):
            cur_node_name = uuid.uuid4().hex
            if len(graph.nodes) == 0:
                graph.add_node(cur_node_name, logId=span_logs[i], spanId=span.id, method=span.method, normal='true')
                prev_node_name = cur_node_name
            else:
                graph.add_node(cur_node_name, logId=span_logs[i], spanId=span.id, method=span.method, normal='true')
                graph.add_edge(prev_node_name, cur_node_name)
                prev_node_name = cur_node_name

    return graph


def add_deviations_to_tree(graph):
    edges = list(graph.edges())
    i = 1
    while i < len(edges) - 1:
        if graph.nodes[edges[i][1]]['spanId'] != graph.nodes[edges[i + 1][0]]['spanId']:
            continue

        if random.random() < INDETERMINACY_PROBABILITY:
            choice = SpanDeviation.make_choice()

            match choice:
                case SpanDeviation.reorder:

                    id1 = uuid.uuid4().hex
                    id2 = uuid.uuid4().hex

                    graph.add_node(id1, logId=graph.nodes[edges[i][1]]['logId'],
                                   spanId=graph.nodes[edges[i][1]]['spanId'],
                                   method=graph.nodes[edges[i][1]]['method'],
                                   normal='true')
                    graph.add_node(id2, logId=graph.nodes[edges[i][0]]['logId'],
                                   spanId=graph.nodes[edges[i][0]]['spanId'],
                                   method=graph.nodes[edges[i][0]]['method'],
                                   normal='true')

                    graph.add_edge(edges[i - 1][0], id1)
                    graph.add_edge(id1, id2)
                    graph.add_edge(id2, edges[i + 1][1])

                    i = i + 2

                    continue

                case SpanDeviation.cycle:
                    prev_node = edges[i][0]

                    for j in range(random.randint(1, 2)):
                        id = uuid.uuid4().hex
                        graph.add_node(id, logId=graph.nodes[edges[i][0]]['logId'],
                                       spanId=graph.nodes[edges[i][0]]['spanId'],
                                       method=graph.nodes[edges[i][0]]['method'],
                                       normal='true')
                        graph.add_edge(prev_node, id)
                        prev_node = id

                    graph.add_edge(prev_node, edges[i][1])

                    i = i + 1

                    continue

        i = i + 1

    return graph


def add_error_after_first_span(graph):
    edges = list(graph.edges())
    i = len(edges) - 1

    while i >= 0:
        if graph.nodes[edges[i][0]]['spanId'] != graph.nodes[edges[i][1]]['spanId']:
            id1 = uuid.uuid4().hex
            graph.add_node(id1, logId=uuid.uuid4().hex,
                           spanId=graph.nodes[edges[i][1]]['spanId'],
                           method=graph.nodes[edges[i][1]]['method'],
                           normal='false')
            graph.add_edge(edges[i][0], id1)

            id2 = uuid.uuid4().hex
            graph.add_node(id2, logId=uuid.uuid4().hex,
                           spanId=graph.nodes[edges[i][1]]['spanId'],
                           method=graph.nodes[edges[i][1]]['method'],
                           normal='false')
            graph.add_edge(id1, id2)
            break
        i = i - 1

    return graph


def generate_examples_tree(base_logs_sequence_length: int,
                           min_span_length: int,
                           max_span_length: int):
    spans = generate_spans(base_logs_sequence_length, min_span_length, max_span_length)

    trace_tree = generate_trace_tree(spans)
    trace_tree = add_deviations_to_tree(trace_tree)
    trace_tree = add_error_after_first_span(trace_tree)

    return trace_tree


def get_normal_walks(g, num_walks=10):
    walks = list()

    for i in range(num_walks):
        walk = list()
        curr_node = list(g.nodes())[0]
        walk.append(curr_node)

        while True:
            neighbors = []
            curr_node_neighbors = list(g.neighbors(curr_node))
            for neighbor in curr_node_neighbors:
                if g.nodes[neighbor]['normal'] == 'true':
                    neighbors.append(neighbor)

            if len(neighbors) > 0:
                curr_node = random.choice(neighbors)
            else:
                break
            walk.append(curr_node)

        walks.append(walk)

    return walks


def get_anomaly_walk(g):
    walk = list()

    curr_node = list(g.nodes())[0]
    walk.append(curr_node)

    while True:
        neighbors = []
        curr_node_neighbors = list(g.neighbors(curr_node))
        for neighbor in curr_node_neighbors:
            if g.nodes[neighbor]['normal'] == 'false':
                neighbors.append(neighbor)

        if len(neighbors) == 0:
            neighbors = curr_node_neighbors

        if len(neighbors) > 0:
            curr_node = random.choice(neighbors)
        else:
            break
        walk.append(curr_node)

    return walk


def generate_samples(scenario_length,
                     train_samples_count,
                     test_samples_count,
                     train_samples_anomaly_percent=0,
                     test_samples_anomaly_percent=0.3):
    examples = []

    for j in range(scenario_length):
        example = {}
        examples_tree = generate_examples_tree(12, 2, 4)
        example['tree'] = examples_tree
        example['train'] = get_normal_walks(examples_tree, train_samples_count)
        example['test'] = get_normal_walks(examples_tree, test_samples_count)
        examples.append(example)

    time = datetime.now()

    train = generate_train_samples(examples, scenario_length, time, train_samples_anomaly_percent, train_samples_count)
    test = generate_test_samples(examples, scenario_length, test_samples_anomaly_percent, test_samples_count, time)

    return train, test


def generate_test_samples(examples, scenario_length, test_samples_anomaly_percent, test_samples_count, time):
    test_sample_normal_count = test_samples_count - int(test_samples_count * test_samples_anomaly_percent)
    test = []
    cur_time = time

    for i in range(test_sample_normal_count):
        sample = []
        user_id = uuid.uuid4().hex

        for j in range(scenario_length):
            cur_time, logs = transform_trace(examples[j]['tree'], examples[j]['test'][i], user_id, cur_time)
            sample.append(logs)
            cur_time = cur_time + timedelta(milliseconds=random.randint(1000, 3000))

        test.append(sample)

    for i in range(test_sample_normal_count, test_samples_count):
        sample = []
        user_id = uuid.uuid4().hex
        anomaly_span_number = int(scenario_length / 2)

        for j in range(anomaly_span_number):
            cur_time, logs = transform_trace(examples[j]['tree'], examples[j]['test'][i], user_id, cur_time)
            sample.append(logs)
            cur_time = cur_time + timedelta(milliseconds=random.randint(1000, 3000))

        tree = examples[anomaly_span_number]['tree']
        cur_time, logs = transform_trace(tree, get_anomaly_walk(tree), user_id, cur_time)
        sample.append(logs)

        test.append(sample)

    return test


def generate_train_samples(examples, scenario_length, time, train_samples_anomaly_percent, train_samples_count):
    train_sample_normal_count = train_samples_count - int(train_samples_count * train_samples_anomaly_percent)
    train = []
    cur_time = time

    for i in range(train_sample_normal_count):
        sample = []
        user_id = uuid.uuid4().hex

        for j in range(scenario_length):
            cur_time, logs = transform_trace(examples[j]['tree'], examples[j]['train'][i], user_id, cur_time)
            sample.append(logs)
            cur_time = cur_time + timedelta(milliseconds=random.randint(1000, 3000))

        train.append(sample)

    for i in range(train_sample_normal_count, train_samples_count):
        sample = []
        user_id = uuid.uuid4().hex
        anomaly_span_number = random.randint(int(scenario_length / 2), scenario_length - 2)

        for j in range(anomaly_span_number):
            cur_time, logs = transform_trace(examples[j]['tree'], examples[j]['train'][i], user_id, cur_time)
            sample.append(logs)
            cur_time = cur_time + timedelta(milliseconds=random.randint(1000, 3000))

        tree = examples[anomaly_span_number]['tree']
        cur_time, logs = transform_trace(tree, get_anomaly_walk(tree), user_id, cur_time)
        sample.append(logs)

        train.append(sample)

    return train


def transform_trace(tree, trace, user_id, start_time):
    logs = []
    time = start_time
    trace_id = uuid.uuid4().hex

    for logId in trace:
        log = {
            "logId": logId,
            "METHOD": tree.nodes[logId]['method'],
            "spanId": tree.nodes[logId]['spanId'],
            "traceId": trace_id,
            "username": user_id,
            "@timestamp": time,
            "is_anomaly": tree.nodes[logId]['normal'] == "false",
        }
        time = time + timedelta(milliseconds=random.randint(10, 200))
        logs.append(log)

    return time, logs


def data_to_csv(data, filename):
    context = "House"
    gen = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')

    logs = []
    spans = []
    methods = []
    traces = []
    users = []
    times = []
    is_anomaly = []

    for scenario in data:
        np.concatenate(scenario)
        for trace in scenario:
            for log in trace:
                if log['logId'] not in logs_dict:
                    log_text = gen(context, max_length=8, do_sample=True)[0]['generated_text'].replace('\n', ' ')
                    logs_dict[log['logId']] = log_text
                    logs.append(log_text)
                else:
                    logs.append(logs_dict[log['logId']])
                spans.append(log['spanId'])
                methods.append(log['METHOD'])
                traces.append(log['traceId'])
                users.append(log['username'])
                times.append(log['@timestamp'])
                is_anomaly.append(log['is_anomaly'])

    pd.DataFrame({
        'message': logs,
        'spanId': spans,
        'METHOD': methods,
        'traceId': traces,
        'username': users,
        '@timestamp': times,
        'is_anomaly': is_anomaly
    }).to_csv(filename)


if __name__ == '__main__':
    train, test = generate_samples(1, 300, 100, 0, 0.5)
    data_to_csv(train, './data/train.csv')
    data_to_csv(test, './data/test.csv')
