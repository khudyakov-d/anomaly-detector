import pandas as pd
import pm4py

from pm4py.objects.conversion.log import converter as log_converter


class ActivityProcess:
    def __init__(self, name, logs=None):
        self.__name = name
        self.__train_traces = []
        self.__test_traces = []
        if logs is not None:
            self.update(logs)

    def update(self, logs):
        self.__update_dfg(logs)
        self.__update_process_tree(logs)
        self.__update_train_traces(logs)

    def __update_dfg(self, logs):
        dfg, start_activities, end_activities = pm4py.discover_dfg(logs)
        self.__dfg = dfg
        self.__start_activities = start_activities
        self.__end_activities = end_activities

    def __update_process_tree(self, logs):
        self.__process_tree = pm4py.discover_process_tree_inductive(logs, noise_threshold=0.15)

    def __update_train_traces(self, logs):
        try:
            alignments = pm4py.conformance_diagnostics_alignments(logs, self.__process_tree)

            traces = set()
            for i in range(len(alignments)):
                if alignments[i]['fitness'] == 1:
                    traces.add(logs[i].attributes['concept:name'])
            self.__train_traces = traces
        except:
            self.__train_traces = (log.attributes['concept:name'] for log in logs)

    def view_process_tree(self):
        pm4py.view_process_tree(self.__process_tree)

    def view_dfg(self):
        pm4py.view_dfg(self.__dfg, self.__start_activities, self.__end_activities)

    @property
    def name(self):
        return self.__name

    @property
    def train_traces(self):
        return self.__train_traces

    @property
    def test_traces(self):
        return self.__test_traces

    @test_traces.setter
    def test_traces(self, logs):
        for log in logs:
            self.__test_traces.append(log.attributes['concept:name'])


def to_logs(traces):
    logs = []

    for index, trace in traces.iterrows():
        for log in trace['logs']:
            log['traceId'] = trace['traceId']
            log['user'] = trace['user']
            logs.append(log)

    df = pd.DataFrame.from_records(logs)
    df.rename(columns={'traceId': 'case:concept:name',
                       'user': 'case:clientID',
                       'method': 'concept:name',
                       'time': 'time:timestamp',
                       'applicationName': 'org:resource'},
              inplace=True)

    logs = log_converter.apply(df)

    return logs


def init_activities(traces):
    logs = to_logs(traces)
    start_activities = pm4py.get_start_activities(logs)

    activities = []

    for activity, count in start_activities.items():
        activity_logs = pm4py.filter_start_activities(logs, [activity])
        activities.append(ActivityProcess(name=activity, logs=activity_logs))

    return activities


def split_test_logs(traces, activities_processes):
    logs = to_logs(traces)

    for activity in activities_processes:
        activity_logs = pm4py.filter_start_activities(logs, [activity.name])
        activity.test_traces = activity_logs
