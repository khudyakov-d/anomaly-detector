import pandas as pd

from pandas import DataFrame


def to_traces(logs):
    raw_traces = [y for x, y in logs.groupby(['traceId'])]

    traces = []
    for trace in raw_traces:
        t = convert_trace(trace)
        if t is not None:
            traces.append(t)

    return DataFrame(traces)


def sort_span(event):
    return event['@timestamp']


def convert_trace(trace):
    spans = trace.groupby(['spanId'])
    spans = [y for x, y in spans]

    trace_start = trace.iloc[0]
    trace_end = trace.iloc[len(trace) - 1]
    trace = {
        'traceId': trace_start['traceId'],
        'startTime': pd.to_datetime(trace_start['@timestamp']),
        'endTime': pd.to_datetime(trace_end['@timestamp']),
        'user': trace_start["username"] or "system",
        'logs': []
    }

    if trace['user'] == 'avtotest_TE':
        return None

    for span in spans:
        logs = trace['logs']

        if span.iloc[0]['METHOD'] is None:
            logs.append('Unknown')
        else:
            span_start = span.iloc[0]
            logs.append({
                'time': pd.to_datetime(span_start['@timestamp']),
                #'method': str(span_start['applicationName']) + "_" + str(span_start['METHOD']),
                'method': str(span_start['METHOD']),
                'spanId': span_start['spanId'],
                #'applicationName': span_start['applicationName']
            })

        logs.sort(key=sort_event)

    return trace


def sort_event(event):
    return event['time']
