# import pandas as pd
import numpy as np


# class MetricTracker_old:
#     """
#     Class to aggregate metrics from many batches.
#     """

#     def __init__(self, *keys, writer=None):
#         """
#         Args:
#             *keys (list[str]): list (as positional arguments) of metric
#                 names (may include the names of losses)
#         """
#         self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
#         self.reset()

#     def reset(self):
#         """
#         Reset all metrics after epoch end.
#         """
#         for col in self._data.columns:
#             self._data[col].values[:] = 0

#     def update(self, key, value, n=1):
#         """
#         Update metrics DataFrame with new value.

#         Args:
#             key (str): metric name.
#             value (float): metric value on the batch.
#             n (int): how many times to count this value.
#         """
#         self._data.loc[key, "total"] += value * n
#         self._data.loc[key, "counts"] += n
#         self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

#     def avg(self, key):
#         """
#         Return average value for a given metric.

#         Args:
#             key (str): metric name.
#         Returns:
#             average_value (float): average value for the metric.
#         """
#         return self._data.average[key]

#     def result(self):
#         """
#         Return average value of each metric.

#         Returns:
#             average_metrics (dict): dict, containing average metrics
#                 for each metric name.
#         """
#         return dict(self._data.average)

#     def keys(self):
#         """
#         Return all metric names defined in the MetricTracker.

#         Returns:
#             metric_keys (Index): all metric names in the table.
#         """
#         return self._data.total.keys()

from collections import defaultdict
class MetricTracker:
    """
    Class to aggregate metrics from many batches.
    """

    def __init__(self, *keys, writer=None):
        """
        Args:
            *keys (list[str]): list (as positional arguments) of metric
                names (may include the names of losses)
        """
        self._data = defaultdict(list)
        self.reset()

    def reset(self):
        """
        Reset all metrics after epoch end.
        """
        for key in self._data.keys():
            self._data[key] = []

    def update(self, key, value):
        """
        Update metrics DataFrame with new value.

        Args:
            key (str): metric name.
            value (float): metric value on the batch.
            n (int): how many times to count this value.
        """
        self._data[key].append(value)

    def avg(self, key):
        """
        Return average value for a given metric.

        Args:
            key (str): metric name.
        Returns:
            average_value (float): average value for the metric.
        """
        return np.mean(self._data[key]) if len(self._data[key]) > 0 else 0

    def result(self):
        """
        Return average value of each metric.

        Returns:
            average_metrics (dict): dict, containing average metrics
                for each metric name.
        """
        result = {
            k: np.mean(v) if len(v) > 0 else 0 for k, v in self._data.items()
        }
        return result

    def keys(self):
        """
        Return all metric names defined in the MetricTracker.

        Returns:
            metric_keys (Index): all metric names in the table.
        """
        return self._data.keys()
