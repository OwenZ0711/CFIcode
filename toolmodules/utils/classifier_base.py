"""
base of classify
"""
from abc import ABC
import pandas as pd


class BaseClassifier(ABC):
    """
    base of classifier
    """
    def __init__(self, interval_map, close_on="right"):
        close_on_para = ("left", "right")
        if close_on not in close_on_para:
            raise RuntimeError(f"unsupported parameter close_on of value {close_on}, avaliable para {close_on_para}")

        self.__interval_map = interval_map
        self.__interval_list, self.__label_list = self.__generate_label_list()
        self.__close_on = close_on

    def get_interval_map(self):
        """
        get interval_map
        """
        return self.__interval_map

    def get_interval_list(self):
        """
        get value list of classify threshold
        """
        return self.__interval_list

    def get_label_list(self):
        """
        get label list of classify threshold
        """
        return self.__label_list

    def get_classifier_df(self):
        """
        get classifier_df
        """
        label_list = self.get_label_list()
        interval_list = self.get_interval_list()

        classifier_df = pd.DataFrame(
                {
                    "label": label_list,
                    "interval": interval_list,
                    "interval_flag": True
                }
            )
        return classifier_df

    def _do_classify(self, interval: pd.Series):
        """
        do classify dataframe according to value in interval;
        dataframe, interval are of the same length
        close_on default left, can be right
        """
        interval_df = pd.DataFrame({"interval": interval})
        classifier_df = self.get_classifier_df()

        if self.__close_on == "left":
            classified_df = pd.concat([classifier_df, interval_df], axis=0, sort=False)
        elif self.__close_on == "right":
            classified_df = pd.concat([interval_df, classifier_df], axis=0, sort=False)

        classified_df.sort_values("interval", inplace=True, kind="mergesort")
        classified_df["label"].fillna(method="bfill", inplace=True)
        classified_df["interval_flag"].fillna(False, inplace=True)
        classified_df = classified_df[~classified_df["interval_flag"]]
        return classified_df["label"]

    def __generate_label_list(self):
        """
        generate label list
        """
        label_list = sorted(list(self.__interval_map.keys()))
        interval_list = [self.__interval_map[key] for key in label_list]
        return interval_list, label_list

    def classify(self, interval: pd.Series):
        """
        classify dataframe
        """
        return self._do_classify(interval)


class GeneralClassifier(BaseClassifier):
    """
    general classifier
    """
