"""
Class used to record prediction probabilities to a log file
"""


import csv
from pathlib import Path


import constants as c


class Recorder:
    """
    Records prediction probabilities to file in the __init__
    """
    def __init__(self, output: str = None):
        if output is None:
            self._is_recording = False
            return
        
        self._is_recording = True
        self.output = Path(output)

        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.unlink(missing_ok=True)
        
        # Add header
        header = ["timestamp"] + ["prob_" + label for label in sorted(c.LABEL_TO_COUNT.keys())]
        with open(self.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(header)


    def record(self, info: dict, sep=";"):
        """
        Records the prediction probabilities to the csv file.
        """
        if not self._is_recording:
            return

        timestamp = info.pop("timestamp")
        
        # Make sure numbers will be in the correct order
        if not sorted(info.keys()) == sorted(c.LABEL_TO_COUNT.keys()):
            raise ValueError(f"Keys in info do not match c.LABEL_TO_COUNT.keys(). Keys in info: {sorted(info.keys())}. Keys in c.LABEL_TO_COUNT.keys(): {sorted(c.LABEL_TO_COUNT.keys())}")
            
        info = [info[k] for k in sorted(info.keys())]

        with open(self.output, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=sep)
            writer.writerow([timestamp] + info)