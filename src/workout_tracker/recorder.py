import csv
from pathlib import Path


import constants as c


class Recorder:

    def __init__(self, output: str = None):
        if output is None:
            self._is_recording = False
            return
        
        self._is_recording = True
        self.output = Path(output)

        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.unlink(missing_ok=True)
            
        with open(self.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["timestamp"] + ["prob_" + label for label in sorted(c.LABEL_TO_COUNT.keys())])


    def record(self, info: dict, sep=";"):
        if not self._is_recording:
            return

        timestamp = info.pop("timestamp")
        assert sorted(info.keys()) == sorted(c.LABEL_TO_COUNT.keys())
        info = [info[k] for k in sorted(info.keys())]

        with open(self.output, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=sep)
            writer.writerow([timestamp] + info)