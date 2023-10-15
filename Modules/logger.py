import datetime
import os
import psutil

class Logger:

    def __init__(self, path: str = None):
        if path is not None:
            self.path = os.path.join(path, "log.txt")
        else:
            self.path = path

    def log(self, msg: str):
        print(f"({datetime.now()})-[INFO]: {msg}", file = open(self.path, "a"))

    def log_memory(self):
        memory = psutil.virtual_memory()
        print(f"({datetime.now()})-[MEMORY]: available {round(memory.available * 1e-9, 2)}, used: {memory.percent}% of total.", 
              file = open(self.path, "a"))


