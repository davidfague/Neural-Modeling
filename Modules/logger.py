from datetime import datetime
import os
import psutil
from multiprocessing import current_process
import csv

class Logger:

    def __init__(self, path: str = None):
        if path is None:
            self.path = None
        else:
            self.path = os.path.join(path, "log.txt")
            self.runtime_path = os.path.join(path, "runtimes.csv")
        
    def set_path(self, path: str):
        self.path = os.path.join(path, "log.txt")
        self.runtime_path = os.path.join(path, "runtimes.csv")

    def log(self, msg: str):
        if self.path is None:
            print(f"({datetime.now()})-[PID: {current_process().pid}]–[INFO]: {msg}")
        else:
            print(f"({datetime.now()})-[PID: {current_process().pid}]–[INFO]: {msg}", file = open(self.path, "a"))

    def log_warining(self, msg: str):
        if self.path is None:
            print(f"({datetime.now()})-[PID: {current_process().pid}]–[WARNING]: {msg}")
        else:
            print(f"({datetime.now()})-[PID: {current_process().pid}]–[WARNING]: {msg}", file = open(self.path, "a"))

    def log_step(self, step: int):
        if self.path is None:
            print(f"({datetime.now()})-[PID: {current_process().pid}]–[STEP]: {step}")
        else:
            print(f"({datetime.now()})-[PID: {current_process().pid}]–[STEP]: {step}", file = open(self.path, "a"))

    def log_memory(self):
        memory = psutil.virtual_memory()
        if self.path is None:
            print(f"({datetime.now()})-[PID: {current_process().pid}]–[MEMORY]: available {round(memory.available * 1e-9, 2)}, used: {memory.percent}% of total.")
        else:
            print(f"({datetime.now()})-[PID: {current_process().pid}]–[MEMORY]: available {round(memory.available * 1e-9, 2)}, used: {memory.percent}% of total.", 
                  file = open(self.path, "a"))

    def log_runtime(self, module_name, function_name, runtime):
        file_exists = os.path.exists(self.runtime_path)
        file_empty = not file_exists or os.path.getsize(self.runtime_path) == 0

        with open(self.runtime_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if file_empty:
                writer.writerow(["timestamp", "module", "function", "runtime"])
            writer.writerow([datetime.now(), module_name, function_name, runtime])
