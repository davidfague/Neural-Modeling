from datetime import datetime
import os
import psutil
from multiprocessing import current_process
import csv
import warnings
from typing import Optional, Any

class Logger:

    def __init__(self, path: str = None):
        self._timers = {}
        if path is None:
            self.path = None
            self.runtime_path = None
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

    def start_timer(self, timer_name: str = "default"):
        """Start a named timer. Can have multiple timers running simultaneously."""
        if not hasattr(self, '_timers'):
            self._timers = {}
        import time
        self._timers[timer_name] = time.time()
    
    def log_runtime(self, module_name, function_name, runtime=None, timer_name: str = "default"):
        """
        Log runtime to CSV file.
        
        Args:
            module_name: Name of the module/script
            function_name: Name of the function/operation
            runtime: Optional explicit runtime in seconds. If None, uses timer started with start_timer()
            timer_name: Name of timer to use if runtime not provided (default: "default")
        """
        if runtime is None:
            if not hasattr(self, '_timers') or timer_name not in self._timers:
                raise ValueError(f"Timer '{timer_name}' was not started. Call start_timer('{timer_name}') first or provide explicit runtime.")
            import time
            runtime = time.time() - self._timers[timer_name]
            # Clear the timer after use
            del self._timers[timer_name]
        
        # If no runtime_path set (temporary logger), skip writing to CSV
        if self.runtime_path is None:
            return
        
        file_exists = os.path.exists(self.runtime_path)
        file_empty = not file_exists or os.path.getsize(self.runtime_path) == 0

        with open(self.runtime_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if file_empty:
                writer.writerow(["timestamp", "module", "function", "runtime_seconds", "runtime_minutes"])
            writer.writerow([datetime.now(), module_name, function_name, runtime, round(runtime / 60, 2)])

def log_or_warn(msg: str, logger: Optional[Any] = None, *, stacklevel: int = 2):
    """
    If a logger is provided, log the message.
    Otherwise, fall back to warnings.warn, pointing at the caller.
    """
    if logger is not None:
        logger.log(msg)
    else:
        # stacklevel=2 makes the warning appear at the caller of this function instead of inside log_or_warn().
        warnings.warn(msg, stacklevel=stacklevel)