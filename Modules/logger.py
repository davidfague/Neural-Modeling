import datetime, time
import os
import psutil

class Logger:

    def __init__(self, output_dir, active = True, name = "General"):
        # Create the output file
        self.log_file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{name}.txt"
        self.output_file = os.path.join(output_dir, self.log_file_name)

        with open(self.output_file, "w") as file:
            file.writelines(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Logger {name} started.\n")

        self.time_from_the_last_call = time.time()
        self.active = active

    def log(self, message):
        if self.active:
            with open(self.output_file, "a") as file:
                file.writelines(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ({time.time() - self.time_from_the_last_call} sec elapsed): {message}\n")
            self.time_from_the_last_call = time.time()

    # For any methods below, use self.log(message)
    # ----------
    def log_memory(self):
        memory = psutil.virtual_memory()
        message = f"available: {round(memory.available * 1e-9, 2)} Gb, used: {memory.percent}% of total."
        self.log(message)
    
    def log_section_start(self, section_name):
        message = f"Starting {section_name}."
        self.log(message)

    def log_section_end(self, section_name):
        message = f"Finished running {section_name}."
        self.log(message)

    def log_snail(self):
        message = '''\n
        __●__ ●
        _ █___█
        __ █__ █_
        __ █__ █
        __ ███____________█████
        _█▒░░█_________██▓▒▒▓██ ☆
        █▒░●░░█___ ██▓▒██▓▒▒▓█   ★
        █░█▒░░██_ ██▓▒██▓▒░▒▓█
        _██▒░░██ ██▓▒░██▓▒░▒▓█    ★
        ____█▒░██ ██▓▒░░ ████▓█
        ___█▒░██__██▓▓▒▒░░░██  ★★
        ____█▒░██___████████████
        _____█▒░█▒▒▒▒▒▒▒▒▒▒▒▒█
        ______██████████████████.•°*”˜҈.•°*”˜҈.\n
        '''
        self.log(message)

