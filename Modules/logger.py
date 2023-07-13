import datetime, time
import os
import psutil

class Logger:

    def __init__(self, output_dir):
        # Create the output file
        log_file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
        self.output_file = os.path.join(output_dir, log_file_name)

        with open(self.output_file, "w") as file:
            file.writelines(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Logger started.")

        self.time_from_the_last_call = time.time()

    def log(self, message):
        with open(self.output_file, "a") as file:
            file.writelines(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (f{time.time() - self.time_from_the_last_call} sec elapsed): {message}")
        self.time_from_the_last_call = time.time()

    # For any methods below, use self.log(message)
    # ----------
    def log_memory(self):
        memory = psutil.virtual_memory()
        message = f"available: {round(memory['available'] * 1e-9, 2)} Gb, {memory['percent']}% of total."
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
        ______██████████████████.•°*”˜҈.•°*”˜҈.
        '''
        self.log(message)

