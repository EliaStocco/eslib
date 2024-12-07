import os
import time
import re
from datetime import datetime

class Logger:
    def __init__(self, log_file: str = None, debug:bool=False, warning:bool=True) -> None:
        """
        Set up a custom logger for the class.

        Args:
            log_file (str): Path to the log file. If None, logs to standard output (stdout).
        """
        self.log_file = log_file
        self._debug = debug
        self._warning = warning
        
        flags = extract_flags(log_file)
        if flags["debug"] is not None:
            self._debug = flags["debug"]
        if flags["warning"] is not None:
            self._warning = flags["warning"]

    def _write_log(self, premessage:str, message: str) -> None:
        """
        Writes the log message to the desired output (file or stdout).

        Args:
            message (str): The log message to write.
        """
        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Format the log message
        log_message = f"{timestamp} - {premessage:<7}: {message:<50}"

        # Write to the log file or stdout
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_message + '\n')
        else:
            print(log_message)

    def info(self, message: str) -> None:
        """Log an info message."""
        self._write_log("INFO",message)

    def debug(self, message: str) -> None:
        """Log a debug message."""
        if self._debug:
            self._write_log("DEBUG",message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self._write_log("ERROR",message)

    def warning(self, message: str) -> None:
        if self._warning:
            self._write_log("WARNING",message)

def check_exit(logger:Logger=None) -> None:
    """
    Check if an 'EXIT' file exists in the current directory. 
    If it exists, terminate the program.
    """
    if os.path.exists("EXIT"):
        if logger is not None:
            logger.info("EXIT file detected. Kill this process.")
        time.sleep(1)
        exit(0)
        
def extract_flags(input_string: str) -> dict:
    flags = {"debug": None, "warning": None}
    if input_string is None or len(input_string) == 0 :
        return flags
    # Define a regex pattern to find both 'debug' and 'warning' and their values (True or False)
    pattern = r"(debug|warning)=(True|False)"
    
    # Search for the pattern in the string
    matches = re.findall(pattern, input_string)
    
    # Initialize a dictionary to store the boolean values for debug and warning
    
    
    # Populate the dictionary with the values from the matches
    for match in matches:
        flag, value = match
        flags[flag] = value == "True"
    
    return flags