import sys


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout, write_log=True):
        self.write_log = write_log
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        if self.write_log:
            self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger()