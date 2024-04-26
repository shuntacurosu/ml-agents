
from mlagents.trainers.learn import parse_command_line, run_cli

from Logger import Logger
logger = Logger(__name__, loglevel=Logger.INFO)

def main():
    run_cli(parse_command_line())


# For python debugger to directly run this script
if __name__ == "__main__":
    main()
