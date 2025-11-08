# Standard imports
import sys
from typing import Optional, ContextManager

class F2kcli(ContextManager):
    """
    Fortran 200x Command Line Interface (Python version)
    Provides methods to access command-line arguments in a way similar to the Fortran F2KCLI module.

    Methods
    -------
    getCommand(command: Optional[list] = None, length: Optional[list] = None, status: Optional[list] = None) -> None
        Returns the entire command by which the program was invoked.
    commandArgumentCount() -> int
        Returns the number of command arguments (excluding the program name).
    getCommandArgument(number: int, value: Optional[list] = None, length: Optional[list] = None, status: Optional[list] = None) -> None
        Returns a specific command argument by index.
    """
    def __enter__(self) -> 'F2kcli':
        # Fortran context: could pin memory, set thread affinity, etc. Not needed in Python.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        # Fortran context: cleanup or release resources if needed. Not needed in Python.
        return None

    @staticmethod
    def getCommand(command: Optional[list] = None, length: Optional[list] = None, status: Optional[list] = None) -> None:
        """
        Returns the entire command by which the program was invoked.

        Parameters
        ----------
        command : list, optional
            A single-element list to store the command string.
        length : list, optional
            A single-element list to store the length of the command string.
        status : list, optional
            A single-element list to store the status code (0 for success).
        """
        cmd = ' '.join(sys.argv)
        if command is not None:
            command[0] = cmd
        if length is not None:
            length[0] = len(cmd)
        if status is not None:
            status[0] = 0

    @staticmethod
    def commandArgumentCount() -> int:
        """
        Returns the number of command arguments (excluding the program name).

        Returns
        -------
        int
            Number of command-line arguments (not counting the program name).
        """
        return max(0, len(sys.argv) - 1)

    @staticmethod
    def getCommandArgument(number: int, value: Optional[list] = None, length: Optional[list] = None, status: Optional[list] = None) -> None:
        """
        Returns a command argument by index.

        Parameters
        ----------
        number : int
            The argument index (0 for program name, 1..N for arguments).
        value : list, optional
            A single-element list to store the argument value.
        length : list, optional
            A single-element list to store the length of the argument value.
        status : list, optional
            A single-element list to store the status code (0 for success, 1 for out-of-range).
        """
        # Fortran: 0 is program name, 1..N are arguments
        if number < 0 or number >= len(sys.argv):
            if value is not None:
                value[0] = ''
            if length is not None:
                length[0] = 0
            if status is not None:
                status[0] = 1  # Out of range
            return
        arg = sys.argv[number]
        if value is not None:
            value[0] = arg
        if length is not None:
            length[0] = len(arg)
        if status is not None:
            status[0] = 0