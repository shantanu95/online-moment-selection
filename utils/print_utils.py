import os
from typing import Optional


def print_and_log(s: str, filepath: Optional[str] = None):
    print(s)
    if filepath is not None:
        os.system("echo '%s' >> %s" % (s, filepath))
