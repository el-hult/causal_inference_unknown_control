import datetime
import argparse

import numpy as np


def printt(*args, t0: datetime.datetime = None, **kwds):
    """Print, but include the current time, and optionally time elapsed
    
    Args:
        As normal for print, but with keyword t0, the timestamp to compute duration from

    Returns:
        Current timestamp
    """
    t1 = datetime.datetime.now()
    if t0 is not None:
        d = t1 - t0
        ts = d.total_seconds()
        h = ts // 3600
        m = (ts // 60) % 60
        s = ts % 60
        sgn = " " if ts > 0 else "-"
        hours = f"{h:03.0f}h" if h > 0 else "    "
        minutes = f"{m:02.0f}m" if m > 0 else "   "
        seconds = f"{s:02.1f}s" if s > 0 else "    "
        duration = sgn + hours + minutes + seconds
    else:
        duration = "            "
    print(t1.isoformat()[:19], duration, *args, **kwds)
    return t1


def cross_moment_4(data):
    """Returns all cross 4th moments of a dataset
    The computation does NOT exploit the symmetries, so there is a lot
    of cleverness to be done here..."""
    n = data.shape[0]
    return np.einsum("ij,ik,il,im->jklm", data, data, data, data) / n


class CheckUniqueStore(argparse.Action):
    """Checks that the list of arguments contains no duplicates, then stores"""

    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) > len(set(values)):
            raise argparse.ArgumentError(
                self,
                "You cannot specify the same value multiple times. "
                + f"You provided {values}",
            )
        setattr(namespace, self.dest, values)


class RandGraphSpec:
    def __init__(self, raw_val):
        """Get the number of nodes and the edge multiplier from a string
        
        RandGraphSpec("d,k")
        d is the number of nodes
        k is how many edges per node, on average
        """
        d, k = raw_val.split(",")
        self.d = int(d)
        self.k = float(k)

    def __repr__(self) -> str:
        return f"RandGraphSpec('{self.d},{self.k}')"

    def __str__(self) -> str:
        return f'"{self.d},{self.k}"'
