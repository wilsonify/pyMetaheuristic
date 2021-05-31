import os


def rando():
    return int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
