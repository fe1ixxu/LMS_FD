#!/usr/bin/env python
import struct
import sys

assert len(sys.argv) == 2

with open(sys.argv[1], "rb") as f:
    assert f.read(9) == b"MMIDIDX\x00\x00"
    _ = f.read(9)
    print(struct.unpack("<Q", f.read(8))[0])
