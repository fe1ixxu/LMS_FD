import fileinput

for s in fileinput.input():
    s = s.rstrip().split(' ')[:512]
    print(' '.join(s))