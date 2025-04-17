import sys

if len(sys.argv)>1 and sys.argv[1]=='test':
    from jpt import test
else:
    from jpt import train

