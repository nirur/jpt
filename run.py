import sys

if len(sys.argv)>1:
    if sys.argv[1]=='test':
        from jpt import test
    else:
        from jpt.data import tokens
        tokens.trn()
else:
    from jpt import train

