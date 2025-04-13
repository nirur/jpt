import sys
import src

if len(sys.argv)>1 and sys.argv[1]=='test':
    src.test()
else:
    src.train()

