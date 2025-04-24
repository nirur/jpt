from jax import numpy as jnp
import numpy as np
import pickle
import random
import time
#import heapq
import tiktoken
#from ..const import span

tknum = '01'
fp_tk = f'saved/tokens/tokenizer-{tknum}.pickle'

class Tokenizer:
    def onehot(self, t):
        ret = np.zeros((len(t), self.span))
        ret[list(range(len(t))), t] = 1
        return ret
    
    def encode(self, t):
        t = self.translate(t)
        t = self.compress(t)
        t = self.onehot(t)
        return t
    
    def sample(self, t):
        #return [np.random.choice(1, i[:-1])[0,0] for i in t]
        return [random.choices(range(self.span), i)[0] for i in t]
    
    def decode(self, t):
        t = self.sample(t)
        t = self.decompress(t)
        t = self.tostr(t)
        return t

class Shakespeare(Tokenizer):
    enc = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    span = len(enc)
    def compress(self, t): return t
    def decompress(self, t): return t
    
    def translate(self, t):
        return [self.enc.index(i) for i in t]
    
    def tostr(self, t):
        return ''.join([self.enc[i] for i in t])

class Ascii(Tokenizer):
    span = 128
    def compress(self, t): return t
    def decompress(self, t): return t
    
    def translate(self, t):
        return [min(max(ord(i),0),self.span-1) for i in t]
    
    def tostr(self, t):
        #print(t)
        return ''.join([chr(i) for i in t])

class BPE(Tokenizer):
    def b2d(self):
        self.df = {}
        self.mln = 0
        for i,tp in enumerate(self.bloc):
            self.df[*tp] = i
            self.mln = max(self.mln, len(tp))
    
    def load(self, fp):
        with open(fp, 'rb') as f:
            self.bloc = pickle.load(f)
        self.b2d()
        self.span = len(self.bloc)
        print('loaded')
    
    def save(self, fp):
        with open(fp, 'wb') as f:
            pickle.dump(self.bloc, f)
    
    def translate(self, t):
        return t.encode('utf-8')
    
    def tostr(self, t):
        return bytes(t).decode('utf-8', errors='replace')
    
    def train(self, data):
        tme = time.time()
        dt = [
            list(self.translate(t['text']))
            for _,t in zip(range(200), data)
        ]
        pair = {}
        def incr(p): pair[p] = pair.get(p,0)+1
        self.bloc = [[x] for x in range(256)]
        for t in dt:
            for a,b in zip(t, t[1:]):
                incr((a,b))
        while len(self.bloc) < self.span:
            if len(self.bloc)%32==0:
                print(len(self.bloc), time.time()-tme)
                tme = time.time()
            p1,p2 = p = max(pair.keys(), key=pair.get)
            pair[p] = 0
            lp = list(p)
            rep = len(self.bloc)
            self.bloc.append(self.bloc[p1]+self.bloc[p2])
            for i in range(len(dt)):
                l = dt[i]
                #print('bln:', len(l))
                j = 0
                lah = 0 # lookahead
                while j+lah<len(l):
                    if l[j+lah:j+lah+2]==list(p):
                        l[j] = rep
                        lah += 1
                        if j+lah>0:
                            incr((l[j-1],rep))
                        if j+lah+1<len(l):
                            incr((rep,l[j+lah+1]))
                    else:
                        l[j] = l[j+lah]
                    j += 1
                dt[i] = l[:j]
                #print('aln:', j)
        self.span = len(self.bloc)
        print(self.bloc)
        self.b2d()
    
    def compress(self, t):
        o = []
        j = 0
        l = self.mln
        while j<len(t):
            for l2 in range(l, 0, -1):
                t2 = tuple(t[j:j+l2])
                if t2 in self.df:
                    o.append(self.df[t2])
                    j += l2
                    break
        print('cf:', len(t)/len(o))
        return o
    
    def decompress(self, arr):
        out = []
        for i in arr:
            out += self.bloc[i]
        #print('c factor:', len(out)/len(arr))
        return out

class TKWrapper(Tokenizer):
    def __init__(self, ec='p50k_base'):
        self.ec = tiktoken.get_encoding('p50k_base')
        self.span = 50257
    def translate(self, t): return t
    def tostr(self, t): return t
    def compress(self, t): return self.ec.encode(t)
    def decompress(self, t): return self.ec.decode(t)

def trn():
    import datasets as ds
    from .configs import conf
    dst = ds.load_dataset(
        **conf[0]
    )
    enc.train(dst)
    enc.save(fp_tk)

if __name__=='__main__':
    trn()
else:
    enc = BPE()
    enc.load(fp_tk)
    #pass
    #enc = TKWrapper()
    ## verify the span matches with TKWrapper
    #enc = Shakespeare()

