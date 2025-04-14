import numpy as np
import pickle
import random
from ..const import span

tknum = '01'
fp_tk = f'saved/tokens/tokenizer-{tknum}.pickle'

class Tokenizer:
    def onehot(self, t):
        ret = np.zeros((len(t), span))
        ret[range(len(t)), t] = 1
        return ret
    
    def encode(self, t):
        t = self.translate(t)
        t = self.compress(t)
        t = self.onehot(t)
        return t

    def decode(self, t):
        t = [np.random.choice(range(span), 1, i) for i in t]
        t = self.decompress(t)
        t = self.tostr(t)
        return t

class Simple(Tokenizer):
    enc = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    def compress(self, t): return t
    def decompress(self, t): return t
    
    def translate(self, t):
        return [self.enc.index(i) for i in t]
    
    def tostr(self, t):
        return [self.enc[i] for i in t]

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
        print('loaded')
    
    def save(self, fp):
        with open(fp, 'wb') as f:
            pickle.dump(self.bloc, f)
    
    def translate(self, t):
        t = t['text']
        return t.encode('utf-8')
    
    def tostr(self, t):
        return bytes(t).decode('utf-8', errors='replace')
    
    def train(self, data):
        dt = [
            self.translate(t)
            for _,t in zip(range(100), data)
        ]
        pair = {}
        def incr(p): pair[p] = pair.get(p,0)+1
        self.bloc = [[x] for x in range(256)]
        for t in dt:
            for a,b in zip(t, t[1:]):
                incr((a,b))
        while len(self.bloc) < span:
            if len(self.bloc)%32==0: print(len(self.bloc))
            p1,p2 = p = max(pair.keys(), key=pair.get)
            pair[p] = 0
            rep = len(self.bloc)
            self.bloc.append(self.bloc[p1]+self.bloc[p2])
            for i in range(len(dt)):
                l = dt[i]
                j = 0
                while j<len(l):
                    if l[j:j+2]==list(p):
                        l = l[:j] + [rep] + l[j+2:]
                        if j>0:
                            incr((l[j],rep))
                        if j+1<len(l):
                            incr((rep,l[j+1]))
                    j += 1
                dt[i] = l
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
        #print('c factor:', len(t)/len(o))
        return o
    
    def decompress(self, arr):
        print(self.bloc)
        out = []
        for i in arr:
            out += self.bloc[i[0]]
        #print('c factor:', len(out)/len(arr))
        return out

enc = BPE()
if __name__=='__main__':
    import datasets as ds
    dst = ds.load_dataset(
        path="openwebtext",
        streaming=True,
        split="train",
    )
    enc.train(dst)
    enc.save(fp_tk)
else:
    enc.load(fp_tk)
assert len(enc.bloc)==span

