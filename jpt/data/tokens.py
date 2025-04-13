import numpy as np
import pickle
import random

span = 512
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
        t = [random.choices(range(span), i) for i in t]
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
    def load(self, fp):
        with open(fp, 'rb') as f:
            self.bloc = pickle.load(f)
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
            for i,t in zip(range(100), data)
        ]
        pair = {}
        self.bloc = [[x] for x in range(256)]
        for t in dt:
            for a,b in zip(t, t[1:]):
                pair[a,b] = pair.get((a,b), 0)+1
        while len(self.bloc) < span:
            p1,p2 = p = max(pair.keys(), key=pair.get)
            pair[p] = 0
            rep = len(self.bloc)
            self.bloc.append(self.bloc[p1]+self.bloc[p2])
            for i in range(len(dt)):
                l = dt[i]
                j = 0
                while j<len(l):
                    if l[j]==p1 and l[j+1]==p2:
                        l = l[:j] + [rep] + l[j+2:]
                    j += 1
                dt[i] = l
    
    def compress(self, t):
        o = []
        j = 0
        while j<len(t):
            for i,b in enumerate(self.bloc[:255:-1]):
                l = len(b)
                if t[j:j+l]==i:
                    o.append(span-1-i)
                    j += l
                    break
            else:
                o.append(t[j])
                j+=1
        return o
    
    def decompress(self, arr):
        for i,bk in enumerate(self.bloc):
            j = 0
            while j<len(arr):
                if arr[j]==bk:
                    arr = arr[:j] + bk + arr[j+1:]
                j+=1
        return arr

enc = BPE()
enc.load(fp_tk)
assert len(enc.bloc)==span
