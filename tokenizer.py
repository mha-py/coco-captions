
class Tokenizer:
    def __init__(self, abc=set()):
        for c in ['{', '}', '^', '°']:
            assert c not in abc
        self.abc = set(list(''.join([self.prep(c) for c in abc])))
        self.abc.add('{') # token for beginning of a sequence
        self.abc.add('}') # token for end of a sequence
        
        self.c2t = { c: t for t, c in enumerate(self.abc) }  # char to token
        self.t2c = { t: c for t, c in enumerate(self.abc) }  # token to char
        self.NTOK = len(self.abc)
    
    def tokenize(self, str):
        str = self.prep(str)
        return [ self.c2t[c] for c in str ]
        
    def detokenize(self, tokens):
        return self.deprep([ self.t2c[t] for t in tokens ])

    
    @staticmethod
    def prep(s):
        '''Adds an extra token for capital letters while lowering these letters. Replaces `Umlaute` by normal vocal plus Umlaut token.
        Irgendein Beruf becomes ^irgendein ^beruf
        Sportärztin becomes ^sport°arztin'''
        
        t = []
        for c in s:
            if c.isupper():
                t += ['^', c.lower()]
            else:
                t += [ c ]
    
        t = ''.join(t)
        t = t.replace('ä', '°a').replace('ö', '°o').replace('ü', '°u')
            
        return t
    
    @staticmethod
    def deprep(s):
        '''Inverse function of prep'''
        s = ''.join(s)
        s = s.replace('°a', 'ä').replace('°o', 'ö').replace('°u', 'ü')
        t = []
        nextupper = False
        for c in s:
            if c=='^':
                nextupper = True
            elif nextupper:
                t += [ c.upper() ]
                nextupper = False
            else:
                t += [ c ]
        return ''.join(t)