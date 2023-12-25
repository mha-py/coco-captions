


from layers import *




class Net(nn.Module):
    def __init__(self, n, nh, ntok):
        super().__init__()

        self.dense0 = nn.Linear(2048, n)
        self.emb = nn.Embedding(ntok, n)
        self.posenc = PositionalEncoding(n)
        self.ln1 = LayerNorm(n)
        self.dec1 = DecoderBlock(n, n, nh)
        self.dec2 = DecoderBlock(n, n, nh)
        self.dec3 = DecoderBlock(n, n, nh)
        #self.dec4 = DecoderBlock(n, n, nh)
        self.ln2 = LayerNorm(n)
        self.dense1 = nn.Linear(n, ntok)
        
        self.dropout01 = nn.Dropout(0.1)
        self.dropout03 = nn.Dropout(0.3)
        self.d_model = n
        self.cuda()
        
    def forward(self, x, y):
        mask = np2t(np.tri(y.shape[1])[None]).type(torch.float32).cuda()

        x = self.dropout03(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.dense0(x)
        
        y = self.emb(y) * np.sqrt(self.d_model)
        y = self.posenc(y)
        y = self.ln1(y)
        y = self.dropout01(y)
        y = self.dec1(y, x, mask)
        y = self.dec2(y, x, mask)
        y = self.dec3(y, x, mask)
        y = self.ln2(y)
        y = self.dense1(y)
        
        return y