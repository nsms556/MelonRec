class AutoEncoder(nn.Module):
    def __init__(self, D_in, H, D_out, dropout):
        super(AutoEncoder, self).__init__()
        encoder_layer = nn.Linear(D_in, H, bias=True)
        decoder_layer = nn.Linear(H, D_out, bias=True)

        torch.nn.init.xavier_uniform_(encoder_layer.weight)
        torch.nn.init.xavier_uniform_(decoder_layer.weight)

        self.encoder = nn.Sequential(
                        nn.Dropout(dropout),
                        encoder_layer,
                        nn.BatchNorm1d(H),
                        nn.LeakyReLU())
        self.decoder = nn.Sequential(
                        decoder_layer,
                        nn.Sigmoid())

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)
        return out_decoder

class string2vec :
    def __init__(self, train_data, size=200, window=5, min_count=2, workers=8, sg=1, hs=1):
        self.model = Word2Vec(size=size, window=window, min_count=min_count, workers=workers, sg=sg, hs=hs)
        self.model.build_vocab(train_data)

    def set_model(self, model_fn):
        self.model = Word2Vec.load(model_fn)

    def save_embeddings(self, emb_fn):
        word_vectors = self.model.wv

        vocabs = []
        vectors = []
        for key in word_vectors.vocab:
            vocabs.append(key)
            vectors.append(word_vectors[key])

        df = pd.DataFrame()
        df['voca'] = vocabs
        df['vector'] = vectors

        df.to_csv(emb_fn, index=False)

    def save_model(self, md_fn):
        self.model.save(md_fn)
        print("word embedding model {} is trained".format(md_fn))

    def show_similar_words(self, word, topn):
        print(self.model.most_similar(positive=[word], topn=topn))
