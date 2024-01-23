from bio_embeddings.embed import BeplerEmbedder
from bio_embeddings.embed import CPCProtEmbedder
from bio_embeddings.embed import ESM1bEmbedder
from bio_embeddings.embed import ESMEmbedder
from bio_embeddings.embed import ESM1vEmbedder
from bio_embeddings.embed import FastTextEmbedder
from bio_embeddings.embed import GloveEmbedder
from bio_embeddings.embed import OneHotEncodingEmbedder
from bio_embeddings.embed import PLUSRNNEmbedder
from bio_embeddings.embed import ProtTransAlbertBFDEmbedder
from bio_embeddings.embed import ProtTransBertBFDEmbedder
from bio_embeddings.embed import ProtTransT5BFDEmbedder
from bio_embeddings.embed import ProtTransT5UniRef50Embedder
from bio_embeddings.embed import ProtTransT5XLU50Embedder
from bio_embeddings.embed import ProtTransXLNetUniRef100Embedder
from bio_embeddings.embed import UniRepEmbedder
from bio_embeddings.embed import Word2VecEmbedder

import numpy as np
from tqdm import tqdm

class using_bioembedding (object):

    def __init__(
            self,
            dataset=None,
            id_seq=None,
            column_seq=None,
            is_reduced=True,
            device = None
            ):
        
        self.dataset = dataset
        self.id_seq = id_seq
        self.column_seq = column_seq
        self.is_reduced=is_reduced
        self.device = device

        # to save the results
        self.embedder = None
        self.embeddings = None
        self.np_data = None

    def __reducing(self):
        self.np_data = np.zeros(shape=(len(self.dataset), self.embedder.embedding_dimension))
        for idx, embed in tqdm(enumerate(self.embeddings), desc="Reducing embeddings"):
            self.np_data[idx] = self.embedder.reduce_per_protein(embed)

    def apply_bepler(self):
        if self.device != None:
            self.embedder = BeplerEmbedder(device=self.device)
        else:
            self.embedder = BeplerEmbedder()
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()
    
    def apply_cpcprot(self):
        if self.device != None:
            self.embedder = CPCProtEmbedder(device=self.device)
        else:
            self.embedder = CPCProtEmbedder()
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()
    
    def apply_esm1b(self):
        if self.device != None:
            self.embedder = ESM1bEmbedder(device=self.device)
        else:
            self.embedder = ESM1bEmbedder()
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()

    def apply_esme(self):
        if self.device != None:
            self.embedder = ESMEmbedder(device=self.device)
        else:
            self.embedder = ESMEmbedder()
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()


    def apply_esm1v(self, ensemble_id=5):
        if self.device != None:
            self.embedder = ESM1vEmbedder(ensemble_id=ensemble_id, device=self.device)
        else:
            self.embedder = ESM1vEmbedder(ensemble_id=ensemble_id)
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()

    def apply_fasttext(self):
        if self.device != None:
            self.embedder = FastTextEmbedder(device=self.device)
        else:
            self.embedder = FastTextEmbedder()
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()

    def apply_glove(self):
        if self.device != None:
            self.embedder = GloveEmbedder(device=self.device)
        else:
            self.embedder = GloveEmbedder()
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()

    def apply_onehot(self):
        if self.device != None:
            self.embedder = OneHotEncodingEmbedder(device=self.device)
        else:
            self.embedder = OneHotEncodingEmbedder()
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()

    def apply_plus_rnn(self):
        if self.device != None:
            self.embedder = PLUSRNNEmbedder(device=self.device)
        else:
            self.embedder = PLUSRNNEmbedder()
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()

    def apply_prottrans_albert(self):
        if self.device != None:
            self.embedder = ProtTransAlbertBFDEmbedder(device=self.device)
        else:
            self.embedder = ProtTransAlbertBFDEmbedder()
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()

    def apply_prottrans_bert(self):
        if self.device != None:
            self.embedder = ProtTransBertBFDEmbedder(device=self.device)
        else:
            self.embedder = ProtTransBertBFDEmbedder()
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()

    def apply_prottrans_T5BFD(self):
        if self.device != None:
            self.embedder = ProtTransT5BFDEmbedder(device=self.device)
        else:
            self.embedder = ProtTransT5BFDEmbedder()
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()

    def apply_prottrans_T5_UniRef(self):
        if self.device != None:
            self.embedder = ProtTransT5UniRef50Embedder(device=self.device)
        else:
            self.embedder = ProtTransT5UniRef50Embedder()
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()

    def apply_prottrans_T5_XLU50(self):
        if self.device != None:
            self.embedder = ProtTransT5XLU50Embedder(half_precision_model=True, device=self.device)
        else:
            self.embedder = ProtTransT5XLU50Embedder(half_precision_model=True)
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()

    def apply_prottrans_XLNetUniRef(self):
        if self.device != None:
            self.embedder = ProtTransXLNetUniRef100Embedder(device=self.device)
        else:
            self.embedder = ProtTransXLNetUniRef100Embedder()
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()

    def apply_unirep(self):
        if self.device != None:
            self.embedder = UniRepEmbedder(device=self.device)
        else:
            self.embedder = UniRepEmbedder()
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()

    def apply_word2vec(self):
        if self.device != None:
            self.embedder = Word2VecEmbedder(device=self.device)
        else:
            self.embedder = Word2VecEmbedder()
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()