"""Apply pretrained models"""
from bio_embeddings.embed import (BeplerEmbedder, CPCProtEmbedder, FastTextEmbedder,
                                 GloveEmbedder, OneHotEncodingEmbedder, PLUSRNNEmbedder,
                                 ProtTransAlbertBFDEmbedder, ProtTransBertBFDEmbedder,
                                 ProtTransT5BFDEmbedder, ProtTransT5UniRef50Embedder,
                                 ProtTransT5XLU50Embedder, ProtTransXLNetUniRef100Embedder,
                                 Word2VecEmbedder)
import numpy as np
from tqdm import tqdm
import pandas as pd
from tqdm import TqdmWarning
import warnings
warnings.filterwarnings("ignore", category=TqdmWarning)

class BioEmbeddings:
    """Apply Bio embeddings to a protein dataset"""
    def __init__(self, dataset, id_column, seq_column, is_reduced=True, device = None):
        self.dataset = dataset
        self.id_column = id_column
        self.seq_column = seq_column
        self.is_reduced=is_reduced
        self.device = device
        self.embedder = None
        self.embeddings = None
        self.np_data = None

    def __reducing(self):
        self.np_data = np.zeros(shape=(len(self.dataset), self.embedder.embedding_dimension))
        for idx, embed in tqdm(enumerate(self.embeddings), desc="Reducing embeddings"):
            self.np_data[idx] = self.embedder.reduce_per_protein(embed)

    def __apply_model(self, model):
        if self.device is not None:
            self.embedder = model(device=self.device)
        else:
            self.embedder = model()
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.seq_column].to_list())
        if self.is_reduced:
            self.__reducing()
        self.np_data =  pd.DataFrame(data = self.np_data, columns=[f"p_{a}" for a in range(self.embedder.embedding_dimension)])
        return self.np_data
    def apply_bepler(self):
        """Apply Bepler embedder"""
        return self.__apply_model(BeplerEmbedder)
    def apply_cpcprot(self):
        """Apply CPCProt embedder"""
        return self.__apply_model(CPCProtEmbedder)
    def apply_fasttext(self):
        """Apply FastText embedder"""
        return self.__apply_model(FastTextEmbedder)
    def apply_glove(self):
        """Apply Glove embedder"""
        return self.__apply_model(GloveEmbedder)
    def apply_onehot(self):
        """Apply OneHotEncoding embedder"""
        return self.__apply_model(OneHotEncodingEmbedder)
    def apply_plus_rnn(self):
        """Apply PLUSRNN embedder"""
        return self.__apply_model(PLUSRNNEmbedder)
    def apply_prottrans_albert(self):
        """Apply ProtTransAlbert embedder"""
        return self.__apply_model(ProtTransAlbertBFDEmbedder)
    def apply_prottrans_bert(self):
        """Apply ProtTransBertBFD embedder"""
        return self.__apply_model(ProtTransBertBFDEmbedder)
    def apply_prottrans_t5_uniref(self):
        """Apply ProtTransT5UniRef50 embedder"""
        return self.__apply_model(ProtTransT5UniRef50Embedder)
    def apply_prottrans_t5_xlu50(self):
        """Apply ProtTransT5XLU50 embedder"""
        return self.__apply_model(ProtTransT5XLU50Embedder)
    def apply_prottrans_t5bdf(self):
        """Apply ProtTransT5BFD embedder"""
        return self.__apply_model(ProtTransT5BFDEmbedder)
    def apply_prottrans_xlnetuniref100(self):
        """Apply ProtTransXLNetUniRef100 embedder"""
        return self.__apply_model(ProtTransXLNetUniRef100Embedder)
    def apply_word2vec(self):
        """Apply Word2Vec embedder"""
        return self.__apply_model(Word2VecEmbedder)
