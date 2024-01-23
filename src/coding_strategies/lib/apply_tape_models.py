import torch
from tape import ProteinBertModel, UniRepModel, TAPETokenizer

class tape_pretrained_models(object):

    def __encoding_sequences(
            self,
            dataset=None,
            column_seq=None,
            model=None,
            tokenizer=None):

        sequence_encoding = []
        poll_data = []

        for index in dataset.index:
            sequence = str(dataset[column_seq][index])
            token_ids = torch.tensor([tokenizer.encode(sequence)])
            output = model(token_ids)

            sequence_encoding.append(output[0])
            poll_data.append(output[1])
        
        dict_encoding = {"sequences_coding" : sequence_encoding, "polls": poll_data}
        return dict_encoding
    
    def apply_bert_model(
            self,
            dataset=None,
            column_seq=None,
            vocab='iupac',
            config_dic='bert-base'):
        
        model = ProteinBertModel.from_pretrained(config_dic)
        tokenizer = TAPETokenizer(vocab=vocab)

        return self.__encoding_sequences(
            dataset=dataset,
            column_seq=column_seq,
            model=model,
            tokenizer=tokenizer
        )

    def apply_unirep_model(
            self,
            dataset=None,
            column_seq=None,
            vocab='unirep',
            config_dic='babbler-1900'):
        
        model = UniRepModel.from_pretrained(config_dic)
        tokenizer = TAPETokenizer(vocab=vocab)
        
        return self.__encoding_sequences(
            dataset=dataset,
            column_seq=column_seq,
            model=model,
            tokenizer=tokenizer
        )
        

        
    


