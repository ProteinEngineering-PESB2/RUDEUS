class constant_values(object):

    def __init__(self):
        self.n_cores = 12
        self.possible_residues = [
            'A',
            'C',
            'D',
            'E',
            'F',
            'G',
            'H',
            'I',
            'N',
            'K',
            'L',
            'M',
            'P',
            'Q',
            'R',
            'S',
            'T',
            'V',
            'W',
            'Y'
        ]

        self.__create_dict_pos()

    def __create_dict_pos(self):

        residues = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'N', 'K', 'L', 'M', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        residues.sort()
        self.dict_value = {}

        for i in range(len(residues)):
            self.dict_value.update({residues[i]: i})
        
