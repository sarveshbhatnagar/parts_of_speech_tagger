class POS_label():
    def __init__(self, pos_label):
        self.pos_label = pos_label

    def split_label(self, tok, split_by="/"):
        """
        Splits the label into (name,pos) pair

        :param tok: e.g. "Sarvesh/NN"
        :param split_by: default="/"
        :return (name,pos):
        """
        return tuple(tok.split(split_by))
