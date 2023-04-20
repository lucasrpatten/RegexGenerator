class Preprocessing:
    def __init__(self, allowed_texts, rejected_texts):
        self.allowed_texts = allowed_texts
        self.rejected_texts = rejected_texts


def char_mapping(allowed_texts, rejected_texts, pattern):
    a = [[ord(i) for i in j] for j in allowed_texts]
    r = [[ord(i) for i in j] for j in rejected_texts]
    p = [ord(i) for i in pattern]
    return a, r, p
