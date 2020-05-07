import torch


class Corpus(object):
    def __init__(self, tagged_sequence):
        self.text_length = len(tagged_sequence)
        self.words_data = torch.empty(self.text_length).long()
        self.tags_data = torch.empty(self.text_length).long()

        self.w_to_i = {}
        self.i_to_w = {}
        self.p_to_i = {}
        self.i_to_p = {}

        index = 0
        w_index = 0
        p_index = 0
        for index, (word, pos_tag) in enumerate(tagged_sequence):
            if word not in self.w_to_i:
                self.w_to_i[word] = w_index
                self.i_to_w[w_index] = word
                w_index += 1
            if pos_tag not in self.p_to_i:
                self.p_to_i[pos_tag] = p_index
                self.i_to_p[p_index] = pos_tag
                p_index += 1

            self.words_data[index] = self.w_to_i[word]
            self.tags_data[index] = self.p_to_i[pos_tag]

        self.num_words = len(self.w_to_i)
        self.num_tags = len(self.p_to_i)

        self.pw_tensor = torch.zeros(self.num_tags, self.num_words)
        for word, pos_tag in tagged_sequence:
            self.pw_tensor[self.p_to_i[pos_tag]][self.w_to_i[word]] = 1.0

    def to_word_inds(self, words, tensor=True):
        items = [self.w_to_i[word] for word in words]
        if tensor:
            return torch.tensor(items)
        return items

    def to_words(self, inds):
        return [self.i_to_w[ind.item()] if torch.is_tensor(ind) else self.i_to_w[ind] for ind in inds]

    def to_tag_inds(self, pos_tags, tensor=True):
        items = [self.p_to_i[pos_tag] for pos_tag in pos_tags]
        if tensor:
            return torch.tensor(items)
        return items

    def to_tags(self, inds):
        return [self.i_to_p[ind.item()] if torch.is_tensor(ind) else self.i_to_p[ind] for ind in inds]


def create_dictionaries(sequence):
    w_to_i = {}
    i_to_w = {}

    index = 0
    for word in sequence:
        if word not in w_to_i:
            w_to_i[word] = index
            i_to_w[index] = word
            index += 1

    return {
        'w_to_i': w_to_i,
        'i_to_w': i_to_w,
        'to_indices': lambda token_seq: list(map(lambda token: w_to_i[token], token_seq)),
        'to_words': lambda indice_seq: list(map(lambda indice: i_to_w[indice.item()], indice_seq)),
        'vocab_size': len(w_to_i)
    }


def create_pos_word_tensor(tagged_sequence, w_to_i, p_to_i):
    pw = torch.zeros(len(p_to_i), len(w_to_i))
    for word, tag in tagged_sequence:
        pw[p_to_i[tag]][w_to_i[word]] = 1.0

    return pw