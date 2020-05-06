import torch

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


def create_pos_dictionaries(tagged_sequence):
    pos_to_words = {}
    for word, tag in tagged_sequence:
        if tag in pos_to_words:
            pos_to_words[tag].add(word)
        else:
            pos_to_words[tag] = set([word])
    
    return pos_to_words

def bow(pos_to_words, w_to_i, p_to_i):
    eye = torch.eye(len(w_to_i))
    bow_reps = {}
    for tag, words in pos_to_words.items():
        indices = [w_to_i[word] for word in words]
        bow_reps[p_to_i[tag]] = torch.sum(eye[indices], 0)

    return bow_reps