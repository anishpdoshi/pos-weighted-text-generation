import torch
import nltk

from data import clean_word

"""A customizable beam_search - accepts decoding constraints
for i in num_outputs:
    for beam_seq in beams
        -  get constraints from constraint_model(beam_seq)
        -  get probabilities from model(beam_seq (with hidden shortcut))
        -  find topk sequence continuations across probabilities
        -  create new beams with these sequence continuationos
    - 
"""
def constrained_beam_search(
    model,
    initial_output,
    initial_hidden,
    get_constraint_vector,
    output_length,
    beam_width = 10
):
    vals, inds = torch.exp(initial_output).topk(beam_width)
    beams = [(
        [ind.item()],
        initial_hidden,
        val.item()
    ) for val, ind in zip(vals, inds)]

    vocab_size = len(initial_output)

    for i in range(output_length):
        possibilities = []
        hiddens = []
        
        for sequence, hidden, probability in beams:
            last_as_seq = torch.tensor(sequence[-1:]).unsqueeze(0).T
            next_out, next_hidden = model(last_as_seq, hidden)
            
            constraint_vector = get_constraint_vector(sequence)

            hiddens.append(next_hidden)
            possibilities.append(probability * constraint_vector * torch.exp(next_out[-1]))

        vals, inds = torch.cat(possibilities).topk(beam_width)
        
        # renormalize probabilities to avoid decay
        max_prob = vals.max().item()
        min_prob = vals.min().item()

        for ind, val in zip(inds.tolist(), vals.tolist()):

            word_index = ind % vocab_size
            beam_index = ind // vocab_size
            beams.append((
                beams[beam_index][0] + [word_index],
                hiddens[beam_index],
                (val - min_prob) / (max_prob - min_prob)
            ))
        beams = beams[-beam_width:]

    return beams


def generate_sentences(initial, word_model, pos_model, corpus, constraint_type='weighted'):
    word_model.eval()
    pos_model.eval()

    def get_constraint_vector(word_ind_sequence):
        if constraint_type == 'none':
            return torch.ones(corpus.num_words)

        word_seq = corpus.to_words(word_ind_sequence)
        pos_ind_sequence = corpus.to_tag_inds(tag for  _, tag in nltk.pos_tag(word_seq))
        stacked_seq = torch.stack([pos_ind_sequence, torch.tensor(word_ind_sequence)], -1)

        hidden = pos_model.init_hidden(1)

        tag_output, _ = pos_model(stacked_seq.unsqueeze(1), hidden)
        tag_distribution = torch.exp(tag_output[-1]).squeeze()

        if constraint_type == 'weighted':
            return torch.matmul(corpus.pw_tensor.T, tag_distribution)
        else:
            next_tag = torch.argmax(tag_distribution)
            return corpus.pw_tensor[next_tag]

    initial_indices = corpus.to_word_inds(map(clean_word, initial.split(' ')))
    test_output, test_hidden = word_model(initial_indices.unsqueeze(0).T, word_model.init_hidden(1))

    results = constrained_beam_search(
        word_model,
        test_output[-1],
        test_hidden,
        get_constraint_vector,
        100,
        beam_width = 10
    )

    return [
        (' '.join(corpus.to_words(seq)), prob)
        for seq, _, prob in results
    ]
