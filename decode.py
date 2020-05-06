import torch
import nltk

"""A customizable beam_search - accepts decoding constraints"""
def constrained_beam_search(
    model,
    initial_output,
    initial_hidden,
    get_constraint_vector,
    output_length,
    beam_width = 10
):
    # for i in num_outputs:
    # for beam_seq in beams
    # -  get constraints from constraint_model(beam_seq)
    # -  get probabilities from model(beam_seq (with hidden shortcut))
    # -  find topk sequence continuations across probabilities
    # -  create new beams with these sequence continuationos
    # - 
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
        
        # renormalize probabilities
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