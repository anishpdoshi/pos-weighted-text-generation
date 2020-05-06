import os
import nltk
import collections
import itertools
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize # punkt tokenizers

AUTHORS = {
    'nietzsche': {
        'MIN_PARAGRAPH_LENGTH': 200
    },
    'lfrankbaum': {
        'MIN_PARAGRAPH_LENGTH': 0
    }
}
EOS = '<EOS>'
EOP = '<EOP>'

def load_paragraphs(author, work= None):
    assert author in list(AUTHORS.keys())

    results = {}
    directory = './datasets/' + author
    for filename in os.listdir(directory):
        name = filename.split('.txt')[0]
        if work and name != work:
            continue
        with open(os.path.abspath(os.path.join(directory, filename)), 'r') as f:
            raw_content = f.read()
            # Strip out intro data
            begin_index = raw_content.find('START OF')
            end_index = raw_content.find('END OF')
            content = raw_content[begin_index:end_index]
            paragraphs = list(
                map(
                    lambda par: par.replace('\n', ' '), 
                    filter(
                        lambda par: len(par) > AUTHORS[author]['MIN_PARAGRAPH_LENGTH'],
                        content.split('\n\n')
                    )
                )
            )
            print('Loaded {}: {} paragraphs'.format(filename, len(paragraphs)))
            if work:
                return paragraphs

            results[name] = paragraphs
    
    return results

def clean_word(word):
    return word.strip().lower().replace('_', '')

def paragraphs_as_token_stream(paragraphs):
    # TODO - make this actually a 'stream' (generator) lol
    stream = []
    for paragraph in paragraphs:
        sentences = nltk.tokenize.sent_tokenize(paragraph)
        for sentence in sentences:
            words = nltk.tokenize.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
            stream.extend((clean_word(word), pos_tag[1]) for word, pos_tag in zip(words, pos_tags))
            stream.append((EOS, 'EOS'))

        stream.append((EOP, 'EOP'))
    return stream

def reduced_text(author, work = None):
    paragraphs = load_paragraphs(author, work)
    if isinstance(paragraphs, dict):
        paragraphs = itertools.chain.from_iterable(paragraphs.values())
    return paragraphs_as_token_stream(paragraphs)

def to_sentence(words):
    sent = ''
    if len(words) == 0:
        return sent

    sent += words[0].title()
    for word in words:
        if word == EOS:
            sent += '.'
        elif word == EOP:
            sent += '\n'
        elif word[0].isalnum():
            sent += ' '
            sent += word
        else:
            sent += word
    return sent

if __name__ == '__main__':
    results = reduced_text('nietzsche')

