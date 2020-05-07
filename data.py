import os
import nltk
import collections
import itertools
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize

EOS = '<EOS>'
EOP = '<EOP>'

def load_from_author(author, work=None):
    def gut_to_paragraphs(content):
        return list(
            map(
                lambda par: par.replace('\n', ' '), 
                filter(
                    # Remove empties
                    lambda par: len(par) > 0,
                    content.split('\n\n')
                )
            )
        )
    if work is None:
        path = os.path.join('./datasets', author)
    else:
        path = os.path.join('./datasets', author, work)
    return load_paragraphs(path, to_paragraphs=gut_to_paragraphs)

def load_paragraphs(path, to_paragraphs=lambda content: content.split('\n')):
    
    def process_individual_file(filepath):
        with open(os.path.abspath(filepath, 'r')) as f:
            content = f.read()
            return to_paragraphs(content)

    if os.path.isdir(path):
        results = []
        num_files = 0
        for filepath in os.listdir(path):
            if os.path.isfile(filepath):
                name = filepath.split('.txt')[0]
                results.extend(process_individual_file(filepath))
                num_files += 1
        print('Loaded {} paragraphs from {} files in {}'.format(len(results), num_files, path))
        return results
    else:
        paragraphs = process_individual_file(path)
        print('Loaded {} paragraphs from {}'.format(len(paragraphs), filename))
        return paragraphs


def clean_word(word):
    return word.strip().lower().replace('_', '')

def paragraphs_as_tokens(paragraphs):
    tokens = []
    for paragraph in paragraphs:
        sentences = nltk.tokenize.sent_tokenize(paragraph)
        for sentence in sentences:
            words = nltk.tokenize.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
            tokens.extend((clean_word(word), pos_tag[1]) for word, pos_tag in zip(words, pos_tags))
            tokens.append((EOS, 'EOS'))

        tokens.append((EOP, 'EOP'))
    return tokens

def reduced_text(path=None):
    if not path:
        paragraphs = load_from_author('lfrankbaum')
    else:
        # SPECIFY A CUSTOM LOAD FUNC HERE
        paragraphs = load_paragraphs(path)
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
    results = reduced_text('nietzsche')[:100]

