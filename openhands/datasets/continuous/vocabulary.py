import torchtext

BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>" 

def get_vocabulary_from_iter(iterator, text=False):
    """
    Returns a torch.vocab.Vocab based on an iterator.

    Sets the special tokens, and the default OOV index to the UNK_TOKEN.
    """
    specials = [UNK_TOKEN, PAD_TOKEN]

    if text:
        specials += [BOS_TOKEN, EOS_TOKEN]

    vocab = torchtext.vocab.build_vocab_from_iterator(
        iterator,
        specials=specials
    )
    vocab.set_default_index(vocab[UNK_TOKEN])
    return vocab