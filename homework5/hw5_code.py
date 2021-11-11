import numpy as np

data = open('shakespeare_train.txt', 'r').read()  # should be simple plain text file

# Using the trained weights 
a = np.load("char-rnn-snapshot.npz", allow_pickle=True)
Wxh = a["Wxh"]  # 250 x 62
Whh = a["Whh"]  # 250 x 250
Why = a["Why"]  # 62 x 250
bh = a["bh"]  # 250 x 1
by = a["by"]  # 62 x 1

chars, data_size, vocab_size, char_to_ix, ix_to_char = a["chars"].tolist(), a["data_size"].tolist(), a[
    "vocab_size"].tolist(), a["char_to_ix"].tolist(), a["ix_to_char"].tolist()

# hyperparameters
hidden_size = 250
seq_length = 1000  # number of steps to unroll the RNN for


def oneHot(vocabSize, word_idx):
    x = np.zeros((vocabSize, 1))
    x[word_idx] = 1
    return x


# Part 1: Generating Samples
def temp(length, alpha=1):
    """
    generate a sample text with assigned alpha value for temperture.
    """
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    inputs = [char_to_ix[ch] for ch in data[:seq_length]]
    hs = np.zeros((hidden_size, 1))

    # generates a sample
    sample_ix = sample(hs, inputs[0], length, alpha)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n%s \n----\n\n' % (txt,))


def sample(h, seed_ix, n, alpha):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    x = oneHot(vocab_size, seed_ix)
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        y *= alpha
        p = np.exp(y) / np.sum(np.exp(y))
        # p实际是个数组，大小应该与指定的a相同，用来规定选取a中每个元素的概率，默认为概率相同
        # ravel方法将数组维度拉成一维数组
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = oneHot(vocab_size, ix)
        ixes.append(ix)
    return ixes


# Part 2: Complete a String
def comp(m, n):
    """
    given a string with length m, complete the string with length n more characters
    """
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    np.random.seed(1)
    # the context string starts from a random position in the data
    start_index = np.random.randint(265000)
    inputs = [char_to_ix[ch] for ch in data[start_index: start_index + seq_length]]
    h = np.zeros((hidden_size, 1))
    word_index = 0
    ix = inputs[word_index]
    x = oneHot(vocab_size, ix)

    ixes = [ix]

    # generates the context text
    for t in range(m):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        '''此时不用随机生成
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        '''
        word_index += 1
        ix = inputs[word_index]
        x = oneHot(vocab_size, ix)
        ixes.append(ix)

    txt = ''.join(ix_to_char[ix] for ix in ixes)
    print('Context: \n----\n%s \n----\n' % (txt,))

    # compute the softmax probability and sample from the data
    # and use the output as the next input where we start the continuation
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = oneHot(vocab_size, ix)

    # start completing the string
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = oneHot(vocab_size, ix)
        ixes.append(ix)

    # generates the continuation of the string
    txt = ''.join(ix_to_char[ix] for ix in ixes)
    print('Continuation: \n----\n%s \n----\n\n\n' % (txt,))


if __name__ == '__main__':
    # Test case
    # Part 1
    print('alpha = 5, sample text:')
    temp(length=200, alpha=5)
    print('alpha = 1, sample text:')
    temp(length=200, alpha=1)
    print('alpha = 0.1, sample text:')
    temp(length=200, alpha=0.1)

    # Part 2
    comp(780, 200)
    comp(50, 500)
    comp(2, 500)
    comp(300, 300)
    comp(100, 500)
