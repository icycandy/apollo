
# coding: utf-8
import numpy as np
import random
import matplotlib
import os
import json
#import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
apollo_root = os.environ['APOLLO_ROOT']

import apollo
import logging
from apollo import layers

import pickle
import sys

def generate_vocab(corpus_file, corpus_pickle, vocab_pickle):
    """
    从训练语料中生成词汇表
    """
    all_chars = set(['\n'])
    data = []
    vocab = {}
    vocab['\n'] = 1
    count = 1
    with open(corpus_file) as f:
        for sentence in f:
            u_sentence = sentence.decode('utf8')
            for c in u_sentence:
                if c not in all_chars:
                    all_chars.add(c)
                    count += 1
                    vocab[c] = count

                data.append(vocab[c])

    with open(corpus_pickle, 'wb') as f:
        pickle.dump(data, f)
    with open(vocab_pickle, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(corpus_pickle, vocab_pickle):
    """
    加载词汇表
    """
    #with open(corpus_pickle, 'rb') as f:
    #    data = pickle.load(f)

    vocab = {}
    with open(vocab_pickle, 'rb') as f:
        vocab = pickle.load(f)

    ivocab = {}
    for c,i in vocab.iteritems():
        ivocab[i] = c

    return vocab,ivocab

def get_data():
    data_source = './data/wangfeng+partial_similar/input.txt'
    if not os.path.exists(data_source):
        raise IOError('You must download the data with ./data/char_model/get_char.sh')
    epoch = 0
    while True:
        with open(data_source, 'r') as f:
            for x in f.readlines():
                data = x
                #data = json.loads(x)
                #if len(data['body']) == 0:
                #    continue
                yield data
        logging.info('epoch %s finished' % epoch)
        epoch += 1

def get_data_batch(data_iter, hyper):
    while True:
        batch = []
        for i in range(hyper['batch_size']):
            batch.append(next(data_iter))
        yield batch

def pad_batch(sentence_batch, hyper):
    max_len = max(len(x.decode('utf8')) for x in sentence_batch)
    result = []
    for sentence in sentence_batch:
        u_sentence = sentence.decode('utf8')
        chars = [vocab[c] for c in u_sentence] 
        result.append(chars + [hyper['zero_symbol']] * (max_len - len(u_sentence)))
    return result

def forward(net, hyper, sentence_batches):
    batch = next(sentence_batches)
    #sentence_batch = np.array(pad_batch([x['body'] for x in batch], hyper))
    sentence_batch = np.array(pad_batch(batch, hyper))
    length = min(sentence_batch.shape[1], 100)
    assert length > 0

    filler = layers.Filler(type='uniform', max=hyper['init_range'],
        min=(-hyper['init_range']))
    net.forward_layer(layers.NumpyData(name='lstm_seed',
        data=np.zeros((hyper['batch_size'], hyper['mem_cells'], 1, 1))))
    net.forward_layer(layers.NumpyData(name='label',
        data=np.zeros((hyper['batch_size'] * length, 1, 1, 1))))
    loss = []
    for step in range(length):
        net.forward_layer(layers.DummyData(name=('word%d' % step),
            shape=[hyper['batch_size'], 1, 1, 1]))
        if step == 0:
            prev_hidden = 'lstm_seed'
            prev_mem = 'lstm_seed'
            word = np.zeros(sentence_batch[:, 0].shape)
        else:
            prev_hidden = 'lstm%d_hidden' % (step - 1)
            prev_mem = 'lstm%d_mem' % (step - 1)
            word = sentence_batch[:, step - 1]
        net.tops['word%d' % step].data[:,0,0,0] = word
        net.forward_layer(layers.Wordvec(name=('wordvec%d' % step),
            bottoms=['word%d' % step],
            dimension=hyper['mem_cells'], vocab_size=hyper['vocab_size'],
            param_names=['wordvec_param'], weight_filler=filler))
        net.forward_layer(layers.Concat(name='lstm_concat%d' % step,
            bottoms=[prev_hidden, 'wordvec%d' % step]))
        net.forward_layer(layers.Lstm(name='lstm%d' % step,
            bottoms=['lstm_concat%d' % step, prev_mem],
            param_names=['lstm_input_value', 'lstm_input_gate',
                'lstm_forget_gate', 'lstm_output_gate'],
            tops=['lstm%d_hidden' % step, 'lstm%d_mem' % step],
            num_cells=hyper['mem_cells'], weight_filler=filler))
        net.forward_layer(layers.Dropout(name='dropout%d' % step,
            bottoms=['lstm%d_hidden' % step], dropout_ratio=0.16))
        
        label = np.reshape(sentence_batch[:, step], (hyper['batch_size'], 1, 1, 1))
        net.forward_layer(layers.NumpyData(name='label%d' % step,
            data=label))
        net.forward_layer(layers.InnerProduct(name='ip%d' % step, bottoms=['dropout%d' % step],
            param_names=['softmax_ip_weights', 'softmax_ip_bias'],
            num_output=hyper['vocab_size'], weight_filler=filler))
        loss.append(net.forward_layer(layers.SoftmaxWithLoss(name='softmax_loss%d' % step,
            ignore_label=hyper['zero_symbol'], bottoms=['ip%d' % step, 'label%d' % step])))

    return np.mean(loss)

def softmax_choice(data):
    probs = data.flatten().astype(np.float64)
    probs /= probs.sum()
    return np.random.choice(range(len(probs)), p=probs)

def eval_forward(net, hyper):
    output_words = []
    filler = layers.Filler(type='uniform', max=hyper['init_range'],
        min=(-hyper['init_range']))
    net.forward_layer(layers.NumpyData(name='lstm_hidden_prev',
        data=np.zeros((1, hyper['mem_cells'], 1, 1))))
    net.forward_layer(layers.NumpyData(name='lstm_mem_prev',
        data=np.zeros((1, hyper['mem_cells'], 1, 1))))
    length = hyper['length']
    for step in range(length):
        net.forward_layer(layers.NumpyData(name=('word'),
            data=np.zeros((1, 1, 1, 1))))
        prev_hidden = 'lstm_hidden_prev'
        prev_mem = 'lstm_mem_prev'
        word = np.zeros((1, 1, 1, 1))
        if step == 0:
            #output = ord('.')
            output = vocab[' ']
        else:
            output = softmax_choice(net.tops['softmax'].data)
        output_words.append(output)
        net.tops['word'].data[0,0,0,0] = output
        net.forward_layer(layers.Wordvec(name=('wordvec'),
            bottoms=['word'],
            dimension=hyper['mem_cells'], vocab_size=hyper['vocab_size'],
            param_names=['wordvec_param'], weight_filler=filler))
        net.forward_layer(layers.Concat(name='lstm_concat',
            bottoms=[prev_hidden, 'wordvec']))
        net.forward_layer(layers.Lstm(name='lstm',
            bottoms=['lstm_concat', prev_mem],
            param_names=['lstm_input_value', 'lstm_input_gate',
                'lstm_forget_gate', 'lstm_output_gate'],
            tops=['lstm_hidden_next', 'lstm_mem_next'],
            num_cells=hyper['mem_cells'], weight_filler=filler))
        net.forward_layer(layers.Dropout(name='dropout',
            bottoms=['lstm_hidden_next'], dropout_ratio=0.16))

        net.forward_layer(layers.InnerProduct(name='ip', bottoms=['dropout'],
            param_names=['softmax_ip_weights', 'softmax_ip_bias'],
            num_output=hyper['vocab_size'], weight_filler=filler))
        net.tops['ip'].data[:] *= hyper['i_temperature']
        net.forward_layer(layers.Softmax(name='softmax',
            ignore_label=hyper['zero_symbol'], bottoms=['ip']))
        net.tops['lstm_hidden_prev'].data_tensor.copy_from(net.tops['lstm_hidden_next'].data_tensor)
        net.tops['lstm_mem_prev'].data_tensor.copy_from(net.tops['lstm_mem_next'].data_tensor)
        net.reset_forward()
    print ''.join([ivocab[x].encode('utf8') for x in output_words])
    return 0.

vocab = {}
ivocab = {}

corpus_file = './data/wangfeng+partial_similar/input.txt'
corpus_pickle = './data/wangfeng+partial_similar/input.pkl'
vocab_pickle = './data/wangfeng+partial_similar/vocab.pkl'
generate_vocab(corpus_file, corpus_pickle, vocab_pickle)
vocab,ivocab = load_vocab(corpus_pickle, vocab_pickle)

def main():
    hyper = {}
    hyper['max_iter'] = 10000
    hyper['snapshot_prefix'] = './char/'
    hyper['schematic_prefix'] = './graph/'
    hyper['snapshot_interval'] = 1000
    hyper['random_seed'] = 22
    hyper['gamma'] = 0.8
    hyper['stepsize'] = 2500
    hyper['graph_interval'] = 1000
    hyper['graph_prefix'] = './graph/'
    hyper['mem_cells'] = 250
    hyper['vocab_size'] = 256
    hyper['batch_size'] = 32
    hyper['init_range'] = 0.1
    #hyper['zero_symbol'] = hyper['vocab_size'] - 1
    #hyper['unknown_symbol'] = hyper['vocab_size'] - 2
    hyper['test_interval'] = None
    hyper['test_iter'] = 1
    hyper['base_lr'] = 0.2
    hyper['weight_decay'] = 0
    hyper['momentum'] = 0.0
    hyper['clip_gradients'] = 100
    hyper['display_interval'] = 100
    hyper['length'] = 2000
    hyper['i_temperature'] = 1.5

    hyper['unknown_symbol'] = len(vocab) + 1
    hyper['zero_symbol'] = len(vocab) + 2
    hyper['vocab_size'] = len(vocab) + 2

    vocab[''] = hyper['unknown_symbol']
    ivocab[hyper['unknown_symbol']] = ''
    ivocab[hyper['zero_symbol']] = ''

    sentences = get_data()
    sentence_batches = get_data_batch(sentences, hyper)

    args = apollo.utils.training.default_parser().parse_args()
    hyper.update({k:v for k, v in vars(args).iteritems() if v is not None})
    apollo.utils.training.default_train(hyper, forward=(
        lambda net, hyper: forward(net, hyper, sentence_batches)),
        test_forward=eval_forward)

if __name__ == '__main__':
    main()

