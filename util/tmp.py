import random

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

'''
    whether to use Xavier initialization, as described in
        http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
'''
USE_XAVIER_INIT = False
'''
    Defaut random generators
'''
#random.seed(5817)
default_rng = np.random.RandomState(random.randint(0,9999))
#default_srng = T.shared_randomstreams.RandomStreams(default_rng.randint(9999))
default_mrng = MRG_RandomStreams(default_rng.randint(9999))
default_srng = default_mrng

'''
    Activation functions
'''
ReLU = lambda x: x * (x > 0)
sigmoid = T.nnet.sigmoid
tanh = T.tanh
softmax = T.nnet.softmax
linear = lambda x: x

def get_activation_by_name(name):
    if name.lower() == "relu":
        return ReLU
    elif name.lower() == "sigmoid":
        return sigmoid
    elif name.lower() == "tanh":
        return tanh
    elif name.lower() == "softmax":
        return softmax
    elif name.lower() == "none" or name.lower() == "linear":
        return linear
    else:
        raise Exception(
            "unknown activation type: {}".format(name)
          )





def set_default_rng_seed(seed):
    global default_rng, default_srng
    random.seed(seed)
    default_rng = np.random.RandomState(random.randint(0,9999))
    default_srng = T.shared_randomstreams.RandomStreams(default_rng.randint(9999))


'''
    Return initial parameter values of the specified size
    Inputs
    ------
        size            : size of the parameter, e.g. (100, 200) and (100,)
        rng             : random generator; the default is used if None
        rng_type        : the way to initialize the values
                            None    -- (default) uniform [-0.05, 0.05]
                            normal  -- Normal distribution with unit variance and zero mean
                            uniform -- uniform distribution with unit variance and zero mean
'''
def random_init(size, rng=None, rng_type=None):
    if rng is None: rng = default_rng
    if rng_type is None:
        #vals = rng.standard_normal(size)
        vals = rng.uniform(low=-0.05, high=0.05, size=size)

    elif rng_type == "normal":
        vals = rng.standard_normal(size)

    elif rng_type == "uniform":
        vals = rng.uniform(low=-3.0**0.5, high=3.0**0.5, size=size)

    else:
        raise Exception(
            "unknown random inittype: {}".format(rng_type)
          )

    return vals.astype(theano.config.floatX)


'''
    return a theano shared variable with initial values as vals
'''
def create_shared(vals, name=None):
    return theano.shared(vals, name=name)


class EmbeddingLayer(object):
    '''
        Embedding layer that
                (1) maps string tokens into integer IDs
                (2) maps integer IDs into embedding vectors (as matrix)
        Inputs
        ------
        n_d             : dimension of word embeddings; may be over-written if embs
                            is specified
        vocab           : an iterator of string tokens; the layer will allocate an ID
                            and a vector for each token in it
        oov             : out-of-vocabulary token
        embs            : an iterator of (word, vector) pairs; these will be added to
                            the layer
        fix_init_embs   : whether to fix the initial word vectors loaded from embs
    '''
    def __init__(self, n_d, vocab, oov="<unk>", embs=None, fix_init_embs=True):

        if embs is not None:
            lst_words = [ ]
            vocab_map = {}
            emb_vals = [ ]
            for word, vector in embs:
                assert word not in vocab_map, "Duplicate words in initial embeddings"
                vocab_map[word] = len(vocab_map)
                emb_vals.append(vector)
                lst_words.append(word)

            self.init_end = len(emb_vals) if fix_init_embs else -1
            if n_d != len(emb_vals[0]):
                say("WARNING: n_d ({}) != init word vector size ({}). Use {} instead.\n".format(
                        n_d, len(emb_vals[0]), len(emb_vals[0])
                    ))
                n_d = len(emb_vals[0])

            say("{} pre-trained embeddings loaded.\n".format(len(emb_vals)))

            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)
                    emb_vals.append(random_init((n_d,))*(0.001 if word != oov else 0.0))
                    lst_words.append(word)

            emb_vals = np.vstack(emb_vals).astype(theano.config.floatX)
            self.vocab_map = vocab_map
            self.lst_words = lst_words
        else:
            lst_words = [ ]
            vocab_map = {}
            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)
                    lst_words.append(word)

            self.lst_words = lst_words
            self.vocab_map = vocab_map
            emb_vals = random_init((len(self.vocab_map), n_d))
            self.init_end = -1

        if oov is not None and oov is not False:
            assert oov in self.vocab_map, "oov {} not in vocab".format(oov)
            self.oov_tok = oov
            self.oov_id = self.vocab_map[oov]
        else:
            self.oov_tok = None
            self.oov_id = -1

        self.embeddings = create_shared(emb_vals)
        if self.init_end > -1:
            self.embeddings_trainable = self.embeddings[self.init_end:]
        else:
            self.embeddings_trainable = self.embeddings

        self.n_V = len(self.vocab_map)
        self.n_d = n_d

    def map_to_words(self, ids):
        n_V, lst_words = self.n_V, self.lst_words
        return [ lst_words[i] if i < n_V else "<err>" for i in ids ]

    def map_to_ids(self, words, filter_oov=False):
        '''
            map the list of string tokens into a numpy array of integer IDs
            Inputs
            ------
            words           : the list of string tokens
            filter_oov      : whether to remove oov tokens in the returned array
            Outputs
            -------
            return the numpy array of word IDs
        '''
        vocab_map = self.vocab_map
        oov_id = self.oov_id
        if filter_oov:
            not_oov = lambda x: x!=oov_id
            return np.array(
                    list(filter(not_oov, [ vocab_map.get(x, oov_id) for x in words ])),
                    dtype="int32"
                )
        else:
            return np.array(
                    [ vocab_map.get(x, oov_id) for x in words ],
                    dtype="int32"
                )

    def forward(self, x):
        '''
            Fetch and return the word embeddings given word IDs x
            Inputs
            ------
            x           : a theano array of integer IDs
            Outputs
            -------
            a theano matrix of word embeddings
        '''
        return self.embeddings[x]

    @property
    def params(self):
        return [ self.embeddings_trainable ]

    @params.setter
    def params(self, param_list):
        self.embeddings.set_value(param_list[0].get_value())
        
        
class Dropout(object):
    '''
        Dropout layer. forward(x) returns the dropout version of x
        Inputs
        ------
        dropout_prob : theano shared variable that stores the dropout probability
        srng         : theano random stream or None (default rng will be used)
        v2           : which dropout version to use
    '''
    def __init__(self, dropout_prob, srng=None, v2=False):
        self.dropout_prob = dropout_prob
        self.srng = srng if srng is not None else default_srng
        self.v2 = v2

    def forward(self, x):
        d = (1-self.dropout_prob) if not self.v2 else (1-self.dropout_prob)**0.5
        mask = self.srng.binomial(
                n = 1,
                p = 1-self.dropout_prob,
                size = x.shape,
                dtype = theano.config.floatX
            )
        return x*mask/d


def apply_dropout(x, dropout_prob, v2=False):
    '''
        Apply dropout on x with the specified probability
    '''
    return Dropout(dropout_prob, v2=v2).forward(x)


class Layer(object):
    '''
        Basic neural layer -- y = f(Wx+b)
        foward(x) returns y
        Inputs
        ------
        n_in            : input dimension
        n_out           : output dimension
        activation      : the non-linear activation function to apply
        has_bias        : whether to include the bias term b in the computation
    '''
    def __init__(self, n_in, n_out, activation,
                            clip_gradients=False,
                            has_bias=True):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.clip_gradients = clip_gradients
        self.has_bias = has_bias

        self.create_parameters()

        # not implemented yet
        if clip_gradients is True:
            raise Exception("gradient clip not implemented")

    def create_parameters(self):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation
        self.initialize_params(n_in, n_out, activation)

    def initialize_params(self, n_in, n_out, activation):
        if USE_XAVIER_INIT:
            if activation == ReLU:
                scale = np.sqrt(4.0/(n_in+n_out), dtype=theano.config.floatX)
                b_vals = np.ones(n_out, dtype=theano.config.floatX) * 0.01
            elif activation == softmax:
                scale = np.float64(0.001).astype(theano.config.floatX)
                b_vals = np.zeros(n_out, dtype=theano.config.floatX)
            else:
                scale = np.sqrt(2.0/(n_in+n_out), dtype=theano.config.floatX)
                b_vals = np.zeros(n_out, dtype=theano.config.floatX)
            W_vals = random_init((n_in,n_out), rng_type="normal") * scale
        else:
            W_vals = random_init((n_in,n_out))
            if activation == softmax:
                W_vals *= 0.001
            if activation == ReLU:
                b_vals = np.ones(n_out, dtype=theano.config.floatX) * 0.01
            else:
                b_vals = random_init((n_out,))
        self.W = create_shared(W_vals, name="W")
        if self.has_bias: self.b = create_shared(b_vals, name="b")

    def forward(self, x):
        if self.has_bias:
            return self.activation(
                    T.dot(x, self.W) + self.b
                )
        else:
            return self.activation(
                    T.dot(x, self.W)
                )

    @property
    def params(self):
        if self.has_bias:
            return [ self.W, self.b ]
        else:
            return [ self.W ]

    @params.setter
    def params(self, param_list):
        self.W.set_value(param_list[0].get_value())
        if self.has_bias: self.b.set_value(param_list[1].get_value())


class RecurrentLayer(Layer):
    '''
        Basic recurrent layer -- h_t = f(Wx + W'h_{t-1} + b)
            forward(x, h_{t-1}) executes one step of the RNN and returns h_t
            forward_all(x, h_0) executes all steps and returns H = {h_0, ... , h_n}
        Inputs
        ------
        n_in            : input dimension
        n_out           : hidden dimension
        activation      : the non-linear function to apply
    '''
    def __init__(self, n_in, n_out, activation,
            clip_gradients=False):
        super(RecurrentLayer, self).__init__(
                n_in, n_out, activation,
                clip_gradients = clip_gradients
            )

    def create_parameters(self):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation

        # re-use the code in super-class Layer
        self.initialize_params(n_in + n_out, n_out, activation)

    def forward(self, x, h):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation
        return activation(
                T.dot(x, self.W[:n_in]) + T.dot(h, self.W[n_in:]) + self.b
            )

    def forward_all(self, x, h0=None):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out,), dtype=theano.config.floatX)
        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = x,
                    outputs_info = [ h0 ]
                )
        return h


class RCNN(Layer):

    '''
        RCNN
        Inputs
        ------
            order           : CNN feature width
            has_outgate     : whether to add a output gate as in LSTM; this can be
                              useful for language modeling
            mode            : 0 if non-linear filter; 1 if linear filter (default)
    '''
    def __init__(self, n_in, n_out, activation=tanh,
            order=1, has_outgate=False, mode=1, clip_gradients=False):

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.order = order
        self.clip_gradients = clip_gradients
        self.has_outgate = has_outgate
        self.mode = mode

        internal_layers = self.internal_layers = [ ]
        for i in range(order):
            input_layer = Layer(n_in, n_out, linear, has_bias=False, \
                    clip_gradients=clip_gradients)
            internal_layers.append(input_layer)

        forget_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        internal_layers.append(forget_gate)

        self.bias = create_shared(random_init((n_out,)), name="bias")

        if has_outgate:
            self.out_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
            self.internal_layers += [ self.out_gate ]

    '''
        One step of recurrent
        Inputs
        ------
            x           : input token at current time/position t
            hc          : hidden/visible states at time/position t-1
        Outputs
        -------
            return hidden/visible states at time/position t
    '''
    def forward(self, x, hc):
        order, n_in, n_out, activation = self.order, self.n_in, self.n_out, self.activation
        layers = self.internal_layers
        if hc.ndim > 1:
            h_tm1 = hc[:, n_out*order:]
        else:
            h_tm1 = hc[n_out*order:]

        forget_t = layers[order].forward(x, h_tm1)
        lst = [ ]
        for i in range(order):
            if hc.ndim > 1:
                c_i_tm1 = hc[:, n_out*i:n_out*i+n_out]
            else:
                c_i_tm1 = hc[n_out*i:n_out*i+n_out]
            in_i_t = layers[i].forward(x)
            if i == 0:
                c_i_t = forget_t * c_i_tm1 + (1-forget_t) * in_i_t
            elif self.mode == 0:
                c_i_t = forget_t * c_i_tm1 + (1-forget_t) * (in_i_t * c_im1_t)
            else:
                c_i_t = forget_t * c_i_tm1 + (1-forget_t) * (in_i_t + c_im1_tm1)
            lst.append(c_i_t)
            c_im1_tm1 = c_i_tm1
            c_im1_t = c_i_t

        if not self.has_outgate:
            h_t = activation(c_i_t + self.bias)
        else:
            out_t = self.out_gate.forward(x, h_tm1)
            h_t = out_t * activation(c_i_t + self.bias)
        lst.append(h_t)

        if hc.ndim > 1:
            return T.concatenate(lst, axis=1)
        else:
            return T.concatenate(lst)

    '''
        Apply recurrent steps to input of all positions/time
        Inputs
        ------
            x           : input tokens x_1, ... , x_n
            h0          : initial states
            return_c    : whether to return hidden states in addition to visible
                          state
        Outputs
        -------
            return visible states (and hidden states) of all positions/time
    '''
    def forward_all(self, x, h0=None, return_c=False):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out*(self.order+1)), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out*(self.order+1),), dtype=theano.config.floatX)
        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = x,
                    outputs_info = [ h0 ]
                )
        if return_c:
            return h
        elif x.ndim > 1:
            return h[:,:,self.n_out*self.order:]
        else:
            return h[:,self.n_out*self.order:]

    def forward2(self, x, hc, f_tm1):
        order, n_in, n_out, activation = self.order, self.n_in, self.n_out, self.activation
        layers = self.internal_layers
        if hc.ndim > 1:
            h_tm1 = hc[:, n_out*order:]
        else:
            h_tm1 = hc[n_out*order:]

        forget_t = layers[order].forward(x, h_tm1)
        lst = [ ]
        for i in range(order):
            if hc.ndim > 1:
                c_i_tm1 = hc[:, n_out*i:n_out*i+n_out]
            else:
                c_i_tm1 = hc[n_out*i:n_out*i+n_out]
            in_i_t = layers[i].forward(x)
            if i == 0:
                c_i_t = forget_t * c_i_tm1 + (1-forget_t) * in_i_t
            elif self.mode == 0:
                c_i_t = forget_t * c_i_tm1 + (1-forget_t) * (in_i_t * c_im1_t)
            else:
                c_i_t = forget_t * c_i_tm1 + (1-forget_t) * (in_i_t + c_im1_tm1)
            lst.append(c_i_t)
            c_im1_tm1 = c_i_tm1
            c_im1_t = c_i_t

        if not self.has_outgate:
            h_t = activation(c_i_t + self.bias)
        else:
            out_t = self.out_gate.forward(x, h_tm1)
            h_t = out_t * activation(c_i_t + self.bias)
        lst.append(h_t)

        if hc.ndim > 1:
            return T.concatenate(lst, axis=1), forget_t
        else:
            return T.concatenate(lst), forget_t

    def get_input_gate(self, x, h0=None):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out*(self.order+1)), dtype=theano.config.floatX)
                f0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out*(self.order+1),), dtype=theano.config.floatX)
                f0 = T.zeros((self.n_out,), dtype=theano.config.floatX)

        [h, f], _ = theano.scan(
                    fn = self.forward2,
                    sequences = x,
                    outputs_info = [ h0,f0 ]
                )
        return 1.0-f

    @property
    def params(self):
        return [ x for layer in self.internal_layers for x in layer.params ] + [ self.bias ]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end
        self.bias.set_value(param_list[-1].get_value())