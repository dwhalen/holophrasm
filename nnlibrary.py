import numpy as np

"""
This is a library to do pretty basic neural network computations automatically.

It should be kept clean and simple.  Computations are made as nodes are added
to the graph and derivatives are propogated in the reverse order.

There are currently just a few operations, which should be enough for me:
graph.dot(vector, matrix)
graph.relu(tensor, alpha=0.0)
graph.add(tensor_list)
graph.multiply(scalar, tensor)
graph.sigmoid()

graph.concat(vector_list)
graph.split(num, vector_list)
"""

NNLibraryDType = np.float64
NNepsilon = np.float64(1.0e-30)

class Node:
    def __init__(self,graph):
        self.is_variable = False
        if graph is not None: graph.append(self)
    def shape(self):
        return self.value.shape
    def __repr__(self):
        if hasattr(self, 'name'):
            return '<{0}: Node of size {1}>'.format(
                    self.name, self.shape()
                    )
        else:
            return '<Node of size {0}>'.format(
                    self.shape()
                    )
    def __str__(self):
        return self.__repr__()

class DotNode(Node):
    def __init__(self, vector, tensor, graph):  # vector and tensor are nodes
        Node.__init__(self, graph)
        assert len(vector.value.shape)<3
        assert len(tensor.value.shape)==2

        #self.value = np.dot(vector.value, tensor.value)
        self.value = np.dot(vector.value, tensor.value)
        if graph is not None:
            self.d = 0.0*self.value

            self.vector = vector
            self.tensor = tensor

    def backprop(self):
        n = len(self.vector.value.shape)
        if n == 1:
             self.vector.d += np.dot(self.d, self.tensor.value.transpose())
             self.tensor.d += np.outer(self.vector.value, self.d)
        if n == 2:
            self.vector.d += np.dot(self.d, self.tensor.value.transpose())
            self.tensor.d += np.dot(self.vector.value.transpose(), self.d)

def MeanNode(tensor_list, graph):
    tensor_sum = AddNode(tensor, graph)
    out = MultiplyNode(1.0/len(tensor_list), tensor_sum, graph)
    return out


class AddNode(Node):
    def __init__(self, tensor_list, graph):
        Node.__init__(self, graph)
        assert len(tensor_list)>0

        if isinstance(tensor_list[0].value, float):
            self.ns = ()
        else:
            self.ns = len(tensor_list[0].value.shape)


        self.value = 1.0 * tensor_list[0].value

        for t in tensor_list[1:]:
            if isinstance(t.value, float) or len(t.value.shape)==self.ns:
                self.value += t.value
            else:
                self.value += np.expand_dims(t.value, axis=0) # should broadcast

        if graph is not None:
            self.d = 0.0*self.value
            self.tensor_list = tensor_list

    def backprop(self):
        for t in self.tensor_list:
            if isinstance(t.value, float) or len(t.value.shape)==self.ns:
                t.d+=self.d
            else:
                t.d+=np.sum(self.d, axis=0)

class TransposeInPlaceNode(Node):
    def __init__(self, matrix, graph):
        Node.__init__(self, graph)
        self.value = matrix.value.transpose()
        if graph is not None:
            self.d = matrix.d.transpose()
            self.matrix = matrix
    def backprop(self):
        pass  # I don't think that I need to do anything here...

class StackNode(Node):  # the vectors are the rows of the matrix.
    def __init__(self, vectors, graph):
        Node.__init__(self, graph)
        self.value = np.stack([t.value for t in vectors], axis=0)
        if graph is not None:
            self.d = 0.0*self.value
            self.vectors = vectors
    def backprop(self):
        for i in range(len(self.vectors)):
            self.vectors[i].d += self.d[i]

class LogNode(Node):
    def __init__(self, tensor, graph):
        Node.__init__(self, graph)
        self.value = np.log(tensor.value)
        if graph is not None:
            self.d = 0.0 * self.value
            self.tensor = tensor
    def backprop(self):
        self.tensor.d += self.d / self.tensor.value

class VVDotNode(Node):
    def __init__(self, v1, v2, graph):
        assert len(v1.value.shape)==1
        assert len(v2.value.shape)==1
        Node.__init__(self, graph)
        self.value = np.dot(v1.value, v2.value)
        if graph is not None:
            self.d = 0.0

            self.v1 = v1
            self.v2 = v2

    def backprop(self):
        self.v1.d += self.d * self.v2.value
        self.v2.d += self.d * self.v1.value


def DotAdd(vector, matrix, bias, graph):
    dot = DotNode(vector, matrix, graph)
    add = AddNode([dot,bias], graph)
    return add

def RELUDotAdd(vector, matrix, bias, graph, alpha=0.01):
    dot = DotNode(vector, matrix, graph)
    add = AddNode([dot,bias], graph)
    relu = RELUNode(add, graph, alpha=alpha)
    return relu

def NegativeSamplingLoss(correct, wrongs, graph):
    correct = SigmoidNode(correct, graph)
    correct = LogNode(correct, graph)

    loss_list = [correct]

    for wrong in wrongs:
        wrong = MultiplyNode(-1.0, wrong, graph)
        wrong = SigmoidNode(wrong, graph)
        wrong = LogNode(wrong, graph)
        loss_list.append(wrong)

    loss_sum = AddNode(loss_list, graph)
    return MultiplyNode(-1.0,loss_sum, graph)

class TensorMultiplyNode(Node):
    def __init__(self, t1, t2, graph):
        Node.__init__(self, graph)
        self.value = t1.value*t2.value
        if graph is not None:
            self.d = 0.0*self.value

            self.t1 = t1
            self.t2 = t2

    def backprop(self):
        self.t1.d += self.d * self.t2.value
        self.t2.d += self.d * self.t1.value


class MultiplyNode(Node):
    def __init__(self, scalar, tensor, graph):
        Node.__init__(self, graph)
        self.value = scalar * tensor.value
        if graph is not None:
            self.d = 0.0*self.value

            self.scalar = scalar
            self.tensor = tensor

    def backprop(self):
        self.tensor.d += self.scalar * self.d

class ScalarAddNode(Node):
    def __init__(self, scalar, tensor, graph):
        Node.__init__(self, graph)
        self.value = scalar + tensor.value
        if graph is not None:
            self.d = 0.0*self.value

            self.scalar = scalar
            self.tensor = tensor

    def backprop(self):
        self.tensor.d += self.d

class RELUNode(Node):
    def __init__(self, tensor, graph, alpha=0.01):
        Node.__init__(self, graph)
        self.value = np.fmax(tensor.value, alpha*tensor.value)
        if graph is not None:
            self.d = 0.0*self.value

            self.alpha = alpha
            self.tensor = tensor

    def backprop(self):
        self.tensor.d += self.d * ((1.0 * (self.tensor.value > 0)) + (self.alpha * (self.tensor.value <= 0)))

class SigmoidNode(Node):
    def __init__(self, tensor, graph):
        Node.__init__(self, graph)
        self.value = 1.0 / (1.0 + np.exp(-1 * tensor.value))
        if graph is not None:
            self.d = 0.0*self.value

            self.tensor = tensor

    def backprop(self):
        self.tensor.d += self.value * (1-self.value) * self.d

class ConcatNode(Node):
    def __init__(self, vector_list, graph):
        Node.__init__(self, graph)
        self.value = np.concatenate([v.value for v in vector_list], axis=0)
        if graph is not None:
            self.d = 0.0 * self.value

            self.vector_list = vector_list

    def backprop(self):
        index = 0
        for v in self.vector_list:
            length = v.value.shape[0]
            v.d += self.d[index:index+length]
            index+=length

def SplitNode(vector, num_split, graph):
    vlen = vector.value.shape[0]
    assert vlen % num_split == 0
    length = vlen//num_split

    out = [IndexNode(length*i, length, vector, graph) for i in range(num_split)]
    return out

class IndexNode(Node):
    def __init__(self, start, length, tensor, graph):
        Node.__init__(self, graph)
        self.value = tensor.value[start:start+length]
        if graph is not None:
            self.d = 0 * self.value

            self.start = start
            self.length = length
            self.tensor = tensor

    def backprop(self):
        self.tensor.d[self.start:self.start+self.length] += self.d

# WARNING: this node should only be used when we start with a constant and
# only assign indices once.
class AssignIndexInPlaceNode(Node):
    def __init__(self, index, start, source, graph):
        Node.__init__(self, graph)
        self.value = start.value
        self.value[index] = source.value  # overwrites the source node information
        if graph is not None:
            self.d = start.d            # overwrites the d-information

            self.start = start
            self.index = index
            self.source = source

    def backprop(self):
        self.source.d += self.d[self.index]

# https://colah.github.io/posts/2015-08-Understanding-LSTMs/
# C, hin, x, bf.shape = (n)
# Wf.shape = (2n, n)
class LSTM_Parameters:
    def __init__(self, Wf=None, bf=None, Wi=None,
            bi=None, WC=None, bC=None, Wo=None, bo=None):
        assert Wf is not None
        assert bf is not None
        assert Wi is not None
        assert bi is not None
        assert WC is not None
        assert bC is not None
        assert Wo is not None
        assert bo is not None
        self.Wf=Wf
        self.bf=bf
        self.Wi=Wi
        self.bi=bi
        self.WC=Wc
        self.bC=bC
        self.Wo=Wo
        self.bo=bo

def LSTMCell(Cin, hin, x, params, graph):
    hx = ConcatNode([hin, x], graph)
    f = DotAdd(hx, params.Wf, params.bf, graph)
    f = SigmoidNode(f, graph)

    i = DotAdd(hx, params.Wi, params.bi, graph)
    i = SigmoidNode(i, graph)

    Ctilde = DotAdd(hx, params.WC, params.bC, graph)
    Ctilde = TanhNode(Ctilde, graph)

    Cout0 = TensorMultiplyNode(f, params.Cin, graph)
    Cout1 = TensorMultiplyNode(i, params.Ctilde, graph)
    Cout = AddNode([Cout0, Cout1], graph)

    o = DotAdd(hx, params.Wo, params.bo, graph)
    o = SigmoidNode(o, graph)

    tanhCout = TanhNode(Cout, graph)
    hout = TensorMultiplyNode(o, tanhCout, graph)
    return Cout, hout




def GRUCell(hin, x, Wz, Wr, W, graph):
    hx = ConcatNode([hin, x], graph)
    z = DotNode(hx, Wz, graph)
    z = SigmoidNode(z, graph)
    r = DotNode(hx, Wr, graph)
    r = SigmoidNode(r, graph)
    rh = TensorMultiplyNode(r, hin, graph)
    rhx = ConcatNode([rh,x], graph)
    htilde = DotNode(rhx, W, graph)
    htilde = TanhNode(htilde, graph)

    negz = MultiplyNode(-1.0, z, graph)
    negz = ScalarAddNode(1, negz, graph)

    negzh = TensorMultiplyNode(negz, hin, graph)
    zhtilde = TensorMultiplyNode(z, htilde, graph)

    hout = AddNode([negzh, zhtilde], graph)

    return hout

# this is really easy because we do every per-sequence
def DropoutNode(tensor, dropout, graph):
    if dropout is None: return tensor
    np_mask = 1.0/(1.0-dropout)*(np.random.uniform(size=tensor.value.shape)>dropout)
    mask = ConstantNode(np_mask, graph)
    return TensorMultiplyNode(tensor, mask, graph)

class GRUbParameters(Node):
    def __init__(self, h_size, graph, x_size = None, name=''):
        if x_size is None: x_size = h_size
        self.Wz = VariableNode([x_size+h_size, h_size], graph, name=name+'.Wz')
        self.bz = VariableNode([h_size], graph, name=name+'.bz')
        self.Wr = VariableNode([x_size+h_size, h_size], graph, name=name+'.Wr')
        self.br = VariableNode([h_size], graph, name=name+'.br')
        self.W = VariableNode([h_size+x_size,h_size], graph, name=name+'.W')

        self.vs = [self.Wz, self.bz, self.Wr, self.br, self.W]
        self.rvs = [self.Wz, self.Wr, self.W]

def GRUbCell(hin, x, GRUparams, graph, dropout=None):
    hx = ConcatNode([hin, x], graph)
    # print hin.value.shape, x.value.shape
    # print hx.value.shape, GRUparams.Wz.value.shape
    z = DotNode(hx, GRUparams.Wz, graph)
    z = AddNode([z, GRUparams.bz], graph)
    z = SigmoidNode(z, graph)
    r = DotNode(hx, GRUparams.Wr, graph)
    r = AddNode([r, GRUparams.br], graph)
    r = SigmoidNode(r, graph)
    rh = TensorMultiplyNode(r, hin, graph)
    rhx = ConcatNode([rh,x], graph)
    htilde = DotNode(rhx, GRUparams.W, graph)
    htilde = TanhNode(htilde, graph)
    htilde = DropoutNode(htilde, dropout, graph)

    negz = MultiplyNode(-1.0, z, graph)
    negz = ScalarAddNode(1, negz, graph)

    negzh = TensorMultiplyNode(negz, hin, graph)
    zhtilde = TensorMultiplyNode(z, htilde, graph)

    hout = AddNode([negzh, zhtilde], graph)

    return hout




class SingleIndexNode(Node):
    def __init__(self, index, tensor, graph):
        Node.__init__(self, graph)
        self.value = tensor.value[index]
        if graph is not None:
            self.d = 0 * self.value

            self.index = index
            self.tensor = tensor

    def backprop(self):
        self.tensor.d[self.index] += self.d

class TanhNode(Node):
    def __init__(self, vector, graph):
        Node.__init__(self, graph)
        self.value = np.tanh(vector.value)
        if graph is not None:
            self.d = 0.0 * self.value

            self.vector = vector

    def backprop(self):
        self.vector.d += (1-(self.value ** 2)) * self.d

class SoftmaxNode(Node):
    def __init__(self, vector, graph):
        Node.__init__(self, graph)
        self.value = softmax(vector.value)
        if graph is not None:
            self.d = 0.0 * self.value

            self.vector = vector

    def backprop(self):
        if len(self.value.shape)==1:
            self.vector.d += self.value * self.d
            self.vector.d -= self.value * (np.dot(self.d, self.value))
        else:
            assert len(self.value.shape)==2
            self.vector.d += self.value * self.d
            self.vector.d -= self.value * np.sum(self.d, self.value, axis=1, keepdims=True)

# def softmax(vector):
#     # only works with a vector, because I'm lazy
#     z = np.exp(vector-np.amax(vector))
#     z /= (np.sum(z))
#     return z
def softmax(vector):
    if len(vector.shape)==1:
        z = np.exp(vector-np.amax(vector))
        z /= (np.sum(z))
        return z
    else:
        # only works with a vector, because I'm lazy
        z = np.exp(vector-np.amax(vector, axis=1, keepdims=True))
        z /= (np.sum(z, axis=1, keepdims = True))
        return z

# log softmax computes -log(softmax(x))
# there's a bit of redundent computation here, but whatever
def negative_log_softmax(vector):
    return -1.0*log_softmax(vector)

def log_softmax(vector):
    z = vector-np.amax(vector, axis=-1, keepdims=True)
    z -= np.log(np.sum(np.exp(z), axis=-1, keepdims = True))
    return z

class SoftmaxCrossEntropyLoss(Node):
    def __init__(self, correct, vector, graph):
        Node.__init__(self, graph)
        self.softmax = softmax(vector.value)
        #self.value = -np.log(self.softmax[correct])
        self.value = negative_log_softmax(vector.value)[correct]
        if graph is not None:
            self.d = 0.0

            self.vector = vector
            self.correct = correct

    def backprop(self):
        self.softmax[self.correct]-=1
        self.vector.d += self.d * self.softmax
        self.softmax[self.correct]+=1

def XavierInitializer(shape):
    if len(shape)==0: return 0

    total = np.sum(shape)
    if total == 0: return np.zeros(size=())
    x = (6.0/total)**0.5
    out = np.random.uniform(low=-x, high=x, size=shape).astype(NNLibraryDType)
    return out

class VariableNode(Node):
    def __init__(self, shape, graph, initializer=XavierInitializer, value=None,
        name='variable'):
        Node.__init__(self, graph)
        self.is_variable = True
        self.name = name
        if value is None:
            self.value = initializer(shape)
        else:
            self.value = value
            shape = value.shape
        self.d = np.zeros(shape, dtype=NNLibraryDType)  # variables can always have a d.  We can allow that.

    def backprop(self):
        pass

    def load(self, other):
        self.value[:] = other.value
        self.d[:] = other.d

        # also need adam_m and adam_v
        if hasattr(self, 'adam_m') and hasattr(other, 'adam_m'):
            self.adam_m[:] = other.adam_m
        if hasattr(self, 'adam_v') and hasattr(other, 'adam_v'):
            self.adam_v[:] = other.adam_v

class L2Node(Node):
    def __init__(self, constant, tensor, graph):
        Node.__init__(self, graph)
        self.value = constant * (np.linalg.norm(tensor.value)**2) / 2.0
        if graph is not None:
            self.d = 0.0

            self.tensor = tensor
            self.constant = constant

    def backprop(self):
        self.tensor.d += self.constant * self.d * self.tensor.value

class ReshapeNode(Node): # DESTRUCTIVE
    def __init__(self, tensor, new_shape, graph):
        Node.__init__(self, graph)
        self.old_shape = tensor.value.shape
        tensor.value.shape = new_shape
        tensor.d.shape = new_shape
        self.value = tensor.value
        if graph is not None:
            self.d = tensor.d
            self.tensor = tensor
    def backprop(self):
        self.tensor.d.shape = self.old_shape
        self.tensor.value.shape = self.old_shape

class ConstantNode(Node):
    def __init__(self, value, graph):
        Node.__init__(self, graph)
        self.value = value
        if graph is not None:
            self.d = 0.0 * self.value

    def backprop(self):
        pass

def ZerosNode(shape, graph):
    matrix = np.zeros(shape, dtype=NNLibraryDType)
    return ConstantNode(matrix, graph)


"""
The class that stores the entire
computational graph.
"""

class ComputationalGraph(object):
    def __init__(self, nodes=[]):
        self.nodes = nodes[:]

    def append(self, node):
        self.nodes.append(node)

    def backprop(self, final_loss):
        # make sure the derivatives for all the variables are zero.
        # we assume that the other deriviatives are zero because they
        # were just constructed.
        for node in self.nodes:
            if node.is_variable:
                node.d *= 0.0

        final_loss.d += 1
        while len(self.nodes) > 0:
            self.nodes.pop().backprop()

class GradientDescentOptimizer:
    def __init__(self, vs, alpha=0.001):
        self.a = alpha
        self.vars = vs
        self.steps = 0

    def step(self):
        for var in self.vars:
            var.value -= self.a * var.d
            var.d *= 0.0
        self.steps += 1

    def reset(self):
        pass


class AdamOptimizer:
    def __init__(self, vs, alpha=0.001, beta1=0.9,
            beta2=0.999, epsilon=1.0e-8, access_hold=None):
        # vars is a list of node vars.  They should already be
        # initialized.
        self.vars = vs
        for var in vs:
            var.adam_m = 0.0*var.value
            var.adam_v = 0.0*var.value

        self.steps = 0
        self.a = alpha
        self.b1 = beta1
        self.b2 = beta2
        self.e = epsilon

        self.gradient_clipping = None

    def update_learning_rate(self, new_rate):
        self.a = new_rate

    def minimize(self, d_vars=None, p=None):
        if d_vars is None:
            for i in range(self.vars):
                self.vars[i].d=d_vars[i]

        # if p is not None:
        #     # p is a Pool() object
        #     self.parallel_minimize(p)
        #     return

        self.steps += 1
        assert(len(d_vars) == len(self.vars))
        for i in range(len(self.vars)):
            var = self.vars[i]
            #g = var.d
            g=d_vars[i]

            # gradient_clipping
            g2=np.square(g)
            if self.gradient_clipping is not None:
                g2mean = np.mean(g2)
                if g2mean > self.gradient_clipping ** 2:
                    g *= self.gradient_clipping / np.sqrt(g2mean)
                    g2=np.square(g)

            var.adam_m *= self.b1
            var.adam_m += (1.0-self.b1) * g
            var.adam_v *= self.b2
            var.adam_v += (1.0-self.b2) * g2
            mhat = var.adam_m/(1 - (self.b1 ** self.steps))
            vhat = var.adam_v/(1 - (self.b2 ** self.steps))
            var.value -= self.a * mhat/(np.sqrt(vhat) + self.e)

    # def parallel_minimize(self, p):
    #     valmvlist = p.map(self.parallel_minimize_function(var), self.vars)
    #     for valmv, var in zip(valmvlist, self.vars):
    #         var.value, var.adam_m, var.adam
    #
    #
    # def parallel_minimize_function(self, var):

    def reset(self):
        for var in self.vars:
            var.adam_m *= 0.0
            var.adam_v *= 0.0
