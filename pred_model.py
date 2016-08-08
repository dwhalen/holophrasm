import nnlibrary as nn
from models import *
import data_utils5 as data_utils
import numpy as np
import random

def validation_data_set(lm):  # lm is a language model
    return lm.validation_proof_steps

def test_data_set(lm):
    return lm.test_proof_steps

def training_data_set(lm):
    np.random.seed()
    return np.random.permutation(lm.training_proof_steps)


class Config(DefaultConfig):
    def __init__(self, language_model):
        # use the standard config initialization
        DefaultConfig.__init__(self, language_model)
        self.p.bidirectional = False
        self.p.out_layers = 1

        self.p.negative_samples = 4  # the number of negative samples


class Variables(DefaultVariables):
    def __init__(self, config):
        DefaultVariables.__init__(self, config)
        # define the specific variables for this section
        r = self.config.p.r
        out_layers = self.config.p.out_layers

        # the parameters for the input section
        self.main_hyp_gru_block = self.add_GRUb_block('main_hyp', bidirectional = self.config.p.bidirectional)
        self.main_statement_gru_block = self.add_GRUb_block('main_statement', bidirectional = self.config.p.bidirectional)
        self.prop_hyp_gru_block = self.add_GRUb_block('main_hyp', bidirectional = self.config.p.bidirectional)
        self.prop_statement_gru_block = self.add_GRUb_block('main_statement', bidirectional = self.config.p.bidirectional)

        self.main_forward_start = [nn.VariableNode([r], None, name='main_forward_h_start'+str(i)) for i in range(self.config.p.gru_depth)]
        self.vs += self.main_forward_start
        if self.config.p.bidirectional:
            self.main_backward_start = [nn.VariableNode([r], None, name='main_backward_h_start'+str(i)) for i in range(self.config.p.gru_depth)]
            self.vs+=self.main_backward_start
        else:
            self.main_backward_start=None

        self.prop_forward_start = [nn.VariableNode([r], None, name='prop_forward_h_start'+str(i)) for i in range(self.config.p.gru_depth)]
        self.vs += self.prop_forward_start
        if self.config.p.bidirectional:
            self.prop_backward_start = [nn.VariableNode([r], None, name='prop_backward_h_start'+str(i)) for i in range(self.config.p.gru_depth)]
            self.vs+=self.prop_backward_start
        else:
            self.prop_backward_start=None

        out_size = r * self.config.p.gru_depth * (2 if self.config.p.bidirectional else 1)
        self.main_first_W = nn.VariableNode([out_size, r], None, name='main_first_W')
        self.main_first_b = nn.VariableNode([r], None, name='main_first_b')
        self.prop_first_W = nn.VariableNode([out_size, r], None, name='prop_first_W')
        self.prop_first_b = nn.VariableNode([r], None, name='prop_first_b')

        self.main_Ws = [nn.VariableNode([r, r], None, name='main_W_'+str(i)) for i in range(self.config.p.out_layers)]
        self.main_bs = [nn.VariableNode([r], None, name='main_b_'+str(i)) for i in range(self.config.p.out_layers)]
        self.prop_Ws = [nn.VariableNode([r, r], None, name='prop_W_'+str(i)) for i in range(self.config.p.out_layers)]
        self.prop_bs = [nn.VariableNode([r], None, name='prop_b_'+str(i)) for i in range(self.config.p.out_layers)]

        self.W = nn.VariableNode([r,r], None, name='W')
        self.vs+=[self.W, self.main_first_W, self.prop_first_W, self.main_first_b,
                self.prop_first_b]+self.main_Ws + self.main_bs+self.prop_Ws+self.prop_bs
        self.rvs+=[self.W, self.main_first_W, self.prop_first_W]+self.main_Ws+self.prop_Ws

        # we always need to add the trainer
        self.add_trainer()

class Model(DefaultModel):
    def __init__(self, variables, config, proof_step, train=False):
        ''' this is the model.  As a single pass, it processes the
        inputs, and computes the losses, and runs a training step if
        train.
        '''
        DefaultModel.__init__(self, config, variables, train=train)

        # fix the random seed
        if not self.train:
            np.random.seed(proof_step.context.number +
                    + proof_step.prop.number + proof_step.tree.size())

        main = self.main_get_vector(
                proof_step.tree, proof_step.context.hyps, proof_step.context.f
                )

        main = nn.DotNode(main, self.v.W, self.g)

        # get a list [right prop, wrong prop 0, ..., wrong_prop n]
        props = self.get_props(proof_step)

        ###DEBUG
        #if not self.train: print [p.label for p in props]
        ###DEBUG
        out_vectors = [self.prop_get_vector(prop.tree, prop.hyps, prop.f)
                for prop in props]
        stacked = nn.StackNode(out_vectors, self.g)
        stacked = nn.TransposeInPlaceNode(stacked, self.g)

        logits = nn.DotNode(main, stacked, self.g)
        cross_entropy = nn.SoftmaxCrossEntropyLoss(0, logits, self.g)
        self.loss = nn.AddNode([self.loss, cross_entropy], self.g)

        accuracy = 1 * (np.argmax(logits.value) == 0)
        self.outputs = [cross_entropy.value, accuracy, 1.0/len(props)]
        self.output_counts = [1, 1, 1]

        # perform the backpropagation if we are training
        if train:
            self.g.backprop(self.loss)


    def get_props(self, proof_step):
        # generate a list of valid propositions that could apply to the prop,
        # choose some number of them, and then return the correct prop
        # and the wrong props
        wrong_props = self.config.p.negative_samples
        labels = self.config.lm.searcher.search(proof_step.tree, proof_step.context, max_proposition=proof_step.context.number, vclass='|-')
        labels.remove(proof_step.prop.label)  # remove the correct label
        wrong_props = min(wrong_props, len(labels))
        #if wrong_props == len(labels): print 'WARNING: VERY FEW ({0}) APPLICABLE PROPOSITIONS'.format(len(labels))
        rand = np.random.choice(len(labels), wrong_props, replace=False)
        rc = [labels[i] for i in rand]
        #rc = random.sample(labels, wrong_props)
        wrong_props = [self.config.lm.database.propositions[label] for label in rc]
        return [proof_step.prop]+wrong_props

    def main_get_vector(self, statement, hyps, f):
        return self.get_vector(
                statement, hyps, f,
                self.v.main_statement_gru_block, self.v.main_hyp_gru_block,
                self.v.main_forward_start, self.v.main_backward_start,
                self.v.main_first_W, self.v.main_first_b, self.v.main_Ws,
                self.v.main_bs
                )

    def prop_get_vector(self, statement, hyps, f):
        return self.get_vector(
                statement, hyps, f,
                self.v.prop_statement_gru_block, self.v.prop_hyp_gru_block,
                self.v.prop_forward_start, self.v.prop_backward_start,
                self.v.prop_first_W, self.v.prop_first_b, self.v.prop_Ws,
                self.v.prop_bs
                )

    def get_vector(self, statement, hyps, f, statement_gru, hyps_gru,
            forward_start, backward_start, first_W, first_b, Ws, bs):
        in_string, in_parents, in_left, in_right, in_params, depths, \
                parent_arity, leaf_position, arity = self.parse_statement_and_hyps(
                statement, hyps, f, statement_gru, hyps_gru)

        #print in_string
        to_middle = self.gru_block(forward_start, in_string, in_params,
                hs_backward=backward_start, parents=in_parents,
                left_siblings=in_left, right_siblings=in_right,
                bidirectional=self.config.p.bidirectional,
                structure_data = zip(depths, parent_arity, leaf_position, arity),
                feed_to_attention=False)

        h = nn.ConcatNode(to_middle, self.g)
        h = nn.DropoutNode(h, self.dropout, self.g)
        h = nn.RELUDotAdd(h, first_W, first_b, self.g)
        h = nn.DropoutNode(h, self.dropout, self.g)
        for i in range(self.config.p.out_layers):
            h = nn.RELUDotAdd(h, Ws[i], bs[i], self.g)
            h = nn.DropoutNode(h, self.dropout, self.g)

        return h

    def parse_statement_and_hyps(self, statement, hyps, f, statement_gru, hyps_gru):
        # statement is a tree, hyps is a list of hyps (we'll filter for e-type)
        random_replacement_dict = self.config.lm.random_replacement_dict_f(f=f)

        statement = statement.copy().replace_values(random_replacement_dict)
        hyps = [h.tree.copy().replace_values(random_replacement_dict) for h in hyps if h.type=='e']

        # and get the graph structures
        statement_graph_structure = TreeInformation([statement],
                start_symbol=None, intermediate_symbol='END_OF_HYP',
                end_symbol='END_OF_SECTION')
        hyps_graph_structure = TreeInformation(hyps,
                start_symbol=None, intermediate_symbol='END_OF_HYP',
                end_symbol='END_OF_SECTION')
        return merge_graph_structures(
                [statement_graph_structure, hyps_graph_structure],
                [statement_gru, hyps_gru])


''' and now the interface for the autmated prover.
This can be ensembled by feeding in multiple variable objects into vs'''
def get_prop_vector(vs, prop):
    vectors = [PropModel(v, prop).vector for v in vs]
    return sum(vectors)/len(vectors)

def get_main_vector(vs, tree, context):
    vectors = [MainModel(v, tree, context).vector for v in vs]
    return sum(vectors)/len(vectors)

class MainModel(Model):
    def __init__(self, v, tree, context):
        # type = 'main' or 'prop'
        DefaultModel.__init__(self, v.config, v, train=False)
        self.g = None  # don't remember all of the stuff in the graph.

        main = self.main_get_vector(
                tree, context.hyps, None # context.f
                )
        self.vector = main.value

class PropModel(Model):
    def __init__(self, v, prop):
        # type = 'main' or 'prop'
        DefaultModel.__init__(self, v.config, v, train=False)
        self.g = None  # don't remember all of the stuff in the graph.

        prop = self.prop_get_vector(
                prop.tree, prop.hyps, prop.f
                )
        prop = np.dot(self.v.W.value, prop.value)
        self.vector = prop
