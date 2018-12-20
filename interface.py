'''
This builds up the interface for the proof search module.
'''

import gen_model_beam_search
import pred_model as pred_model_run
import payout_model_5_train as payout_model_run
from beam_search import *
import os
import sys
import numpy as np
import pickle as pickle
import data_utils5 as data_utils
import nnlibrary as nn

import constructor_list

NUM_ALLOWED_CONSTRUCTORS = None
DISALLOWED_PROPS = ['idi', 'dummylink']
PRED_ENSEMBLE = 1
PRED_CACHE_ENSEMBLE = 1
PAYOUT_ENSEMBLE = 1
GEN_ENSEMBLE = 1
PAYOUT_SCALE = 1.0 # Chosen to make the spread of payouts roughly uniform over 0.5-1.0.

if NUM_ALLOWED_CONSTRUCTORS is None:
    ALLOWED_CONSTRUCTORS = None
else:
    ALLOWED_CONSTRUCTORS = set(constructor_list.order[:NUM_ALLOWED_CONSTRUCTORS])



class ProofInterface:
    def __init__(self, lm, recalculate_props=False, skip_payout=False, directory='searcher'):
        self.lm = lm

        # load all the variables and parameters
        # I'm fixing the file locations by hand because lazy.
        self.gen_config = gen_model_beam_search.Config(lm)
        self.gen_config.load(directory+'/gen.parameters')
        self.gen_var = gen_model_beam_search.Variables(self.gen_config)
        self.gen_var.load(directory+'/gen.weights')

        self.pred_config = pred_model_run.Config(lm)
        self.pred_config.load(directory+'/pred.parameters')
        self.pred_var = pred_model_run.Variables(self.pred_config)
        self.pred_var.load(directory+'/pred.weights')

        self.skip_payout = skip_payout
        if not skip_payout:
            self.payout_config = payout_model_run.Config(lm)
            self.payout_config.load(directory+'/payout.parameters')
            self.payout_var = payout_model_run.Variables(self.payout_config)
            self.payout_var.load(directory+'/payout.weights')

        # beam search interface
        self.bsi = gen_model_beam_search.BeamSearchInterface([self.gen_var]*GEN_ENSEMBLE)

        # remember the answer so that we don't need to constantly recalculate it
        file_path = directory+'/pred_database'
        if os.path.isfile(file_path) and not recalculate_props:
            print('loading proposition vectors')
            with open(file_path, 'rb') as handle:
                self.pred_database = pickle.load(handle)
        else:
            print('using proposition vectors at '+file_path)
            self.initialize_pred(file_path)



    def initialize_pred(self, file_path):
        # this initializes all of the proposition vectors in database,
        # so that we can call them quickly when we need to.
        # this should include the multiplication
        #self.pred_database = [pred_model_run.get_prop_vector([self.pred_var]*ENSEMBLE, prop) for prop in self.lm.database.propositions_list)]
        self.pred_database = []
        for i, prop in enumerate(self.lm.database.propositions_list):
            sys.stdout.write('\rvectorizing proposition '+str(i))
            sys.stdout.flush()
            self.pred_database.append(pred_model_run.get_prop_vector([self.pred_var]*PRED_CACHE_ENSEMBLE, prop))
        print('\rdone adding propositions')
        self.pred_database = np.stack(self.pred_database, axis=0)

        # save the database
        with open(file_path, 'wb') as handle:
            pickle.dump(self.pred_database, handle)

    def payout(self, tree, context):
        assert not self.skip_payout, "Attempted to evaluate payout on an interface without with skip_payout=True."
        return payout_model_run.get_payout([self.payout_var]*PAYOUT_ENSEMBLE, tree, context)

    def initialize_payout(self, context):
        #context.difficulty = self.payout(context.tree, context)
        pass

    def get_payout(self, tree, context):
        ''' note: the test dataset had the following histogram for delta,
        [ 0.05543478,  0.01594203,  0.01376812,  0.00797101,  0.00398551,
        0.00144928,  0.00144928] using bin sizes of 0.5, i.e. 0-0.5, 0.5-1,...
        '''
        include_score = self.payout(tree, context)
        #print 'getting payout'
        # return difficulty
        # delta = (context.difficulty - difficulty) * PAYOUT_SCALE
        # delta = (difficulty) * PAYOUT_SCALE
        # return delta
        return np.exp(include_score)/(1.0+np.exp(include_score))

    def props(self, tree, context):
        # returns the sorted list of propositions.
        vec = pred_model_run.get_main_vector([self.pred_var]*PRED_ENSEMBLE, tree, context)
        labels = self.lm.searcher.search(tree, context, max_proposition=context.number, vclass='|-')
        # we disallow these two particular propositions
        for label in DISALLOWED_PROPS:
            if label in labels:
                labels.remove(label)
        prop_nums = [self.lm.database.propositions[label].number for label in labels]
        submatrix =  self.pred_database[np.array(prop_nums), :]
        logits = np.dot(submatrix, vec)

        # print labels, nn.log_softmax(logits)
        # input("Press Enter to continue...")
        return labels, logits - np.max(logits)  # rescaled log-probability
        #return labels, nn.log_softmax(logits)

        # # we don't need to do the sorting here
        # prop_indices = np.argsort(logits)[::-1]
        # sorted_labels = [labels[index] for index in prop_indices]
        # probs = nn.log_softmax(logits)
        # probs = probs[prop_indices]
        # return sorted_labels, probs  # highest to lowest

    def apply_prop(self, tree, context, prop_name, n=10, return_replacement_dict=False):
        # shortcut if the unconstrainer arity is 0
        prop = self.lm.database.propositions[prop_name]
        if prop.unconstrained_arity() == 0:
            return [(0.0, self.lm.simple_apply_prop(tree, prop, context, vclass='|-'))]

        ''' in this case, params = tree, context, prop_name '''
        beam_searcher = BeamSearcher(self.bsi, (tree, context, prop_name, ALLOWED_CONSTRUCTORS, return_replacement_dict))
        out = beam_searcher.best(n, n, n) #(width, k, num_out)  See notes regarding accuracy
        #print 'out', out
        return out

    def is_tautology(self, tree, context):
        '''
        check to see wether the tree is tautologically true.
        We can do this *really* quickly, so we might as well.

        There's a little redundency in that we calculate the
        viable props twice, but it's a pretty quick process.

        Returns None if not a tautology, otherwise returns a
        label for a proposition that proves it immediately.
        '''
        labels = self.lm.searcher.search(tree, context, max_proposition=context.number, vclass='|-')
        tauts = set(labels).intersection(self.lm.tautologies)
        if len(tauts)==0:
            return None
        else:
            return tauts.pop()
