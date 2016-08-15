import write_proof
def run_all_props_with_current_settings(passes, timeout):
    i=-1
    num_proven = 0
    num_this_proven = 0
    for prop in vprops:
        #if prop.number > 5000: continue
        i+=1
        print 'proposition {0} ({1}): {2}/{3} (with {4} steps)'.format(prop.label,prop.number,i,len(vprops), prop.num_entails_proof_steps)
        if prop.type == 'a':
            continue
        if write_proof.proof_exists(prop.label, directory=directory):
            num_proven += 1
            continue
        # set up the searcher
        searcher = proof_search.ProofSearcher(prop, language_model, directory=directory, timeout=timeout)

        # run the searcher
        searcher.run(passes, multi=True, threads=threads, print_output=False, clear_output=False)

        if searcher.proven():
            num_proven += 1
            num_this_proven += 1
            searcher.proof_object()
            
        print 'proven {0}/{1} this: {3}, {2:5.1f}%'.format(num_proven, i+1, 100 *num_proven/(i+1.0), num_this_proven)









'''
and the start of the actual script
'''


directory = 'searcher'
threads = 15
passes = 200

# set up the language model
import time
from tree_parser import *
from data_utils5 import *
import cPickle as pickle
import numpy as np

timeout = int(input("timeout (min)? "))
passes = int(input("num passes? "))
beams = int(input("search beam width? "))
hyp_bonus = 1.0*int(input("hyp bonus? (0 in the paper, but 10 may give better results)"))

with open('lm', 'rb') as handle:
    database = pickle.load(handle)
database.remember_proof_steps = False
language_model = LanguageModel(database)


x = [(p.num_entails_proof_steps, p.number, p.label) for p in language_model.test_propositions
             if p.vclass == '|-' and p.type=='p']
x.sort()
_, _, labels = zip(*x)
vprops = [language_model.database.propositions[l] for l in labels]

# set up the interface
import proof_search

for sets in [True, False]:
    proof_search.HYP_BONUS = hyp_bonus
    proof_search.BEAM_SIZE = beams
    proof_search.gen_model_beam_search.PERMIT_NEW_SETS = sets
    run_all_props_with_current_settings(passes, timeout)
