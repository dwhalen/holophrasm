import time
from tree_parser import *
from data_utils5 import *
import os
import withpool

text = file_contents()
database = meta_math_database(text,n=40000, remember_proof_steps=True)
print()
lm = LanguageModel(database)

saved_interface = None



# import
import build_payout_data_set as pd
pd.initialize_interface(lm, 'searcher')


valp = lm.validation_propositions
testp = lm.test_propositions
trainp = lm.training_propositions

chunk_size = len(trainp)//8
chunks = [
valp,
testp,
trainp[:chunk_size],
trainp[chunk_size:2*chunk_size],
trainp[2*chunk_size:3*chunk_size],
trainp[3*chunk_size:4*chunk_size],
trainp[4*chunk_size:5*chunk_size],
trainp[5*chunk_size:6*chunk_size],
trainp[6*chunk_size:7*chunk_size],
trainp[7*chunk_size:]
]

def process_chunk(x):
    i,j = x
    print('on chunk', i, 'item', j, '/', len(chunks[i]))
    return pd.PropositionsData(chunks[i][j])

for i, chunk in enumerate(chunks):
    filename = 'payout_data_'+str(i)
    if os.path.exists(filename):
        continue

    # okay.  we're doing this one.
    # claim it.
    with open(filename, 'wb') as handle:
        pickle.dump(None, handle)

    # do the stuff.
    print('generating chunk: '+filename)
    with withpool.Pool(None) as pool:
        allpds = pool.map(process_chunk, [(i,j) for j in range(len(chunk))], chunksize=1)

    print('saving '+filename)
    with open(filename, 'wb') as handle:
        pickle.dump(allpds, handle)

#
# def validation_data(n):
#     print 'starting item',n
#     return pd.PropositionsData(valp[n])

''' let's do this in chunks


import withpool
with withpool.Pool(8) as pool:
    start = time.time()
    allpds = {}
    allpds['validation'] = pool.map(validation_data, range(len(valp)), chunksize=1)
    print (time.time()-start), (time.time()-start)/len(valp)
    allpds['test'] = pool.map(test_data, range(len(testp)), chunksize=1)
    print (time.time()-start), (time.time()-start)/(len(valp)+len(testp))
    allpds['training'] = pool.map(training_data, range(len(trainp)), chunksize=1)
    print (time.time()-start), (time.time()-start)/(len(valp)+len(testp)+len(trainp))


print 'saving database'

import pickle
with open('payout_data','wb') as handle:
    pickle.dump(allpds, handle)
    '''
