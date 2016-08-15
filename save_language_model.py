import cPickle as pickle
from data_utils5 import *

text = file_contents()
database = meta_math_database(text,n=100000, remember_proof_steps=True)
language_model = LanguageModel(database)
# forget the training proof steps, freeing up about 10G of memory
language_model.training_proof_steps = None
language_model.test_proof_steps = None
language_model.validation_proof_steps = None
language_model.all_proof_steps = None
for p in language_model.database.propositions_list:
    p.entails_proof_steps = None
with open('lm', 'wb') as handle:
    pickle.dump(database, handle)