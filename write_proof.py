from shutil import move
import os
import fileinput
import pickle as pickle

in_file = 'set.mm'
out_file = 'modified_set.mm'
temp_file = 'temp_set.mm'

NUM_ADDITIONAL_SET_VARIABLES = 20

def reset():
    if os.path.exists(out_file):
        os.remove(out_file)
    write({}, reset=True)

def d_condition_string(label):
    out= ["$d", label]
    for i in range(NUM_ADDITIONAL_SET_VARIABLES):
        name = 'SetVar'+str(i)
        out.append(name+'v')
    out.append("$.\n")
    return out
    
def reset_string():
    # this is a string that goes at the beginning of the thing
    # and describes the additional set variables and the
    # d statements that go with them.
    out = []
    for i in range(NUM_ADDITIONAL_SET_VARIABLES):
        name = 'SetVar'+str(i)
        out.append('$v {1} $.\n{0} $f set {1} $.\n'.format(name, name+'v'))
    out+= ["$d"]
    for i in range(NUM_ADDITIONAL_SET_VARIABLES):
        name = 'SetVar'+str(i)
        out.append(name+'v')
    out.append("$.\n")
    return out

def write(dictionary, reset=False):
    proof = None
    added_proofs = 0
    
    need_to_add_d_statement = None
    path = in_file if reset else out_file
    #full_path = os.path.realpath(filename)

    with open(path, 'r') as f:
        tokens = f.read().split()

    out = []
    if reset:
        out += reset_string()
    
    label = None
    in_statement = False
    will_start_proof = False
    in_proof = True
    comment_depth = 0
    writing = True
    for token in tokens:
        assert comment_depth >= 0
        # check if it's a comment
        if token == '$(':
            comment_depth+=1
            continue
        elif token == '$)':
            comment_depth-=1
            continue

        # parse stuff if we're not in a comment

        if comment_depth>0:
            # we're in a comment!  ignore everything!
            continue

        if will_start_proof:
            # hopefully this is the start of the proof
            assert token == '$p'
            in_proof = True
            will_start_proof = False
        if in_proof and token == '$=':
            # this is the part we overwrite
            writing = False
            out.append('$=')
            out.append(proof)
            out.append('$.')
            added_proofs += 1

        if writing:
            out.append(token)
        
        if need_to_add_d_statement is not None and token != '$.':
            need_to_add_d_statement.append(token)
            
        if token == '$f':
            need_to_add_d_statement = []
        
        if token in ['$c', '$v', '$d', '$f', '$e','$a','$p','$[']:
            #print out
            #print
            #print token
            assert not in_statement
            in_statement=True
        elif token in ['$','$.','$]']:
            out.append("\n")
            assert in_statement
            if need_to_add_d_statement is not None and reset:
                need_to_add_d_statement = need_to_add_d_statement[1:]
                #print need_to_add_d_statement
                for var in need_to_add_d_statement:
                    out += d_condition_string(var)
                need_to_add_d_statement = None
            in_statement = False
            label = None
            writing = True
            in_proof = False
        elif in_statement:
            pass
        elif token in ['${', '$}']:
            out.append("\n")
            pass
        else:
            # this must be a label!
            if label is not None:
                print('label', label)
            assert label is None
            label = token
            if label not in dictionary:
                continue
            proof = dictionary[label]
            will_start_proof = True

    # write the file
    out = ' '.join(out)
    out = out.replace("\n ","\n")
    out = out.replace(" \n","\n")
    if reset:
        out = out.replace("$c set $.\n", "")
        out = "$c set $.\n"+out
    with open(out_file, 'w') as handle:
        handle.write(out)
    
    print('added', added_proofs, 'proofs')


'''
This is a class that will be used to store a proof object, which will subsequently be pickled and saved.
'''
def forget_proofs(directory='searcher'):
    directory += '/proofs'
    for root, dirs, files in os.walk(directory):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

def proof_exists(label, directory='searcher'):
    return os.path.exists(directory+'/proofs/'+label)

def load_proof(label, directory='searcher'):
    assert proof_exists(label, directory=directory)
    file_path = directory+'/proofs/'+label
    #print 'loading proof at '+file_path
    with open(file_path, 'rb') as handle:
        proof = pickle.load(handle)
    return proof

def write_all_known_proofs(directory='searcher'):
    d = {}
    new_directory=directory+'/proofs'
    proof_list = os.listdir(new_directory)
    for label in proof_list:
        if label[0]=='.': continue
        proof = load_proof(label, directory)
        d[proof.label] = proof.proof
    labels = list(d.keys())
    print('Writing proofs for '+str(len(labels))+' propositions:')
    # print labels
    # add create the new set.mm
    write(d, reset=True)  # include a reset.

class Proof:
    def __init__(self, label, proof, passes):
        self.label = label
        self.proof = proof
        self.passes = passes

    def save(self, directory='searcher'):
        directory += '/proofs'
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = directory +'/'+self.label

        if os.path.exists(file_path):
            print('Failed to write to '+file_path+': proof already exists')
            return

        with open(file_path, 'wb') as handle:
            pickle.dump(self, handle)

    def add_to_file(self):
        write({self.label:self.proof})
