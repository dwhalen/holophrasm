# Given a statement, find the parse tree that led to it.

from tree_parser import *
import pickle   #pickle.dump(database.proof_datas, open(output_file,'wb'))

from copy import deepcopy
from tree import *

#output_file = 'tree_parse_data'

# class InitialStringSearcher:
#     def __init__(self):
#         self.nodes = {}
#         self.return_value = None
#
#     def add(self,string,value):
#         return self.add()
#
#     def add_with_index(self,string, value,index):
#         if index==len(string):
#             self.value=value
#         char = string[index]
#         if char not in self.nodes:
#             self.nodes[char] = InitialStringSearcher()
#         self.nodes[char].add_with_index(string,value,index+1)


class InitialStringSearcher:
    def __init__(self):
        self.known_values = {}
        #self.known_lengths = {}

    def add(self,string,value):
        self.known_values[tuple(string)]=value
        #print 'found new substring', value[0]
        #self.known_lengths[string]=len(string)

    def find_longest(self,string):
        #label=any(label for label in self.known_values if is_initial_string(label,string))
        for l in self.known_values:
            if is_initial_string(l,string): return self.known_values[l]
        return False

    def clear(self):
        self.known_values = {}

class StatementParser:
    def __init__(self,database):
        # this is a really ugly hack.
        self.wff_variables = ['ph','ps','ch','th','et','ze','si','rh','mu','la','ka']

        self.search = InitialStringSearcher()
        self.failed_parses = set()

        self.propositions = database.propositions
        # self.propositions = {}
        # for label in database.propositions:
        #     statement = database.propositions[label].statement
        #     if statement[0] != '|-': # used in the parse tree
        #         self.propositions[label] = database.propositions[label]#statement

        # add all the variables we ever use to self.variables
        # self.variables = set()
        # for label in database.propositions:
        #     self.variables=self.variables.union(database.propositions[label].block.v)
        # predefine self.variables because why not
        self.variables = set(['.0.', "d'", 'd"', 'D"', 'c0_', 'd0', 'd1', 'q1', 'q0', 'x1', 'G2', 'G1', 'G0', "G'", '-t', '-w', 'q"', 'G"', 'th', 'ta', 'g1', 'g0', 'H', '+t', 'p0', '.0b', "g'", 'P', 'F3', 'g"', 'X', "W'", 'W"', 'h', "ch'", 'J0', 'J1', 't0', 't1', 'p', 'W2', 'W1', '.(x)', "t'", "J'", 't"', 'J"', "w'", "M'", 'j', 'w"', 'M"', 'ze', 'j0', 'j1', '.id', 'M1', 'M0', 'w1', 'M2', "j'", 'j"', 'mu', 'et0', 'et1', 'H1_', "Z'", 'Z"', 'et"', "et'", 'Z0', 'Z1', 'C', 'P1', 'K', "z'", 'z"', 'P"', 'S', "v'_", "P'", 'z0', 'z1', 'c', 'L1_', '0w', "s'_", 'k', './\\', 'r"', 's', 'ch', 'O1_', 'S2', 'S1', 'S0', 'S"', 'v"_', 'b0_', 'ps', "m'", 'E1', 'm"', 'C"', 'o0_', "C'", 'G1_', 'm1', 'm0', 'C2', 'ph', 'C0', 'S1_', "q'", 'F', 'c"', "c'", 'N', 'V', 'f0_', 'F0', 'F1', 'F2', 'p1', '0t', 'f', 'D1_', 'n', 'p"', 'ch1', 'F"', "p'", 'v', 'f1', 'ph"', "ph'", 'o"_', 'f"', "f'", 'ph1', 'ph0', "ze'", 'V"', '.dom', "O'", "V'", 'I1', 'I0', 's1', 's0', 's"_', 'th1', 'th0', 'V0', 'V1', 'V2', 'V3', "th'", 's"', 'I"', "s'", 'th"', "I'", 'A', "L'", 'v"', 'L"', 'ta1', "v'", 'I', 'i0', 'Q', 'v1', 'v2', "ta'", 'L2', 'L3', 'L0', 'L1', 'Y', 'i"', '/t', "i'", 'a', 'si1', 'si0', 'Y1', 'i', 'Y"', 'D2', "Y'", 'q', 'si"', 'si', 'ze1', "si'", 'y', 'Y0', 'I2', 'ps"', 'y"', "ps'", "y'", 'y1', 'y0', 'ps0', 'ps1', 'O2', 'O1', 'O0', "o'_", 'la', '.Morphism', 'n0_', 'k0', 'O"', 'ze"', 'R0', 'D', 'L', 'ze0', "R'", 'c1_', 'T', 'R"', '.X.', '.1.', 'a0_', "l'", "B'", 'l"', 'd', 'B"', 'l', 'B0', 't', 'l0', 'l1', "b'", 'b"', '.(+)', 'U1', 'U0', 'h"', 'b0', 'b1', 'ta0', "U'", 'U"', "o'", "S'", "E'", 'o"', 'E"', 'i1', '+w', 'F1_', '.xb', 'Ro1', 'Ro2', 'rh', 'E0', 'o1', 'o0', "F'", 'B2', 'G', 'R1_', "e'", 'W0', 'I1_', 'O', 'e"', '._|_', 'W', 'x', 'e1', 'e0', '1t', '<_b', 'v0', '<_a', 'r0', 'r1', 'g', 'H2', 'H0', 'H1', 'o', "r'", 'w', 'H"', '.x.', "H'", 'K"', "K'", 'ta"', 'h0', 'h1', "X'", 'K1', 'K0', "T'", 'ch"', "h'", 'M1_', '.*', '.+', '.,', '.-', './', 'X"', 'u1', 'u0', '.<', "u'", 'X1', 'u"', 'ch0_', 'B', 'x"', 'N"', 'J', "x'", "N'", 'R', '.^', 'N0', 'N1', 'x0', 'Z', '.cod', 'C1', 'b', 'o1_', 'X0', '.graph', 'r', '.~', 'B1_', 'z', '.t', '.<_', '.w', 'Q1', 'Q0', 'V1_', 'rh1', 'rh0', 'w0', '~t', '~w', 'Q"', "Q'", 'et', 'rh"', '.||', "rh'", 'k"', 'A"', "k'", "A'", 'P0', 'a1_', '.\\/', 'B1', 'A1', 'A0', 'k1', 'A2', 'C1_', '.+b', 'a"', 'E', "a'", 'A1_', 'M', 'T0', 'T1', 'a1', 'a0', 'U', 'b1_', 'T"', 'ka', 'e', "D'", 'n"', 'm', "n'", 'u', 'D0', 'n1', '.Object', '.+^', 'D1'])

    def parse_new_statement(self,statement):
        string, tree = self.parse_statement(statement)
        self.search.clear()
        self.failed_parses = set()
        return string,tree

    # This checks whether we're just parsing a variable and otherwise check whether any of the propositions
    # describes it.
    def parse_statement(self,statement):
        # this should return a parse tree for a list of strings that form statement

        # check whether we are doomed to failure
        if (statement[0],len(statement)) in self.failed_parses:
            return False,False

        # attempt to search for the phrase
        tuples = self.search.find_longest(statement)
        if tuples: return tuples

        # check if it's just a variable
        if statement[1] in self.variables and not (statement[0]=='wff' and statement[1] not in self.wff_variables):
            #yep.  It's just a variable.  Skip the type checking.  Variables can only have one type ever, right?
            length = 1; # exclude the wff term
            string = statement[:2]
            tree = Tree(value='VAR'+statement[1])
            return string,tree

        #found_valid_parsing = False
        for prop_label in self.propositions:
            prop = self.propositions[prop_label]
            string, tree = self.proposition_describes_statement(prop,statement)
            if string == False:
                continue
            #print 'found',string
            #if statement[0]=='wff'

            self.search.add(string,(string,tree)) # add to the search tree
            return string, tree

        #print 'could not find expression for ',statement
        self.failed_parses.add((statement[0],len(statement)))
        return False, False


    # this is a brutally inefficient way to go about this
    # when it completes it returns a parse tree for the statement and the length of the tree
    def proposition_describes_statement(self,proposition,s):
        prop_s = proposition.statement

        # the types of all the free variables
        variable_types = {hyp.statement[1]:hyp.statement[0] for hyp in proposition.block.hypotheses}

        # string definitions of all the free variables
        variable_definitions = {hyp.statement[1]:None for hyp in proposition.block.hypotheses}

        #tree defintions of all the free variables
        variable_trees = {hyp.statement[1]:None for hyp in proposition.block.hypotheses}

        index_into_s = 0
        index_into_prop_s=0

        while index_into_prop_s < len(prop_s):
            if index_into_s>=len(s):
                #print 'ran out of s'
                return False,False
            prop_value = prop_s[index_into_prop_s]
            if prop_value in variable_types:
                # it's a variable
                # check if already defined
                if variable_definitions[prop_value]==None:
                    #then we need to figure out the parsing of the substatement
                    #print 'testing ',[variable_types[prop_value]]+s[index_into_s:],' because of ',proposition.label,prop_s
                    string, tree = self.parse_statement([variable_types[prop_value]]+s[index_into_s:])
                    if string == False:
                        return False,False
                    length = len(string)-1 # skip the wff/set/class bit
                    index_into_s+=length
                    index_into_prop_s+=1
                    variable_definitions[prop_value] = string[1:]
                    variable_trees[prop_value] = tree
                    continue
                else:
                    #we've already seen this expression before
                    if is_initial_string(variable_definitions[prop_value],statement):
                        # Yes, yes, we get the point
                        index_into_s+=len(variable_definitions[prop_value])
                        index_into_prop_s+=1
                        continue
                    else:
                        return False,False # eh.  Whatever.
            else:
                #it's not a variable
                if prop_value == s[index_into_s]:
                    index_into_s+=1
                    index_into_prop_s+=1
                    continue
                else:
                    #it's not a variable and it doesn't match
                    return False,False

        #we have the entire parsing and it appears to work
        leaves = [variable_trees[hyp.statement[1]] for hyp in proposition.block.hypotheses]
        tree = Tree(value=proposition.label,leaves = leaves)

        #now construct the string.
        out_string = s[:index_into_s]

        return out_string, tree

def is_initial_string(initial_string,string):
    #return (len(string)>=len(initial_string)) and string[:len(initial_string)]==initial_string
    if len(initial_string)>len(string): return False
    for i in range(len(initial_string)):
        if initial_string[i]!=string[i]: return False
    return True
