'''
This is a bunch of code for determining whether the current assertion has a proposition
that can be applied to it that returns only hypotheses.

Code that detects propositions that return hypotheses and tautologies
would be better, but I feel as though that goes against the nature of
the project.
'''

'''
tuples
I would have expected this to be a built in function in python,
but I couldn't figure out what it was called
input: a list of lists [l0,...,ln]
output: a list of all lists of the form (i0,...,in),
    such that ik is in lk
'''
def tuples(list_of_lists, current_index=0):
    # check terminal case
    if current_index == len(list_of_lists)-1:
        return [[value] for value in list_of_lists[current_index]]
    
    tups = tuples(list_of_lists, current_index=current_index+1)
    out = [[value] + tup for tup in tups for value in list_of_lists[current_index]]
    return out
            

def is_easy(tree, context, lm):
    labels = lm.searcher.search(tree, context, max_proposition=context.number, vclass='|-')
    for label in labels:
        prop = lm.database.propositions[label]
        
        if prop.unconstrained_arity() == 0:
            fit = is_label_easy_trivial(tree, context, lm, prop)
        else:
            fit = is_label_easy(tree, context, lm, prop)
        if fit is not None:
            return label, fit

                
def is_label_easy_trivial(tree, context, lm, prop):
    fit = lm.prop_applies_to_statement(tree, prop, context, vclass='|-')
    assert fit is not None
    prop_hyps = [h.tree.copy().replace(fit) for h in prop.hyps if h.type == 'e']
    context_hyps = [h.tree for h in context.hyps if h.type == 'e']
    
    if all(h in context_hyps for h in prop_hyps):
        return prop_hyps
    else:
        return None

''' this is a horribly inefficient mess, but I'm going to avoid premature
optimization here.  Id est, I'm lazy and don't want to deal with it right now.'''                
def is_label_easy(tree, context, lm, prop):
    full_fit = lm.prop_applies_to_statement(tree, prop, context, vclass='|-')
    assert full_fit is not None
    prop_hyps = [h.tree for h in prop.hyps if h.type == 'e']
    context_hyps = [h.tree for h in context.hyps if h.type == 'e']
    prop_variables = [f for f in prop.f]
    
    # check each hyp individually
    all_fits = []
    for ph in prop_hyps:
        this_ch_fits = []
        for ch in context_hyps:
            fit = ch.fit(ph,prop_variables)
            if fit is None: continue
            # verify compatibility with full_fit
            if not verify_compatibility(full_fit, fit):
                continue
            this_ch_fits.append(fit)
        all_fits.append(this_ch_fits)
    
    #print 'all fits', prop.label, all_fits
    
    # this is a list of lists of all fits, which need to be merged together    
    fit_tuples = tuples(all_fits)
    #print 'tuples', fit_tuples
    for fit_tuple in fit_tuples:
        merged_fit = are_valid_fits(full_fit, fit_tuple, prop, context)
        if merged_fit is not None:
            prop_hyps = [h.tree.copy().replace(merged_fit) for h in prop.hyps if h.type == 'e']
            return prop_hyps
    return None
    
def verify_compatibility(a,b):
    for key, value in b.items():
        if key in a:
            if a[key]!=b[key]:
                return False
    return True
    
    
def are_valid_fits(initial_fit, other_fits, prop, context):
    fit = initial_fit.copy()
    #print 'are_valid_fits', initial_fit, other_fits
    # start by merging the dictionaries
    for other in other_fits:
        for key, value in other.items():
            if key in fit:
                if value != fit[key]:
                    return None
            else:
                fit[key]=value
                
    # now check for consistency with the d-conditions
    
    # we only need the mandatory variables
    context_variables = [x.label for x in context.hyps if x.type=='f']
    prop_variables = [x.label for x in prop.hyps if x.type=='f']
    vars_that_appear = {v:fit[v].set().intersection(context_variables) for v in fit}

    # print 'context_variables', context_variables
    # print 'prop_variables', prop_variables
    # print 'vars_that_appear', vars_that_appear
    #print fit, 'checking d_labels now'
    for (xvar,yvar) in prop.d_labels:
        if xvar not in fit or yvar not in fit: continue
        for i in vars_that_appear[xvar]:
            for j in vars_that_appear[yvar]:
                if (i,j) not in context.d_labels:
                    return None
    return fit