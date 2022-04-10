#!/usr/bin/env python
# Chengxi Li <cli289@uky.edu>
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from collections import defaultdict
import numpy as np
import math

def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    '''
    return precook(test, n, True)

class StyleScorer(object):
    """CIDEr scorer.
    """

    def copy(self):
        ''' copy the refs.'''
        new = StyleScorer(n=self.n,allngram_sdic=self.allngram_sdic)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, test=None, refs=None, n=4, sigma=6.0,allngram_sdic=None):
        ''' singular instance '''
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len = None
        self.allngram_sdic = allngram_sdic

    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''
        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test)) ## N.B.: -1
            else:
                self.ctest.append(None) # lens of crefs and ctest have to match

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            ## avoid creating new StyleScorer instances
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self
    def compute_doc_freq(self):
        '''
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        '''
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def compute_style(self):
        def ngram2vec(cnts,ngram_weight):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram,term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                #df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                #vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                vec[n][ngram] = ngram_weight[len(ngram)][ngram]
                if math.isnan(vec[n][ngram]):
                    vec[n][ngram] = 0
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            '''
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            '''
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram,count) in vec_hyp[n].items():
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val[n]))
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
            return val
        # compute log reference length
        self.ref_len = np.log(float(len(self.crefs)))

        scores_style = defaultdict(list)
        onlystyle = defaultdict(list)
        onlystyle_exist = defaultdict(list)
        vec_display = defaultdict(list)
        styleterm = defaultdict(list)
        #import pdb; pdb.set_trace()
        for k,v in self.allngram_sdic.items():
            count =0
            for test, refs in zip(self.ctest, self.crefs):
                #if count ==243 and k=='Cheerful':
                #    import pdb; pdb.set_trace()
                count += 1
                # compute vector for test captions
                vec, norm, length = ngram2vec(test,v)
                curr_gramdic = {}
                for i in range(len(vec)):
                    gramkey = str(i+1)+"gram"
                    if gramkey not in curr_gramdic:
                        curr_gramdic[gramkey] = (vec[i],len(list(filter(lambda x:x[1]!=0 ,vec[i].items()))))
                vec_display[k].append(curr_gramdic)
                eachgen_score = []
                eachgen_score_exist = []
                eachgen_styleterm = {}
                for ni in range(self.n-1,-1,-1):
                    if len(vec[ni].values()):
                        for sgram,sv in vec[ni].items():
                            eachgen_styleterm[sgram] = []
                        break
                for hgram in eachgen_styleterm:
                    for n in range(1,len(hgram)+1):
                        for i in range(len(hgram)-n+1):
                            ngram = tuple(hgram[i:i+n])
                            eachgen_styleterm[hgram].append(vec[n-1][ngram])

                eachterm_style = []
                #for hgram,value in eachgen_styleterm.items():
                #    eachterm_style.append(np.prod(np.array(value))**(1/len(value)))

                for ni in range(self.n):
                    if len(vec[ni].values()): # you have 2gram when ni=2
                        eachgen_score.append(sum(vec[ni].values())/len(vec[ni].values()))
                        
                        if sum(vec[ni].values())!=0:
                            eachgen_score_exist.append(sum(vec[ni].values())/len(list(filter(lambda x:x[1]!=0 ,vec[ni].items()))))

                        else:
                            continue
                    else:
                        continue
                if len(eachgen_score):
                    onlystyle[k].append(sum(eachgen_score)/len(eachgen_score))
                else:
                    onlystyle[k].append(0)
                if len(eachgen_score_exist):
                    onlystyle_exist[k].append(sum(eachgen_score_exist)/len(eachgen_score_exist))
                else:
                    onlystyle_exist[k].append(0)
                
                if eachterm_style:
                    styleterm[k].append(sum(eachterm_style)/len(eachterm_style))
                else:
                    styleterm[k].append(0)
                # compute vector for ref captions
                score = np.array([0.0 for _ in range(self.n)])
                for ref in refs:
                    vec_ref, norm_ref, length_ref = ngram2vec(ref,v)
                    score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
                # change by vrama91 - mean of ngram scores, instead of sum
                score_avg = np.mean(score)
                # divide by number of references
                score_avg /= len(refs)
                # multiply score by 10
                score_avg *= 10.0
                # append score of an image to the score list
                scores_style[k].append(score_avg)
        return scores_style,onlystyle,onlystyle_exist,vec_display,styleterm

    def compute_score(self, option=None, verbose=0):
        # compute idf
        #self.compute_doc_freq()
        # assert to check document frequency
        #assert(len(self.ctest) >= max(self.document_frequency.values()))
        # compute cider score
        score,onlystyle,onlystyle_exist,vec_display,styleterm = self.compute_style()
        meanstyle_score = defaultdict(float)
        mean_onlystyle_score = defaultdict(float)
        mean_onlystyle_exist_score = defaultdict(float)
        mean_styleterm = defaultdict(float)
        for k,v in score.items():
            meanstyle_score[k] = np.mean(np.array(v))
        for ko, vo in onlystyle.items():
            mean_onlystyle_score[ko] = np.mean(np.array(vo))

        for koe, voe in onlystyle_exist.items():
            mean_onlystyle_exist_score[koe] = np.mean(np.array(voe))
        for kst, vst in styleterm.items():
            mean_styleterm[kst] = np.mean(np.array(vst))

        # debug
        # print score
        return meanstyle_score, score,mean_onlystyle_score,onlystyle,mean_onlystyle_exist_score,onlystyle_exist,vec_display,mean_styleterm,styleterm
