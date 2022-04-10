# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric 
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .style_scorer import StyleScorer

class StyleCider:
    """
    Main Class to compute the StyleCIDEr metric 

    """
    def __init__(self, test=None, refs=None, n=4, sigma=6.0,allngram_sdic=None):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma
        self.allngram_sdic = allngram_sdic
    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus 
        """

        assert(list(gts.keys()) == list(res.keys()))
        imgIds = list(gts.keys())

        style_scorer = StyleScorer(n=self._n, sigma=self._sigma,allngram_sdic=self.allngram_sdic)

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]
            hypo = list(map(lambda x:x.replace(' unk ',' UNK '),hypo))
            ref = list(map(lambda x:x.replace(' unk ',' UNK '),ref))

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            style_scorer += (hypo[0], ref)
        (score, scores,onlystyle_score,onlystyle_scores, onlystyle_existscore,onlystyle_existscores,vec_display,meanstyleterm,styleterm) = style_scorer.compute_score()

        return score, scores,onlystyle_score,onlystyle_scores,onlystyle_existscore,onlystyle_existscores,vec_display,meanstyleterm,styleterm

    def method(self):
        return "styleCIDEr"
