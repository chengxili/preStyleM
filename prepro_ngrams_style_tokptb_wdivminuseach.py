"""
get scores and save them
"""
import sys
import os
import json
import argparse
from six.moves import cPickle
import misc.utils as utils
from collections import defaultdict
import string
from statsmodels.distributions.empirical_distribution import ECDF
import math
import numpy as np
from tokenizer.ptbtokenizer import PTBTokenizer
from scipy import spatial
punc=string.punctuation.replace("'", "")
tokenizer = PTBTokenizer()
use_unk = 0
import re
RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)
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

def create_crefs(refs):
  crefs = []
  for ref in refs:
    # ref is a list of 5 captions
    crefs.append(cook_refs(ref))
  return crefs

def compute_doc_freq(crefs):
  '''
  Compute term frequency for reference data.
  This will be used to compute idf (inverse document frequency later)
  The term frequency is stored in the object
  :return: None
  '''
  document_frequency = defaultdict(float)
  for refs in crefs:
    # refs, k ref captions of one image
    for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
      document_frequency[ngram] += 1
      # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
  return document_frequency

def compute_doc_freq_style(crefs):
    '''
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    This seems count each ngram in each sentence as 1 for document_frequency 
    '''
    document_frequency = {}
    for refs in crefs:
# refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
            if len(ngram) not in document_frequency:
                document_frequency[len(ngram)] = defaultdict(float)
                document_frequency[len(ngram)][ngram] = 1
            else:
                document_frequency[len(ngram)][ngram] += 1
          # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    return document_frequency

def build_dict(imgs, wtoi, params):
  wtoi['<eos>'] = 0
  all_imgid_wcounts = {}
  count_imgs = 0
  numsents = 0
  rm_punc=params['rm_punc']
  refs_words = []
  refs_idxs = []
  all_ngram_count_single = {} # for later map sentence and ngrams
  newimgs ={}
  for cnt, img in enumerate(imgs):
      newimgs[img['image_hash']+"_"+img['personality']]=[]
      for i in range(len(img['sentences'])):
          sent = img['sentences'][i]
          txt= RETOK.findall(sent)
          caption = " ".join(txt)
          img['caption'] = caption
          img['id'] = cnt
          newimgs[img['image_hash']+"_"+img['personality']].append(img)
  
  ptbimgs = tokenizer.tokenize(newimgs)
  for img in imgs:
    imgid = img['image_hash']
    img['final_captions'] = []
    if imgid not in all_imgid_wcounts:
        all_imgid_wcounts[imgid] = {}
    if (params['split'] == img['split']) or \
      (params['split'] == 'train' and img['split'] == 'restval') or \
      (params['split'] == 'all'):
      #(params['split'] == 'val' and img['split'] == 'restval') or \
      ref_words = []
      ref_idxs = []
      for sent in ptbimgs[imgid+"_"+img['personality']]:
        if isinstance(sent,dict) and 'tokens' in sent:
            txt=sent['tokens']
        elif isinstance(sent,str):
            if use_unk:
                sent=sent.lower()
            if rm_punc>0:
                sent=sent.translate(str.maketrans('', '', punc))
            txt = sent.split()
            if("[" in txt) and len(txt)==3:
                print(txt)
                continue
            #print(txt)
        if hasattr(params, 'bpe'):
            txt = params.bpe.segment(' '.join(txt)).strip().split(' ')
        tmp_tokens = txt + ['<eos>']
        if use_unk:
            tmp_tokens = [_ if _ in wtoi else 'UNK' for _ in tmp_tokens]
        img['final_captions'].append(tmp_tokens)
        ref_words.append(' '.join(tmp_tokens))
        if use_unk:
            ref_idxs.append(' '.join([str(wtoi[_]) for _ in tmp_tokens]))
        # get ngram per img
        ngram_count_single = cook_refs([ref_words[-1]])
        all_ngram_count_single[ref_words[-1]] = ngram_count_single 
        for eachitem in ngram_count_single:
            for ngram, nc in eachitem.items():
                all_imgid_wcounts[imgid][ngram] = all_imgid_wcounts[imgid].get(ngram, 0) + nc
       
        numsents = numsents +1
      refs_words.append(ref_words)
      refs_idxs.append(ref_idxs)
      count_imgs += 1
  print('total imgs:', count_imgs)

  ngram_words = compute_doc_freq(create_crefs(refs_words))
  if use_unk:
      ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
  else:
      ngram_idxs = None
  return ngram_words, ngram_idxs, count_imgs,numsents,all_imgid_wcounts,all_ngram_count_single,ptbimgs


def build_dict_style(imgs, wtoi, params,ptbimgs):
  wtoi['<eos>'] = 0

  count_imgs = 0
  numsents = 0
  rm_punc=params['rm_punc']
  refs_words = []
  refs_idxs = []
  for img in imgs:
    if (params['split'] == img['split']) or \
      (params['split'] == 'train' and img['split'] == 'restval') or \
      (params['split'] == 'all'):
      ref_words = []
      ref_idxs = []
      for sent in ptbimgs[img['image_hash']+"_"+img['personality']]:
        #print(sent)
        if isinstance(sent,dict) and 'tokens' in sent:
            txt=sent['tokens']
        elif isinstance(sent,str):
            if use_unk:
                sent=sent.lower()
            if rm_punc>0:
                sent=sent.translate(str.maketrans('', '', punc))
            #import re
            #RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)
            #txt= RETOK.findall(sent)
            txt = sent.split()
            if("[" in txt) and len(txt)==3:
                print(txt)
                continue
            #print(txt)
        if hasattr(params, 'bpe'):
            txt = params.bpe.segment(' '.join(txt)).strip().split(' ')
        tmp_tokens = txt + ['<eos>']
        if use_unk:
            tmp_tokens = [_ if _ in wtoi else 'UNK' for _ in tmp_tokens]
        ref_words.append(' '.join(tmp_tokens))
        if use_unk:
            ref_idxs.append(' '.join([str(wtoi[_]) for _ in tmp_tokens]))
        numsents = numsents +1
      refs_words.append(ref_words)
      refs_idxs.append(ref_idxs)
      count_imgs += 1
  print('total imgs:', count_imgs)

  ngram_words = compute_doc_freq_style(create_crefs(refs_words))
  if use_unk:
      ngram_idxs = compute_doc_freq_style(create_crefs(refs_idxs))
  else:
      ngram_idxs = None
  return ngram_words, ngram_idxs, count_imgs,numsents






def contrast_p(imgs, doc_freq,N,allstyle_docfreq,allstyle_numsents,all_imgid_wcounts,ecdf_dic,all_ngram_count_single, cnt_dic):
    styles = list(allstyle_docfreq.keys())
    sent_sdic = defaultdict(dict)
    sent_odic = defaultdict(dict)
    for img in imgs:
        imgid = img['image_hash']
        for sent in img['final_captions']:
          sentstr = ' '.join(sent)
          ngram_count_sent = all_ngram_count_single[sentstr]
          #sent_vdic = {}
          print(sentstr,img['personality'])
          s_w_bucket = {}
          s_w_k_bucket = {}
          for group in ngram_count_sent:
              for tw, cnt in group.items(): # loop ngram
                  n = len(tw)
                  if n not in sent_sdic:
                      sent_sdic[n] = defaultdict(float)
                  if tw in sent_sdic[n]:
                      continue
                  if n not in sent_odic:
                      sent_odic[n] = {}
                  w = ' '.join(tw)
                  s_w=[]
                  s_w_bucket[tw]=[]
                  for k in allstyle_docfreq: # get all styles
                      ecdf = ecdf_dic[k][n]
                      if k!= img['personality']: 
                          if tw in allstyle_docfreq[k][n]:
                              s_w.append(ecdf(allstyle_docfreq[k][n][tw]))
                              s_w_bucket[tw].append(ecdf(allstyle_docfreq[k][n][tw]))
                          else: # when ngram not in current style
                              s_w.append(ecdf(0))
                              #s_w *=1-ecdf(0)
                              s_w_bucket[tw].append(ecdf(0))
                      else:
                          s_w_k = ecdf(allstyle_docfreq[k][n][tw])
                          s_w_k_bucket[tw]=s_w_k

                  if s_w:
                      # correct their probability by normlizing them 
                      occur_num = len(np.array(s_w)[np.array(s_w)!=0])+1
                      oldsw = s_w 
                      s_w = np.array(s_w)/occur_num
                      s_w_bucket[tw] = list(s_w) 
                      oldswk = s_w_k
                      s_w_k = s_w_k/occur_num
                      s_w_k_bucket[tw]=s_w_k
                      if tw in sent_sdic[n]:
                          assert(sent_sdic[n][tw]==np.mean(s_w_k-np.array(s_w)))
                      #sent_sdic[n][tw] =(s_w_k *contrast)**(1/2)
                      afterminus = np.mean(s_w_k-np.array(s_w))
                      sent_sdic[n][tw] = afterminus
                      print(tw,"previous:",np.mean(oldswk-np.array(oldsw)),"sent_sdic[n][tw]:",sent_sdic[n][tw], img['personality'])
                      # print(tw,len(np.array(s_w)[np.array(s_w)!=0])/len(s_w),badscore,badscorep,s_w_k,sent_sdic[n][tw],img['personality'])
                  else:
                      s_w = s_w_k
        #import pdb; pdb.set_trace()
#          for n in sent_odic:
#              sent_odic[n] = sorted([(weight,w) for w,weight in sent_odic[n].items()], reverse=True)
#              sent_sdic[n] = sorted([(weight,w) for w,weight in sent_sdic[n].items()], reverse=True)
#          print("o:")
#          print(sent_odic)
#          #print("v:")
#          #print(sent_vdic)
#          print("s:")
#          print(sent_sdic) # something consider to save later
    return sent_sdic
def load_personalities(personality_path):
  perss = []
  with open(personality_path) as f:
      for line in f:
          if 'Trait' not in line:
              perss.append(line[0:-1])
  return perss


def main(params):
    imgs = json.load(open(params['input_json'], 'r'))
    dict_json = json.load(open(params['dict_json'], 'r'))
    itow = dict_json['ix_to_word']
    wtoi = {w:i for i,w in itow.items()}
    personality_path = "personalities.txt"
    perss=load_personalities(personality_path)
# Load bpe
    if 'bpe' in dict_json:
        import tempfile
        import codecs
        codes_f = tempfile.NamedTemporaryFile(delete=False)
        codes_f.close()
        with open(codes_f.name, 'w') as f:
          f.write(dict_json['bpe'])
        with codecs.open(codes_f.name, encoding='UTF-8') as codes:
          bpe = apply_bpe.BPE(codes)
        params.bpe = bpe

    preload = 0
    if not preload:
        imgs = imgs['images']
        imgs=[img for img in imgs if not img['image_hash'].startswith('ac8')]
        imgs = [img for img in imgs if img['personality'] in perss]
        imgs = [img for img in imgs if '[DISCONNECT]' not in img['sentences'][0]  and '[TIMEOUT]' not in img['sentences'][0] and '[RETURNED]' not in img['sentences'][0]]
        styleimage = {}
        for img in imgs:
            personality = img['personality']
            if personality not in styleimage:
                styleimage[personality]=[img]
            else:
                styleimage[personality].append(img)

        ngram_words, ngram_idxs, ref_len,numofsents,all_imgid_wcounts,all_ngram_count_single,ptbimgs = build_dict(imgs, wtoi, params)
        # create the vocab 
        allstyle_docfreq = {}
        allstyle_numsents = {}
        styles = list(styleimage.keys())

        for k, vimgs in styleimage.items():
            ngramstyledoc_freq,ngramstyledoc_freq_idx, imagect,numsents = build_dict_style(vimgs, wtoi, params,ptbimgs)
            allstyle_docfreq[k] = ngramstyledoc_freq
            allstyle_numsents[k] = numsents

        ecdf_dic = {}
        cnt_dic = {}
        for k in allstyle_docfreq:
            ecdf_dic[k] = {}
            #cnt_dic[k] = {}
            for n in allstyle_docfreq[k]:
                ecdf_dic[k][n] = ECDF(list(allstyle_docfreq[k][n].values()))
        utils.pickle_dump({'document_frequency': ngram_words, 'ref_len': ref_len,'numofsents':numofsents,'all_imgid_wcounts':all_imgid_wcounts,'all_ngram_count_single':all_ngram_count_single,'imgs':imgs}, open(params['output_pkl']+'-words.p','wb'))
        
        utils.pickle_dump({'allstyle_docfreq': allstyle_docfreq, 'allstyle_numsents': allstyle_numsents,"ecdf_dic":ecdf_dic,'styleimage':styleimage, "cnt_dic":cnt_dic}, open(params['output_pkl']+'style-words.p','wb'))
    else:
        print("begin to load")
        pkl_file = utils.pickle_load(open(params['output_pkl']+'-words.p','rb')) 
        ngram_words = pkl_file['document_frequency']
        ref_len = np.log(float(pkl_file['ref_len']))
        numofsents = pkl_file['numofsents']
        all_imgid_wcounts = pkl_file['all_imgid_wcounts']
        all_ngram_count_single = pkl_file['all_ngram_count_single']
        imgs = pkl_file['imgs']
        style_pkl_file = utils.pickle_load(open(params['output_pkl']+'style-words.p','rb'))
        allstyle_docfreq = style_pkl_file['allstyle_docfreq']
        allstyle_numsents = style_pkl_file['allstyle_numsents']
        ecdf_dic = style_pkl_file['ecdf_dic']
        styleimage = style_pkl_file['styleimage']
        #cnt_dic = style_pkl_file['cnt_dic']
        print("finish load")
    all_sent_sdic=defaultdict(dict)
    tmp_cnt = 0
    for k, vimgs in styleimage.items():
        if k!='Gloomy':
           continue
        import pdb; pdb.set_trace()
        print("=========personality ", k, " =============")
        print("======= total "+str(len(vimgs))+" images =======")
        if len(vimgs)<20:
            import pdb; pdb.set_trace()
        tmp_sent_sdic = contrast_p(vimgs, ngram_words,numofsents,allstyle_docfreq, allstyle_numsents, all_imgid_wcounts,ecdf_dic,all_ngram_count_single,None)
        all_sent_sdic[k] = tmp_sent_sdic
        tmp_cnt +=1
        print("=========cnt====:",tmp_cnt,)
    utils.pickle_dump({'all_sent_sdic': all_sent_sdic}, open(params['output_pkl']+'ngram_sprob.p','wb'))



if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json', default='FlickrStyle_v0.9/dataset_person_flik.json', help='input json file to process into hdf5')
  parser.add_argument('--dict_json', default='data/personcap_flik.json', help='output json file')
  parser.add_argument('--output_pkl', default='data/person-all', help='output pickle file')
  parser.add_argument('--split', default='all', help='test, val, train, all')
  parser.add_argument('--rm_punc', default=0, type=int, help='remove punctuation')
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  main(params)
