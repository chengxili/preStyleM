# preStyleM
A repos for people who are interested in StyleM before I clean further

## prepro_ngrams_style_tokptb_wdivminuseach.py is the file to calculate CNG score
```
//sample command for it

python prepro_ngrams_style_tokptb_wdivminuseach.py --rm_punc 0 --input_json  data/dataset_person.json --dict_json data/personcap_added1.json  --split all --output_pkl data/person-metric-tokptb-wdivminuseach
```
## coco-captions/pycocoevalcap contains OnlyStyle and StyleCider

# Consider to cite my paper
```
@article{li2022stylem,
  title={StyleM: Stylized Metrics for Image Captioning Built with Contrastive N-grams},
  author={Li, Chengxi and Harrison, Brent},
  journal={arXiv preprint arXiv:2201.00975},
  year={2022}
}
```

