
## Domain Adaptation for Arabic Cross-Domain and Cross-Dialect Sentiment Analysis from Contextualized Word Embedding


#### Code for NAACL 2021 paper [Domain Adaptation for Arabic Cross-Domain and Cross-Dialect Sentiment Analysis from Contextualized Word Embedding](https://www.aclweb.org/anthology/2021.naacl-main.226/)

## Requirements
Please make sure you have `pytorch >=1.8` , `transformers >= 3.0.0` , `farasa` and `pyarabic` installed.

## Datasets
The `data/` folder includes a dataset example used in the paper (TEAD_MSA to BRAD_LEV). 

To reproduce the results achieved in paper please use the following datasets:
### Scenario 1: Domain adaptation for dialects of the same region.
* `ArSentD-LEV`: Arabic Sentiment Twitter Dataset for LEVantine dialect [(Ramy, 2018)](http://oma-project.com/ArSenL/ArSenTD_Lev_Intro)
### Scenario 2: Domain adaptation across regional dialects.
* `HARD`:  Hotel Arabic-Reviews Dataset [(Elnagar, 2018)](https://link.springer.com/chapter/10.1007%2F978-3-319-67056-0_3)
* `BRAD`: Book reviews in Arabic dataset [(Elnagar, 2016)](https://ieeexplore.ieee.org/document/7945800/)
* `TEAD`: Large Scale Arabic Dataset for Sentiment Analysis [(Abdellaoui, 2018)](http://www.cys.cic.ipn.mx/ojs/index.php/CyS/article/view/3031)
### Scenario 3: Domain adaptation from MSA to Arabic dialects using social media data. 
* `ArSAS`:  An Arabic Speech-Act and Sentiment Corpus of Tweets [(Elmadany, 2018)](http://lrec-conf.org/workshops/lrec2018/W30/pdf/22_W30.pdf)
* `MSAC`:  Arabic Sentiment Analysis corpus [(Link)](https://github.com/ososs/Arabic-Sentiment-Analysis-corpus)
* `TSAC`:  Tunisian Sentiment Analysis Corpus [(Medhaffar, 2017)](https://www.aclweb.org/anthology/W17-1307/)
* `ASTD`:  Arabic Sentiment Tweets Dataset [(Nabil, 2015)](https://www.aclweb.org/anthology/D15-1299/)
* `AJGT`:  Arabic Jordanian General Tweets [(Link)](https://github.com/komari6/Arabic-twitter-corpus-AJGT)
* `TweetSYR` [(Saif, 2018)](http://saifmohammad.com/WebPages/ResearchInterests.html#ArabicSentiment)
* `AraSenti-Tweet`: A Corpus for Arabic Sentiment Analysis of Saudi Tweets [(Al-Twairesh, 2017)](https://www.sciencedirect.com/science/article/pii/S1877050917321518)

## Setting Up the Data
### Binary sentiment analysis
Format your data for the binary sentiment analysis to have two classes: 
```
Positive: for the positive rows.
Negative: for the negative rows.
```

Typically each data folder has 2 files: `{task}_train.csv` and `{task}_test.csv`.


## Training
To evaluate or predict labels using a finetuned model: 
```
python train.py ALDA 
				--lr 5e-6 
				--source TEAD_MSA 
				--target BRAD_LEV
```

Where you can specify your source and target datasets. You can replace `ALDA` with `MMD`, `CORAL` or `DANN` if you want to train your model using the other domain adaptation methods.


## Citation 
If you use this code, please cite this paper
```
@inproceedings{el-mekki-etal-2021-domain,
    title = "Domain Adaptation for {A}rabic Cross-Domain and Cross-Dialect Sentiment Analysis from Contextualized Word Embedding",
    author = "El Mekki, Abdellah  and
      El Mahdaouy, Abdelkader  and
      Berrada, Ismail  and
      Khoumsi, Ahmed",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.226",
    pages = "2824--2837",
    abstract = "Finetuning deep pre-trained language models has shown state-of-the-art performances on a wide range of Natural Language Processing (NLP) applications. Nevertheless, their generalization performance drops under domain shift. In the case of Arabic language, diglossia makes building and annotating corpora for each dialect and/or domain a more challenging task. Unsupervised Domain Adaptation tackles this issue by transferring the learned knowledge from labeled source domain data to unlabeled target domain data. In this paper, we propose a new unsupervised domain adaptation method for Arabic cross-domain and cross-dialect sentiment analysis from Contextualized Word Embedding. Several experiments are performed adopting the coarse-grained and the fine-grained taxonomies of Arabic dialects. The obtained results show that our method yields very promising results and outperforms several domain adaptation methods for most of the evaluated datasets. On average, our method increases the performance by an improvement rate of 20.8{\%} over the zero-shot transfer learning from BERT.",
}
```

## Acknowledgment

The structure of this code is largely based on [ALDA](https://github.com/ZJULearning/ALDA/) and [CDAN](https://github.com/thuml/CDAN). We are very grateful for their open source.