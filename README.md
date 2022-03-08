# 日本語Wikipediaを用いたBART事前学習

# 概要
 
[BART](https://aclanthology.org/2020.acl-main.703/)を日本語Wikipediaを用いて事前学習する。
 
# インストール
 
```bash

#cuda=11.3のpytorchをインストール
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

#その他のライブラリをインストール
pip install -r requirements.txt
```
[fairseq](https://github.com/pytorch/fairseq)のインストール

```bash
git clone https://github.com/pepe-shumpei/style-bart
cd fairseq
pip install --editable ./
```
 
# 使用方法

## (1) 日本語Wikipediaの前処理
日本語Wikipediaをダウンロードして解凍する。
```bash
cd data
wget https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2
bunzip2 jawiki-latest-pages-articles.xml.bz2
```

日本語Wikipediaの前処理を行う。
```bash
python -m wikiextractor.WikiExtractor jawiki-latest-pages-articles.xml
find text/ | grep wiki | awk '{system("cat "$0" >> wiki.txt")}'
sed -i '/^<[^>]*>$/d' wiki.txt
sed -i '/^$/d' wiki.txt
```

## (1') 日本語Wikipediaの前処理(小規模 ver)
日本語Wikipediaのすべてのダウンロードと前処理には時間がかかるため、先頭100万文の日本語Wikipedia(wiki.txt.tar.gz)を用意した。

wiki.txt.tar.gzを解凍する。
```bash
cd data
tar -zxvf wiki.txt.tar.gz
```

**Note:** (1)もしくは(1')の片方を実行する。

## (2) 訓練データ、開発データ、テストデータに分割
train,validation,testデータに分割する。
```bash
bash split-dataset.sh
```

## (3)トーカナイズ
トーカナイズには[sentencepiece](https://github.com/google/sentencepiece)を用いる。

まず、sentencepieceモデルを訓練する。
```bash
python train-sp.py
```
次に、訓練したsentencepieceを適用する。
```bash
python apply-sp.py
```

## (4) BARTの事前学習
fairsq-preprocessとfairseq-trainを用いてBARTの事前学習を行う。
fairseq-preprocessとfairseq-trainの詳細は[fairseq Command-line Tools](https://fairseq.readthedocs.io/en/latest/command_line_tools.html)を参照。

まず、fairseq-preprocessを用いてデータをバイナリ化する。バイナリ化したデータはdata-bin/wiki-ja/に生成される。
```bash
bash preprocess.sh
```

次に、fairseq-trainを用いてBARTで事前学習を行う。
```bash
bash train.sh
```
