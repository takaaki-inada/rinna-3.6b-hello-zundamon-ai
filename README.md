# rinna-3.6b-hello-zundamon-ai

最初に、こちらの解説記事をお読みください。

[ローカルで動く大規模言語モデル(Rinna-3.6B)を使ってあなただけのAIパートナーを作ろう - Qiita](https://qiita.com/takaaki_inada/items/9a9c07e85e46ec0e872e)

## 必要モジュールのインストール

```
git clone https://github.com/takaaki-inada/rinna-3.6b-hello-zundamon-ai.git
cd rinna-3.6b-hello-zundamon-ai
python -m venv .venv
source .venv/bin/activate  # windows環境だと既にここから手順違います
pip install --upgrade pip setuptools
pip install -r requirements.txt
pip install --no-cache git+https://github.com/huggingface/transformers@de9255de27abfcae4a1f816b904915f0b1e23cd9
```

## データ数200で学習する

```
python scripts/train_short.py
```

## データ数200で学習した結果を確認する

```
python scripts/generate_sample.py
```

## データ数を増やして学習する

学習用データセットのダウンロード
```
# cd {rinna-3.6b-hello-zundamon-ai}
wget -P datasets https://huggingface.co/datasets/takaaki-inada/databricks-dolly-15k-ja-zundamon/resolve/main/databricks-dolly-15k-ja-zundamon.json
```

学習
```
python scripts/train.py
```

## Webで動かす

### テキストチャット
#### 生成途中を画面表示して動かす(streaming)

```
python scripts/webui_streaming.py
```

#### 高速に動かす(CTranslate2)

CTranslate2変換
```
python scripts/convert_ctranslate2.py
```

webuiを起動して確認
```
python scripts/webui_ct2.py
```

### アバターと音声でチャット
#### Streaming APIサーバを起動する

```
uvicorn zundamon_fastapi:app --reload --port 8000
```

#### ChatVRMを起動してずんだもんと音声チャットする

```
cd ChatVRM
npm install
npm run dev
```
