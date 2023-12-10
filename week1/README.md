# Week 1

```bash
export PYTHONPATH='.'
```

## vocab

text 뭉치를 token으로 분리하고, frequency로 자릅니다

```bash
python3 week1/vocab.py
```

## process

model에 들어갈 data 함수 (collate_fn)가 구현돼있습니다. 
text 뭉치가 들어오면 cbow window size로 잘라서 정답 레이블과 함께 반환합니다.

## train

model을 train합니다. 
model은 embedding layer, linear layer, softmax layer 총 3개로 구성됩니다.

## embedding

model의 embedding layer의 weight는 단어의 embedding vector입니다.
단어를 입력하면 cos similarity로 vector search를 할 수 있습니다.

```bash
python3 week1/embedding.py
```