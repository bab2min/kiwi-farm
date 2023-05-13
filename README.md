# 키위 농장 Kiwi Farm
[한국어 형태소 분석기 Kiwi](https://github.com/bab2min/Kiwi)를 활용한 딥러닝 언어 모델들을 실험적으로 키우는 공간입니다.

BERT, GPT, BART와 같은 딥러닝 언어모델에서는 크기가 고정된 닫힌 어휘 집합을 사용합니다. 
따라서 딥러닝 언어모델에 텍스트를 입력하려면 임의의 텍스트를 고정된 어휘 집합으로 분할하여 변환해주는 토크나이저(Tokenizer)가 필수적입니다.
한국어의 경우 오랫동안 개발되어온 형태소 분석기가 있으나, 기존의 형태소 분석기들은 분석결과를 고정된 개수의 어휘로 출력하는 기능이 없었으므로 
형태소 분석기를 토크나이저로 사용할 수 없었습니다.
그래서 한국어의 특징을 고려하지 못함에도 Byte Pair Encoding이나 SentencePiece 등을 토크나이저로 사용하고 있는 상황입니다.

`Kiwi`는 0.15버전에서부터 형태소 분석과 Subword 분절 기능을 통합한 Unigram 토크나이저를 제공합니다. 
이 저장소에서는 Kiwi를 기반으로한 토크나이저의 성능을 실험하고, 실제로 이 토크나이저를 기반으로 학습한 딥러닝 모델의 특징을 분석해보고자 합니다.

이 저장소에서 공개된 Kiwi 기반의 딥러닝 언어 모델을 사용하려면 [`kiwipiepy>=0.15.1`](https://github.com/bab2min/kiwipiepy)과 `transformers>=4.12`가 필요합니다. 요구사항이 모두 준비된 상황이라면 아래와 같이 간단하게 [kiwi-farm](https://huggingface.co/kiwi-farm)의 모델을 가져와서 사용할 수 있습니다. 
```python
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
)
import kiwipiepy.transformers_addon
# kiwipiepy.transformers_addon를 import해야
# KiwiTokenizer가 AutoTokenizer에 등록된다.
# KiwiTokenizer는 PreTrainedTokenizer와 대부분의 기능이 호환되므로
# 기존의 transformers 코드를 그대로 사용할 수 있다.

tokenizer = AutoTokenizer.from_pretrained('kiwi-farm/roberta-base-32k')
model = AutoModelForMaskedLM.from_pretrained('kiwi-farm/roberta-base-32k')
```

## 토크나이저 

`KiwiTokenizer`를 학습하는 데에는 다음과 같은 말뭉치를 사용했습니다.

* 모두의 말뭉치: 문어 말뭉치
* 모두의 말뭉치: 구어 말뭉치
* 모두의 말뭉치: 비출판물 말뭉치
* 모두의 말뭉치: 신문 말뭉치

아래는 `KiwiTokenizer`와 다른 토크나이저를 비교한 결과입니다.

### 토크나이저별 통계

<table>
<tr><th>토크나이저</th><th>Vocab Size</th><th>Special Tokens</th><th>Hangul Tokens</th><th>Alnum Tokens</th><th>Hanja Tokens</th></tr>
<tr><th>KiwiTokenizer 16k</th><td>16000</td><td>7</td><td>12777 (11130 / 1647)</td><td>804 (295 / 509)</td><td>1598 (1598 / 0)</td></tr>
<tr><th>KiwiTokenizer 32k</th><td>32000</td><td>7</td><td>26285 (21917 / 4368)</td><td>2446 (965 / 1481)</td><td>2334 (2334 / 0)</td></tr>
<tr><th>KiwiTokenizer 48k</th><td>48000</td><td>7</td><td>39670 (32345 / 7325)</td><td>4478 (1960 / 2518)</td><td>2799 (2799 / 0)</td></tr>
<tr><th>KiwiTokenizer 64k</th><td>64000</td><td>7</td><td>52877 (42693 / 10184)</td><td>6806 (3150 / 3656)</td><td>3173 (3173 / 0)</td></tr>
<tr><th>klue/roberta-base</th><td>32000</td><td>5</td><td>28388 (24638 / 3750)</td><td>2389 (1461 / 928)</td><td>335 (335 / 0)</td></tr>
<tr><th>beomi/kcbert-base</th><td>30000</td><td>5</td><td>28137 (18874 / 9263)</td><td>630 (425 / 205)</td><td>0 (0 / 0)</td></tr>
<tr><th>HanBert-54kN-torch</th><td>54000</td><td>5</td><td>45702 (32462 / 13240)</td><td>3533 (2096 / 1437)</td><td>1821 (914 / 907)</td></tr>
</table>

* 괄호 안의 숫자는 차례로 `(Word의 개수 / Subword의 개수)` 입니다.
* 여기서 `Word`는 공백으로 시작하는 토큰을 가리키며, `Subword`는 공백 없이 이어지는 토큰을 가리킵니다. 

### 토크나이징 사례

<table>
<tr><th>토크나이저</th><th>제임스웹우주망원경이 발사됐다.</th></tr>
<tr><th>KiwiTokenizer 32k</th><td>['제임스', '##', '웹', '##우주', '##', '망원경', '이/J', '발사', '되/V', '었/E', '다/E', '.']</td></tr>
<tr><th>klue/roberta-base</th><td>['제임스', '##웹', '##우주', '##망', '##원', '##경', '##이', '발사', '##됐', '##다', '.']</td></tr>
<tr><th>beomi/kcbert-base</th><td>['제', '##임', '##스', '##웹', '##우', '##주', '##망', '##원', '##경이', '발사', '##됐다', '.']</td></tr>
<tr><th>HanBert-54kN-torch</th><td>['제임스', '##웹', '##우', '##주', '##망', '##원경', '~~이', '발사', '##됐다', '.']</td></tr>
</table>

* `KiwiTokenizer`는 Glue 토큰(`##`)을 사용합니다. 특정 문자열을 Subword로 분절하는 것보다 Glue + Word를 사용하는게 낫다고 판단되면 후자를 선택합니다. 그 결과 다른 토크나이저에서는 `망원경`이 망/원/경, 혹은 망/원경 등으로 분절되지만, `KiwiTokenizer`에서는 `망원경` 원형 전체가 보존됩니다.

<table>
<tr><th>토크나이저</th><th>힘들어도 끝까지 버텼다.</th></tr>
<tr><th>KiwiTokenizer 32k</th><td>['힘들/V', '어도/E', '끝', '까지/J', '버티/V', '었/E', '다/E', '.']</td></tr>
<tr><th>klue/roberta-base</th><td>['힘들', '##어도', '끝', '##까', '##지', '[UNK]', '.']</td></tr>
<tr><th>beomi/kcbert-base</th><td>['힘들어도', '끝까지', '버', '##텼', '##다', '.']</td></tr>
<tr><th>HanBert-54kN-torch</th><td>['힘들', '##어', '##도', '끝', '~~까지', '버텼', '##다', '.']</td></tr>
</table>

* 다른 토크나이저에서는 비교적 출현 빈도가 적은 음절인 `텼`이 어휘집합에 포함되어 있지 않아 `UNK`(알 수 없는 토큰)로 처리되곤 합니다. `KiwiTokenizer`에서는 동사/형용사에 대해서는 형태소 분석을 수행하므로 어간이 어미와 결합해 희귀한 형태가 되더라도 절대 `UNK`로 처리되지 않습니다. 
* 추가적으로 한글의 경우 초/중성 + 종성을 분리해 표현하는 대체 기능이, 그 외의 문자에 대해서는 UTF8 byte 단위로 분리해 표현하는 대체 기능이 포함되어 있어서, 어떤 문자에 대해서도 `UNK`가 나오지 않습니다.

<table>
<tr><th>토크나이저</th><th>달려가는 날쌘돌이</th></tr>
<tr><th>KiwiTokenizer 32k</th><td>['달려가/V', '는/E', '날쌔/V', 'ᆫ/E', '##돌', '이/J']</td></tr>
<tr><th>klue/roberta-base</th><td>['달려가', '##는', '[UNK]']</td></tr>
<tr><th>beomi/kcbert-base</th><td>['달려', '##가는', '날', '##쌘', '##돌이']</td></tr>
<tr><th>HanBert-54kN-torch</th><td>['달려가', '~~는', '[UNK]', '~~이']</td></tr>
</table>

* 위와 유사하게 `쌘`이라는 음절 때문에 `UNK`가 생성되는 토크나이저가 있습니다.

<table>
<tr><th>토크나이저</th><th>주거니 받거니 줬거니 받았거니</th></tr>
<tr><th>KiwiTokenizer 32k</th><td>['주/V', '거니/E', '받/V', '거니/E', '주/V', '었/E', '거니/E', '받/V', '었/E', '거니/E']</td></tr>
<tr><th>klue/roberta-base</th><td>['주거', '##니', '받', '##거', '##니', '줬', '##거', '##니', '받', '##았', '##거', '##니']</td></tr>
<tr><th>beomi/kcbert-base</th><td>['주거', '##니', '받', '##거니', '줬', '##거니', '받았', '##거니']</td></tr>
<tr><th>HanBert-54kN-torch</th><td>['주거', '##니', '받', '##거니', '줬', '##거니', '받', '##았', '##거니']</td></tr>
</table>

* Kiwi는 동/형용사의 경우 형태소 분석을 사용하기 때문에 동일한 단어가 활용형이 달라져서 다른 토큰으로 처리되는 경우가 적습니다.

<table>
<tr><th>토크나이저</th><th>띄 어 쓰 기 엉 망 진 창 으 로 하 기</th></tr>
<tr><th>KiwiTokenizer 32k</th><td>['띄/V', '어/E', '쓰/V', '기/E', '엉망', '지/V', 'ᆫ/E', '창', '으로/J', '하/V', '기/E']</td></tr>
<tr><th>klue/roberta-base</th><td>['띄', '어', '쓰', '기', '엉', '망', '진', '창', '으', '로', '하', '기']</td></tr>
<tr><th>beomi/kcbert-base</th><td>['띄', '어', '쓰', '기', '엉', '망', '진', '창', '으', '로', '하', '기']</td></tr>
<tr><th>HanBert-54kN-torch</th><td>['띄', '어', '쓰', '기', '엉', '망', '진', '창', '으', '로', '하', '기']</td></tr>
</table>

<table>
<tr><th>토크나이저</th><th>띄어쓰기엉망진창으로하기</th></tr>
<tr><th>KiwiTokenizer 32k</th><td>['띄', '##어', '##쓰기', '##', '엉망', '##진', '##창', '으로/J', '하/V', '기/E']</td></tr>
<tr><th>klue/roberta-base</th><td>['띄', '##어', '##쓰기', '##엉', '##망', '##진', '##창', '##으로', '##하기']</td></tr>
<tr><th>beomi/kcbert-base</th><td>['띄', '##어', '##쓰기', '##엉', '##망', '##진창', '##으로', '##하기']</td></tr>
<tr><th>HanBert-54kN-torch</th><td>['띄', '##어', '##쓰기', '##엉', '##망', '##진', '##창', '##으로', '##하기']</td></tr>
</table>

* Kiwi가 가지고 있는 띄어쓰기 오류 보정 모델 덕분에 띄어쓰기가 엉망인 텍스트에 대해서도 잘 작동합니다.

<table>
<tr><th>토크나이저</th><th>나랏〮말〯ᄊᆞ미〮 듀ᇰ귁〮에〮달아〮 문ᄍᆞᆼ〮와〮로〮서르ᄉᆞᄆᆞᆺ디〮아니〮ᄒᆞᆯᄊᆡ〮</th></tr>
<tr><th>KiwiTokenizer 32k</th><td>['나', '##랏', '말', 'ᄊ', '##ᆞ', '##미', 'ᄃ', '##ᅲ', '##ᇰ', '##귀', '##ᆨ', '에/J', '달/V', '어/E', '문', '##ᄍ', '##ᆞ', '##ᆼ', '<0xE3>', '<0x80>', '<0xAE>', '##와', '<0xE3>', '<0x80>', '<0xAE>', '##로', '<0xE3>', '<0x80>', '<0xAE>', '##서', '##르', '##ᄉ', '##ᆞ', '##ᄆ', '##ᆞ', '##ᆺ', '##디', '아니', 'ᄒ', '##ᆞ', '##ᆯ', '##ᄊ', '##ᆡ']</td></tr>
<tr><th>klue/roberta-base</th><td>['[UNK]', '[UNK]', '[UNK]']</td></tr>
<tr><th>beomi/kcbert-base</th><td>['[UNK]', '[UNK]', '[UNK]']</td></tr>
<tr><th>HanBert-54kN-torch</th><td>['나', '##랏', '[UNK]', '말', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '에', '[UNK]', '달아', '[UNK]', '[UNK]', '[UNK]', '와', '[UNK]', '로', '[UNK]', '[UNK]', '[UNK]', '아니', '[UNK]', '[UNK]', '[UNK]']</td></tr>
</table>

* Kiwi는 첫가끝 코드를 지원하여 옛한글에 대해서도 `UNK`를 생성하지 않습니다. 다만 일부 방점은 어휘집합에 포함되지 않아서 UTF8 byte로 분절됩니다.

## AutoEncoding 언어 모델
`KiwiTokenizer`가 딥러닝 언어 모델에서 얼마나 유용한지 확인하기 위해 실제로 RoBERTa 모델을 사전학습해 보았습니다. 사전학습은 바닥부터 진행된 것은 아니며 이미 강력한 것으로 확인된 [klue/roberta-base 모델](https://huggingface.co/klue/roberta-base)을 재활용하여 어휘 집합만 갈아끼운 뒤 추가 학습을 진행하는 방식으로 수행되었습니다. 사전 학습은 klue/roberta-base와 동일한 어휘 집합 크기를 가진 KiwiTokenizer 32k를 바탕으로 진행되었습니다. 사전 학습 절차에 대해서는 `train_bert.py` 코드를 참조해주세요. 사전학습이 완료된 모델은 [kiwi-farm/roberta-base-32k(huggingface 모델 저장소)](https://huggingface.co/kiwi-farm/roberta-base-32k)에서 다운로드 받을 수 있습니다.

사전 학습이 완료된 모델의 성능을 평가하기 위해 다양한 데이터셋으로 미세조정을 실시하였습니다. 노이즈가 많은 환경을 고려하여 미세조정을 평가할 때 평가데이터셋을 크게 3종류로 변형하였습니다.
* 기본: 변형 적용 안 함
* NoSpace: 평가 텍스트에서 공백을 모두 제거
* AllSpace: 평가 텍스트의 각 글자 사이에 공백 모두 삽입
* Random: 20%의 확률로 공백을 삽입하거나 제거함

<table>
<caption>평가 결과 요약</caption>
<tr><th>모델</th><th>NSMC</th><th>KLUE YNAT</th></tr>
<tr><th>Kiwi RoBERTa Base</th><td><b>0.8992</b></td><td><b>0.8501</b></td></tr>
<tr><th>Klue RoBERTa Base</th><td>0.8282</td><td>0.7088</td></tr>
<tr><th>Beomi KcBert Base</th><td>0.8353</td><td>0.6456</td></tr>
<tr><th>HanBert 54kN Base</th><td>0.8363</td><td>0.7345</td></tr>
</table>

* 기본, NoSpace, AllSpace, Random 테스트 결과를 평균낸 것
* 변형이 적용 안 된 평가셋에 대해서는 Klue 모델이 제일 좋은 성능을 내었으나, 공백 오류가 들어갈 수록 모델의 성능이 급하락
* Kiwi 모델의 경우 공백 오류에 대해 대체적으로 강건한 성능을 보임

### 미세조정: NSMC

따라해보기
```bash
python src/finetuning/sequence_classification.py --model_name_or_path kiwi-farm/roberta-base-32k --output_dir results --dataset nsmc --key document --num_train_epochs 2

python src/finetuning/sequence_classification.py --model_name_or_path klue/roberta-base --output_dir results --dataset nsmc --key document --num_train_epochs 2

python src/finetuning/sequence_classification.py --model_name_or_path beomi/kcbert-base --output_dir results --dataset nsmc --key document --num_train_epochs 2
```

<table>
<caption>Kiwi RoBERTa Base</caption>
<tr><th></th><th>기본</th><th>NoSpace</th><th>AllSpace</th><th>Random</th></tr>
<tr><th>Train 기본</th><td>0.90852</td><td>0.90304</td><td>0.89204</td><td>0.8933</td></tr>
<tr><th>Train NoSpace</th><td>0.90894</td><td>0.90692</td><td>0.89142</td><td>0.897</td></tr>
<tr><th>Train AllSpace</th><td>0.9055</td><td>0.89748</td><td>0.90544</td><td>0.897</td></tr>
<tr><th>Train Random</th><td>0.9063</td><td>0.9054</td><td>0.9006</td><td>0.90262</td></tr>
</table>

<table>
<caption>Klue RoBERTa Base</caption>
<tr><th></th><th>기본</th><th>NoSpace</th><th>AllSpace</th><th>Random</th></tr>
<tr><th>Train 기본</th><td>0.91336</td><td>0.88068</td><td>0.7013</td><td>0.81746</td></tr>
<tr><th>Train NoSpace</th><td>0.91014</td><td>0.8928</td><td>0.73992</td><td>0.84966</td></tr>
<tr><th>Train AllSpace</th><td>0.88248</td><td>0.84712</td><td>0.89244</td><td>0.85532</td></tr>
<tr><th>Train Random</th><td>0.9039</td><td>0.88418</td><td>0.8723</td><td>0.88838</td></tr>
</table>

<table>
<caption>Beomi KcBert Base</caption>
<tr><th></th><th>기본</th><th>NoSpace</th><th>AllSpace</th><th>Random</th></tr>
<tr><th>Train 기본</th><td>0.90508</td><td>0.88036</td><td>0.73222</td><td>0.82366</td></tr>
<tr><th>Train NoSpace</th><td>0.89216</td><td>0.88262</td><td>0.76896</td><td>0.83242</td></tr>
<tr><th>Train AllSpace</th><td>0.85986</td><td>0.84332</td><td>0.88926</td><td>0.8565</td></tr>
<tr><th>Train Random</th><td>0.89212</td><td>0.87988</td><td>0.86962</td><td>0.88248</td></tr>
</table>

<table>
<caption>HanBert 54kN Base</caption>
<tr><th></th><th>기본</th><th>NoSpace</th><th>AllSpace</th><th>Random</th></tr>
<tr><th>Train 기본</th><td>0.90594</td><td>0.8733</td><td>0.74226</td><td>0.82358</td></tr>
<tr><th>Train NoSpace</th><td>0.89868</td><td>0.8911</td><td>0.8171</td><td>0.8501</td></tr>
<tr><th>Train AllSpace</th><td>0.87606</td><td>0.85156</td><td>0.88936</td><td>0.85506</td></tr>
<tr><th>Train Random</th><td>0.89408</td><td>0.88142</td><td>0.87072</td><td>0.88</td></tr>
</table>

### 미세조정: Klue YNAT
따라해보기
```bash
python src/finetuning/sequence_classification.py --model_name_or_path kiwi-farm/roberta-base-32k --output_dir results --dataset klue --subset ynat --key title --num_train_epochs 3

python src/finetuning/sequence_classification.py --model_name_or_path klue/roberta-base --output_dir results --dataset klue --subset ynat --key title --num_train_epochs 3

python src/finetuning/sequence_classification.py --model_name_or_path beomi/kcbert-base --output_dir results --dataset klue --subset ynat --key title --num_train_epochs 3
```

<table>
<caption>Kiwi RoBERTa Base</caption>
<tr><th></th><th>기본</th><th>NoSpace</th><th>AllSpace</th><th>Random</th></tr>
<tr><th>Train 기본</th><td>0.86570</td><td>0.85275</td><td>0.84396</td><td>0.83814</td></tr>
<tr><th>Train NoSpace</th><td>0.86274</td><td>0.85560</td><td>0.84396</td><td>0.84671</td></tr>
<tr><th>Train AllSpace</th><td>0.86449</td><td>0.84396</td><td>0.85582</td><td>0.83902</td></tr>
<tr><th>Train Random</th><td>0.86603</td><td>0.85736</td><td>0.84385</td><td>0.84737</td></tr>
</table>

<table>
<caption>Klue RoBERTa Base</caption>
<tr><th></th><th>기본</th><th>NoSpace</th><th>AllSpace</th><th>Random</th></tr>
<tr><th>Train 기본</th><td>0.86845</td><td>0.82431</td><td>0.43043</td><td>0.71186</td></tr>
<tr><th>Train NoSpace</th><td>0.87152</td><td>0.85703</td><td>0.53167</td><td>0.76128</td></tr>
<tr><th>Train AllSpace</th><td>0.80125</td><td>0.77610</td><td>0.78401</td><td>0.75974</td></tr>
<tr><th>Train Random</th><td>0.86054</td><td>0.84495</td><td>0.68957</td><td>0.81058</td></tr>
</table>

<table>
<caption>Beomi KcBert Base</caption>
<tr><th></th><th>기본</th><th>NoSpace</th><th>AllSpace</th><th>Random</th></tr>
<tr><th>Train 기본</th><td>0.83770</td><td>0.79620</td><td>0.29559</td><td>0.65279</td></tr>
<tr><th>Train NoSpace</th><td>0.82980</td><td>0.81618</td><td>0.31997</td><td>0.67179</td></tr>
<tr><th>Train AllSpace</th><td>0.74371</td><td>0.71845</td><td>0.77303</td><td>0.72823</td></tr>
<tr><th>Train Random</th><td>0.81805</td><td>0.80432</td><td>0.59591</td><td>0.77709</td></tr>
</table>

<table>
<caption>HanBert 54kN Base</caption>
<tr><th></th><th>기본</th><th>NoSpace</th><th>AllSpace</th><th>Random</th></tr>
<tr><th>Train 기본</th><td>0.86680</td><td>0.82573</td><td>0.51564</td><td>0.72976</td></tr>
<tr><th>Train NoSpace</th><td>0.85297</td><td>0.82595</td><td>0.50444</td><td>0.73448</td></tr>
<tr><th>Train AllSpace</th><td>0.77105</td><td>0.74536</td><td>0.76798</td><td>0.73679</td></tr>
<tr><th>Train Random</th><td>0.84901</td><td>0.81969</td><td>0.65872</td><td>0.78851</td></tr>
</table>

## 참고
* 키위 농장의 huggingface 페이지: https://huggingface.co/kiwi-farm
* KLUE 모델 및 데이터셋: https://github.com/KLUE-benchmark/KLUE
* NSMC 데이터셋: https://github.com/e9t/nsmc
* Beomi KcBert: https://github.com/Beomi/KcBERT
* HanBert: https://github.com/monologg/HanBert-Transformers
