from transformers import BertForMaskedLM, BertTokenizerFast, pipeline

model = BertForMaskedLM.from_pretrained('/opt/ml/code/KBOBERT/model_output/checkpoint-130000')

tokenizer = BertTokenizerFast(
    vocab_file="/opt/ml/code/KBOBERT/vocab.txt",
    max_len=512,
    do_lower_case=False,
)

tokenizer.add_special_tokens({'mask_token':'[MASK]'})

nlp_fill = pipeline('fill-mask', top_k=5, model=model, tokenizer=tokenizer)

print('1회 선두타자가 중전 안타로 출루했으나 1사 후 2루 도루를 시도했다가 태그 아웃됐다.')
print(nlp_fill('1회 선두타자가 중전 [MASK] 출루했으나 1사 후 2루 도루를 시도했다가 태그 아웃됐다.'))
print(nlp_fill('1회 선두타자가 중전 안타로 [MASK] 1사 후 2루 도루를 시도했다가 태그 아웃됐다.'))
print(nlp_fill('1회 선두타자가 중전 안타로 출루했으나 [MASK] 후 2루 도루를 시도했다가 태그 아웃됐다.'))
print(nlp_fill('1회 선두타자가 중전 안타로 출루했으나 1사 후 [MASK] 도루를 시도했다가 태그 아웃됐다.'))
print(nlp_fill('1회 선두타자가 중전 안타로 출루했으나 1사 후 2루 [MASK] 시도했다가 태그 아웃됐다.'))
print(nlp_fill('1회 선두타자가 중전 안타로 출루했으나 1사 후 2루 도루를 시도했다가 [MASK] 아웃됐다.'))
print(nlp_fill('1회 선두타자가 중전 안타로 출루했으나 1사 후 2루 도루를 시도했다가 태그 [MASK] 됐다.'))