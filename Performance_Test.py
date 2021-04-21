from transformers import BertForMaskedLM, BertTokenizerFast, pipeline

model = BertForMaskedLM.from_pretrained('/opt/ml/code/KBOBERT/model_output')

tokenizer = BertTokenizerFast(
    vocab_file="/opt/ml/code/KBOBERT/vocab.txt",
    max_len=64,
    do_lower_case=False,
)

tokenizer.add_special_tokens({'mask_token':'[MASK]'})

nlp_fill = pipeline('fill-mask', top_k=5, model=model, tokenizer=tokenizer)
