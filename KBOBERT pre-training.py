import os
import pickle
import random
import time
import torch
from torch.utils.data.dataset import Dataset
from tokenizers import BertWordPieceTokenizer
from transformers import BertConfig, BertForPreTraining, BertTokenizerFast, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import List

from filelock import FileLock

from transformers.utils import logging

logger = logging.get_logger(__name__)

class TextDatasetForNextSentencePrediction(Dataset):

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            file_path: str,
            block_size: int,
            overwrite_cache=False,
            short_seq_probability=0.1,
            nsp_probability=0.5,
    ):

        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=True)
        self.short_seq_probability = short_seq_probability
        self.nsp_probability = nsp_probability

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "cached_nsp_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        self.tokenizer = tokenizer

        lock_path = cached_features_file + ".lock"

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")
                self.documents = [[]]
                with open(file_path, encoding="utf-8") as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        line = line.strip()
                        if not line and len(self.documents[-1]) != 0:
                            self.documents.append([])
                        tokens = tokenizer.tokenize(line)
                        tokens = tokenizer.convert_tokens_to_ids(tokens)
                        if tokens:
                            self.documents[-1].append(tokens)

                logger.info(f"Creating examples from {len(self.documents)} documents.")
                self.examples = []
                for doc_index, document in enumerate(self.documents):
                    self.create_examples_from_document(document, doc_index)
                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def create_examples_from_document(self, document: List[List[int]], doc_index: int):
        max_num_tokens = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=True)
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_probability:
            target_seq_length = random.randint(2, max_num_tokens)

        current_chunk = []
        current_length = 0
        i = 0

        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)
                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])
                    tokens_b = []
                    if len(current_chunk) == 1 or random.random() < self.nsp_probability:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)
                        for _ in range(10):
                            random_document_index = random.randint(0, len(self.documents) - 1)
                            if random_document_index != doc_index:
                                break
                        random_document = self.documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
                        while True:
                            total_length = len(tokens_a) + len(tokens_b)
                            if total_length <= max_num_tokens:
                                break
                            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                            assert len(trunc_tokens) >= 1
                            if random.random() < 0.5:
                                del trunc_tokens[0]
                            else:
                                trunc_tokens.pop()

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                    token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

                    example = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                        "next_sentence_label": torch.tensor(1 if is_random_next else 0, dtype=torch.long),
                    }

                    self.examples.append(example)

                current_chunk = []
                current_length = 0

            i += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

if __name__ == "__main__":

    wp_tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False,
        )

    wp_tokenizer.train(
        files='/opt/ml/code/KBOBERT/NEWS.txt',
        vocab_size=32000,
        min_frequency=2,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        wordpieces_prefix="##"
        )

    wp_tokenizer.save_model('./')

    tokenizer = BertTokenizerFast(
        vocab_file="/opt/ml/code/KBOBERT/vocab.txt",
        max_len=64,
        do_lower_case=False,
        )

    tokenizer.add_special_tokens({'mask_token':'[MASK]'})

    # https://huggingface.co/transformers/model_doc/bert.html#bertconfig

    config = BertConfig(
        vocab_size=32000,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=128,
        type_vocab_size=2,
        pad_token_id=0,
        position_embedding_type="absolute"
    )

    model = BertForPreTraining(config=config)
    model.num_parameters()

    dataset = TextDatasetForNextSentencePrediction(
        tokenizer=tokenizer,
        file_path='/opt/ml/code/KBOBERT/NEWS.txt',
        block_size=128,
        overwrite_cache=False,
        short_seq_probability=0.1,
        nsp_probability=0.5,
        )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

    training_args = TrainingArguments(
        output_dir='./model_output',
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        save_steps=10000,
        save_total_limit=5,
        logging_steps=10000
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
        )

    trainer.train()

    trainer.save_model('./')