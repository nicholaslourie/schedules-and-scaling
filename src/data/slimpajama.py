import logging

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import os


logger = logging.getLogger(__name__)

tknzr = tiktoken.get_encoding("gpt2")


def get_slimpajama_data(datasets_dir, num_proc=40):
    SPJ_DATA_PATH = os.path.join(datasets_dir, "slimpajama6B/")
    if not os.path.exists(os.path.join(SPJ_DATA_PATH, "train.bin")):
        os.makedirs(SPJ_DATA_PATH, exist_ok=True)
        dataset = load_dataset("DKYoon/SlimPajama-6B")
        dataset["val"] = dataset.pop("validation")
        if set(dataset.keys()) != {"train", "val", "test"}:
            raise RuntimeError(
                "Found unexpected splits in SlimPajama-6B.",
            )

        def process(example):
            ids = tknzr.encode_ordinary(
                example["text"]
            )  # encode_ordinary ignores any special tokens
            ids.append(
                tknzr.eot_token
            )  # add the end of text token, e.g. 50256 for gpt2 bpe
            out = {"ids": ids, "len": len(ids)}
            return out

        # tokenize the dataset
        tokenized = dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(SPJ_DATA_PATH, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    return {
        "train": os.path.join(SPJ_DATA_PATH, "train.bin"),
        "val": os.path.join(SPJ_DATA_PATH, "val.bin"),
        "test": os.path.join(SPJ_DATA_PATH, "test.bin"),
    }
