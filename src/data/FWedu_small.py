# from tqdm import tqdm
# import numpy as np
# import tiktoken
# from datasets import load_dataset
# import os

# # Set path to store preprocessed binary data
# FWEDU_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/fineweb_edu_small/")

# tknzr = tiktoken.get_encoding("gpt2")

# def get_fineweb_edu_small(num_proc=40):
#     if not os.path.exists(os.path.join(FWEDU_DATA_PATH, "train.bin")):
#         os.makedirs(FWEDU_DATA_PATH, exist_ok=True)

#         print("Downloading and loading FineWeb-Edu dataset from HuggingFace...")
#         # dataset = load_dataset("HuggingFaceFW/fineweb-edu", "default")
#         dataset = load_dataset("HuggingFaceFW/fineweb-edu", "default", streaming=True, trust_remote_code=True)



#         # PATCH: Add dummy "date" column if it's missing
#         if "date" not in dataset["train"].column_names:
#             dataset = dataset.map(lambda x: {**x, "date": ""})


#         split_dataset = dataset["train"].train_test_split(
#             test_size=0.0005, seed=2357, shuffle=True
#         )
#         split_dataset["val"] = split_dataset.pop("test")

#         def process(example):
#             ids = tknzr.encode_ordinary(example["text"])
#             ids.append(tknzr.eot_token)
#             return {"ids": ids, "len": len(ids)}

#         tokenized = split_dataset.map(
#             process,
#             remove_columns=["text"],
#             desc="Tokenizing the splits",
#             num_proc=num_proc,
#         )

#         for split, dset in tokenized.items():
#             arr_len = np.sum(dset["len"])
#             filename = os.path.join(FWEDU_DATA_PATH, f"{split}.bin")
#             dtype = np.uint16
#             arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
#             total_batches = min(1024, len(dset))

#             idx = 0
#             for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
#                 batch = dset.shard(
#                     num_shards=total_batches, index=batch_idx, contiguous=True
#                 ).with_format("numpy")
#                 arr_batch = np.concatenate(batch["ids"])
#                 arr[idx : idx + len(arr_batch)] = arr_batch
#                 idx += len(arr_batch)
#             arr.flush()

#     train_data = np.memmap(
#         os.path.join(FWEDU_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r"
#     )
#     val_data = np.memmap(
#         os.path.join(FWEDU_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r"
#     )

#     return {"train": train_data, "val": val_data}

# from tqdm import tqdm
# import numpy as np
# import tiktoken
# from datasets import load_dataset
# import os

# FWEDU_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/fineweb_edu_small/")
# tknzr = tiktoken.get_encoding("gpt2")

# def get_fineweb_edu_small(num_proc=40, max_examples=10000):
#     if not os.path.exists(os.path.join(FWEDU_DATA_PATH, "train.bin")):
#         os.makedirs(FWEDU_DATA_PATH, exist_ok=True)

#         print("Downloading and loading FineWeb-Edu dataset from HuggingFace...")
#         dataset_iter = load_dataset(
#             "HuggingFaceFW/fineweb-edu",
#             "default",
#             streaming=True,
#             trust_remote_code=True,
#         )["train"]

#         # Convert to a list of dicts
#         dataset = list(dataset_iter.take(max_examples))

#         # Add 'date' column if missing
#         for example in dataset:
#             if "date" not in example:
#                 example["date"] = ""

#         # Use HF Dataset interface for convenience
#         import datasets
#         dataset = datasets.Dataset.from_list(dataset)

#         split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
#         split_dataset["val"] = split_dataset.pop("test")

#         def process(example):
#             ids = tknzr.encode_ordinary(example["text"])
#             ids.append(tknzr.eot_token)
#             return {"ids": ids, "len": len(ids)}

#         tokenized = split_dataset.map(
#             process,
#             remove_columns=["text"],
#             desc="Tokenizing the splits",
#             num_proc=num_proc,
#         )

#         for split, dset in tokenized.items():
#             arr_len = np.sum(dset["len"])
#             filename = os.path.join(FWEDU_DATA_PATH, f"{split}.bin")
#             dtype = np.uint16
#             arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
#             total_batches = min(1024, len(dset))

#             idx = 0
#             for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
#                 batch = dset.shard(
#                     num_shards=total_batches, index=batch_idx, contiguous=True
#                 ).with_format("numpy")
#                 arr_batch = np.concatenate(batch["ids"])
#                 arr[idx : idx + len(arr_batch)] = arr_batch
#                 idx += len(arr_batch)
#             arr.flush()

#     train_data = np.memmap(os.path.join(FWEDU_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r")
#     val_data = np.memmap(os.path.join(FWEDU_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r")
#     return {"train": train_data, "val": val_data}

  
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset
import os
from datasets import Features, Value

FWEDU_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/fineweb_edu_small/")
tknzr = tiktoken.get_encoding("gpt2")

def get_fineweb_edu_small(num_proc=40):
    if not os.path.exists(os.path.join(FWEDU_DATA_PATH, "train.bin")):
        os.makedirs(FWEDU_DATA_PATH, exist_ok=True)

        print("Downloading and loading FineWeb-Edu dataset from HuggingFace...")
        features = Features({
            "text": Value("string"),
            "id": Value("string"),
            "dump": Value("string"),
            "url": Value("string"),
            "date": Value("string"),  # Force inclusion of 'date'
            "file_path": Value("string"),
            "language": Value("string"),
            "language_score": Value("float64"),
            "token_count": Value("int64"),
            "score": Value("float64"),
            "int_score": Value("int64"),
        })

        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "default",
            features=features,
            trust_remote_code=True,
        )["train"]

        # Add 'date' column if missing
        if "date" not in dataset.column_names:
            dataset = dataset.map(lambda x: {"date": ""}, remove_columns=[], desc="Adding missing 'date' column")

        # Split into train/val
        split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        split_dataset["val"] = split_dataset.pop("test")

        def process(example):
            ids = tknzr.encode_ordinary(example["text"])
            ids.append(tknzr.eot_token)
            return {"ids": ids, "len": len(ids)}

        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="Tokenizing the splits",
            num_proc=num_proc,
        )

        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(FWEDU_DATA_PATH, f"{split}.bin")
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    train_data = np.memmap(os.path.join(FWEDU_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(FWEDU_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r")
    return {"train": train_data, "val": val_data}
