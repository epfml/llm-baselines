import importlib
## load arc
import os

import numpy as np
import tiktoken
import torch
from datasets import load_dataset
from tqdm import tqdm

# Initialize the tokenizer
tknzr = tiktoken.get_encoding("gpt2")
ARC_E_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/arc_easy/")
ARC_C_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/arc_challenge/")
HELLASWAG_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/hellaswag/")
LOGIQA_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/logiqa/")
PIQA_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/piqa/")
SCIQ_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/sciq/")
HUMANEVAL_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/humaneval/")
KODCODE_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/kodcode/")
GSM8K_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/gsm8k/")
MATHQA_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/mathqa/")
MEDQA_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/medqa/")


def tokenize_with_pad(text, pad_to_multiple=1024):
    ids = tknzr.encode_ordinary(text)
    ids.append(tknzr.eot_token)
    pad_token_id = tknzr.eot_token
    # Calculate padding length (next multiple of pad_multiple)
    padded_length = (
        (len(ids) + pad_to_multiple - 1) // pad_to_multiple
    ) * pad_to_multiple
    # Initialize the padded array with pad token (not zeros)
    padded_tokens = np.ones(padded_length, dtype=np.uint16) * pad_token_id
    padded_tokens[: len(ids)] = ids
    return padded_tokens


def get_humaneval(num_proc=10, return_torch=False, pad_to_multiple=1024):
    """
    Load and process the HumanEval (for coding) dataset.
    Tokenize the text and store it in binary format for efficient loading.
    """
    if not os.path.exists(os.path.join(HUMANEVAL_DATA_PATH, "val.bin")):
        os.makedirs(HUMANEVAL_DATA_PATH, exist_ok=True)

        # Load the HumanEval dataset from Hugging Face Datasets
        human_eval_path = "openai/openai_humaneval"
        dataset = load_dataset(human_eval_path, trust_remote_code=True)
        split_dataset = dataset["test"].train_test_split(
            test_size=0.5, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")
        data_dict = {
            "train": split_dataset["train"],
            "val": split_dataset["val"],
        }

        def process(example):
            """
            Tokenize the example text by encoding it into token IDs.
            """
            prompt = example["prompt"]
            completion = example["canonical_solution"]

            concatenated_text = f"{prompt} \n {completion}"
            # print(concatenated_text)
            ids = tokenize_with_pad(
                text=concatenated_text, pad_to_multiple=pad_to_multiple
            )
            return {"ids": ids, "len": len(ids)}

        # Tokenize and map the dataset
        tokenized = {}
        for split, dset in data_dict.items():
            tokenized[split] = dset.map(
                process,
                remove_columns=[
                    "task_id",
                    "prompt",
                    "canonical_solution",
                    "test",
                    "entry_point",
                ],
                desc=f"Tokenizing {split} split",
                num_proc=num_proc,
            )

        # Concatenate all the token IDs into one large binary file per split
        for split, dset in tokenized.items():
            # Save token IDs length
            len_arr = np.array(dset["len"], dtype=np.uint16)
            with open(os.path.join(HUMANEVAL_DATA_PATH, f"{split}.len"), "wb") as f:
                np.save(f, len_arr)
            # Total number of tokens
            arr_len = np.sum(dset["len"])
            filename = os.path.join(HUMANEVAL_DATA_PATH, f"{split}.bin")
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 10

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
                batch = (
                    dset.shard(
                        num_shards=total_batches, index=batch_idx, contiguous=True
                    )
                    .with_format("numpy")
                    .to_dict()
                )
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    # Load tokenized binary files for training, validation
    train_data = np.memmap(
        os.path.join(HUMANEVAL_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(HUMANEVAL_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r"
    )

    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.uint16))
        val_data = torch.tensor(np.array(val_data, dtype=np.uint16))
    print(f"Benchmark HumanEval: train[{len(train_data)}] | val[{len(val_data)}]")
    return {
        "train": train_data,
        "train_len": np.load(os.path.join(HUMANEVAL_DATA_PATH, "train.len")),
        "val": val_data,
        "val_len": np.load(os.path.join(HUMANEVAL_DATA_PATH, "val.len")),
    }


def get_kodcode(num_proc=10, return_torch=False, pad_to_multiple=1024):
    """
    Load and process the KodCode (for code reasoning) dataset.
    Tokenize the text and store it in binary format for efficient loading.
    """
    if not os.path.exists(os.path.join(KODCODE_DATA_PATH, "val.bin")):
        os.makedirs(KODCODE_DATA_PATH, exist_ok=True)

        # Load the GSM8K dataset from Hugging Face Datasets
        dataset = load_dataset("KodCode/KodCode-V1-SFT-R1", trust_remote_code=True)
        dataset = dataset["train"].train_test_split(
            test_size=0.1, seed=2357, shuffle=True
        )
        data_dict = {
            "train": dataset["train"],
            "val": dataset["test"],
        }

        def process(example):
            """
            Tokenize the example text by encoding it into token IDs.
            """
            question = example["question"]
            answer = example["solution"]

            concatenated_text = f"{question}\n{answer}"
            # print(concatenated_text)
            ids = tokenize_with_pad(
                text=concatenated_text, pad_to_multiple=pad_to_multiple
            )
            return {"ids": ids, "len": len(ids)}

        # Tokenize and map the dataset
        tokenized = {}
        for split, dset in data_dict.items():
            tokenized[split] = dset.map(
                process,
                remove_columns=[
                    "style",
                    "question_id",
                    "subset",
                    "question",
                    "solution",
                    "test_code",
                    "test_info",
                    "gpt_pass_sequence",
                    "gpt_pass_trial_num",
                    "gpt_difficulty",
                    "gpt_pass_percentage",
                    "r1_pass_sequence",
                    "r1_pass_trial_num",
                    "r1_correctness",
                    "r1_solution",
                    "metadata",
                    "conversations",
                ],
                desc=f"Tokenizing {split} split",
                num_proc=num_proc,
            )

        # Concatenate all the token IDs into one large binary file per split
        for split, dset in tokenized.items():
            # Save token IDs length
            len_arr = np.array(dset["len"], dtype=np.uint16)
            with open(os.path.join(KODCODE_DATA_PATH, f"{split}.len"), "wb") as f:
                np.save(f, len_arr)
            # Total number of tokens
            arr_len = np.sum(dset["len"])
            filename = os.path.join(KODCODE_DATA_PATH, f"{split}.bin")
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 10

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
                batch = (
                    dset.shard(
                        num_shards=total_batches, index=batch_idx, contiguous=True
                    )
                    .with_format("numpy")
                    .to_dict()
                )
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    # Load tokenized binary files for training, validation
    train_data = np.memmap(
        os.path.join(KODCODE_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(KODCODE_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r"
    )

    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.uint16))
        val_data = torch.tensor(np.array(val_data, dtype=np.uint16))
    print(f"Benchmark KodCode: train[{len(train_data)}] | val[{len(val_data)}]")
    return {
        "train": train_data,
        "train_len": np.load(os.path.join(KODCODE_DATA_PATH, "train.len")),
        "val": val_data,
        "val_len": np.load(os.path.join(KODCODE_DATA_PATH, "val.len")),
    }


def get_gsm8k(num_proc=10, return_torch=False, pad_to_multiple=1024):
    """
    Load and process the GSM8K (for math) dataset.
    Tokenize the text and store it in binary format for efficient loading.
    """
    if not os.path.exists(os.path.join(GSM8K_DATA_PATH, "val.bin")):
        os.makedirs(GSM8K_DATA_PATH, exist_ok=True)

        # Load the GSM8K dataset from Hugging Face Datasets
        dataset = load_dataset("openai/gsm8k", "main", trust_remote_code=True)
        data_dict = {
            "train": dataset["train"],
            "val": dataset["test"],
        }

        def process(example):
            """
            Tokenize the example text by encoding it into token IDs.
            """
            question = example["question"]
            answer = example["answer"]

            concatenated_text = f"{question}\n{answer}"
            # print(concatenated_text)
            ids = tokenize_with_pad(
                text=concatenated_text, pad_to_multiple=pad_to_multiple
            )
            return {"ids": ids, "len": len(ids)}

        # Tokenize and map the dataset
        tokenized = {}
        for split, dset in data_dict.items():
            tokenized[split] = dset.map(
                process,
                remove_columns=["question", "answer"],
                desc=f"Tokenizing {split} split",
                num_proc=num_proc,
            )

        # Concatenate all the token IDs into one large binary file per split
        for split, dset in tokenized.items():
            # Save token IDs length
            len_arr = np.array(dset["len"], dtype=np.uint16)
            with open(os.path.join(GSM8K_DATA_PATH, f"{split}.len"), "wb") as f:
                np.save(f, len_arr)
            # Total number of tokens
            arr_len = np.sum(dset["len"])
            filename = os.path.join(GSM8K_DATA_PATH, f"{split}.bin")
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 10

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
                batch = (
                    dset.shard(
                        num_shards=total_batches, index=batch_idx, contiguous=True
                    )
                    .with_format("numpy")
                    .to_dict()
                )
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    # Load tokenized binary files for training, validation
    train_data = np.memmap(
        os.path.join(GSM8K_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(GSM8K_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r"
    )

    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.uint16))
        val_data = torch.tensor(np.array(val_data, dtype=np.uint16))
    print(f"Benchmark GSM8k: train[{len(train_data)}] | val[{len(val_data)}]")
    return {
        "train": train_data,
        "train_len": np.load(os.path.join(GSM8K_DATA_PATH, "train.len")),
        "val": val_data,
        "val_len": np.load(os.path.join(GSM8K_DATA_PATH, "val.len")),
    }


def get_mathqa(num_proc=10, return_torch=False, pad_to_multiple=1024):
    """
    Load and process the MATH-QA dataset.
    Tokenize the text and store it in binary format for efficient loading.
    """
    if not os.path.exists(os.path.join(MATHQA_DATA_PATH, "val.bin")):
        os.makedirs(MATHQA_DATA_PATH, exist_ok=True)

        # Load the MATH-QA dataset from Hugging Face Datasets
        dataset = load_dataset("allenai/math_qa", trust_remote_code=True)
        data_dict = {
            "train": dataset["train"],
            "val": dataset["test"],
        }

        def process(example):
            """
            Tokenize the example text by encoding it into token IDs.
            """
            question = example["Problem"]
            choices = {
                d.strip()[0]: d.split(")")[-1].strip()
                for d in example["options"].split(",")
            }
            answer = choices.get(example["correct"])

            concatenated_text = f"{question} {answer}"
            # print(concatenated_text)
            ids = tokenize_with_pad(
                text=concatenated_text, pad_to_multiple=pad_to_multiple
            )
            return {"ids": ids, "len": len(ids)}

        # Tokenize and map the dataset
        tokenized = {}
        for split, dset in data_dict.items():
            tokenized[split] = dset.map(
                process,
                remove_columns=[
                    "Problem",
                    "Rationale",
                    "options",
                    "correct",
                    "annotated_formula",
                    "linear_formula",
                    "category",
                ],
                desc=f"Tokenizing {split} split",
                num_proc=num_proc,
            )

        # Concatenate all the token IDs into one large binary file per split
        for split, dset in tokenized.items():
            # Save token IDs length
            len_arr = np.array(dset["len"], dtype=np.uint16)
            with open(os.path.join(MATHQA_DATA_PATH, f"{split}.len"), "wb") as f:
                np.save(f, len_arr)
            # Total number of tokens
            arr_len = np.sum(dset["len"])
            filename = os.path.join(MATHQA_DATA_PATH, f"{split}.bin")
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 10

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
                batch = (
                    dset.shard(
                        num_shards=total_batches, index=batch_idx, contiguous=True
                    )
                    .with_format("numpy")
                    .to_dict()
                )
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    # Load tokenized binary files for training, validation
    train_data = np.memmap(
        os.path.join(MATHQA_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(MATHQA_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r"
    )

    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.uint16))
        val_data = torch.tensor(np.array(val_data, dtype=np.uint16))
    print(f"Benchmark MATHQA: train[{len(train_data)}] | val[{len(val_data)}]")
    return {
        "train": train_data,
        "train_len": np.load(os.path.join(MATHQA_DATA_PATH, "train.len")),
        "val": val_data,
        "val_len": np.load(os.path.join(MATHQA_DATA_PATH, "val.len")),
    }


def get_medqa(num_proc=10, return_torch=False, pad_to_multiple=1024):
    """
    Load and process the MEDQA dataset.
    Tokenize the text and store it in binary format for efficient loading.
    """
    if not os.path.exists(os.path.join(MEDQA_DATA_PATH, "val.bin")):
        os.makedirs(MEDQA_DATA_PATH, exist_ok=True)

        # Load the MATH-QA dataset from Hugging Face Datasets
        dataset = load_dataset("bigbio/med_qa", trust_remote_code=True)
        data_dict = {
            "train": dataset["train"],
            "val": dataset["test"],
        }

        def process(example):
            """
            Tokenize the example text by encoding it into token IDs.
            """
            question = example["question"]
            answer = example["answer"]

            concatenated_text = f"{question} {answer}"
            # print(concatenated_text)
            ids = tokenize_with_pad(
                text=concatenated_text, pad_to_multiple=pad_to_multiple
            )
            return {"ids": ids, "len": len(ids)}

        # Tokenize and map the dataset
        tokenized = {}
        for split, dset in data_dict.items():
            tokenized[split] = dset.map(
                process,
                remove_columns=[
                    "meta_info",
                    "question",
                    "answer_idx",
                    "answer",
                    "options",
                ],
                desc=f"Tokenizing {split} split",
                num_proc=num_proc,
            )

        # Concatenate all the token IDs into one large binary file per split
        for split, dset in tokenized.items():
            # Save token IDs length
            len_arr = np.array(dset["len"], dtype=np.uint16)
            with open(os.path.join(MEDQA_DATA_PATH, f"{split}.len"), "wb") as f:
                np.save(f, len_arr)
            # Total number of tokens
            arr_len = np.sum(dset["len"])
            filename = os.path.join(MEDQA_DATA_PATH, f"{split}.bin")
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 10

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
                batch = (
                    dset.shard(
                        num_shards=total_batches, index=batch_idx, contiguous=True
                    )
                    .with_format("numpy")
                    .to_dict()
                )
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    # Load tokenized binary files for training, validation
    train_data = np.memmap(
        os.path.join(MEDQA_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(MEDQA_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r"
    )

    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.uint16))
        val_data = torch.tensor(np.array(val_data, dtype=np.uint16))
    print(f"Benchmark MedQA: train[{len(train_data)}] | val[{len(val_data)}]")
    return {
        "train": train_data,
        "train_len": np.load(os.path.join(MEDQA_DATA_PATH, "train.len")),
        "val": val_data,
        "val_len": np.load(os.path.join(MEDQA_DATA_PATH, "val.len")),
    }


def get_arc_easy(num_proc=10, return_torch=False, pad_to_multiple=1024):
    """
    Load and process the AI2-ARC dataset.
    Tokenize the text and store it in binary format for efficient loading.
    """
    if not os.path.exists(os.path.join(ARC_E_DATA_PATH, "val.bin")):
        os.makedirs(ARC_E_DATA_PATH, exist_ok=True)

        # Load the AI2-ARC dataset from Hugging Face Datasets
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=["train", "test"])
        data_dict = {
            "train": dataset[0],
            "val": dataset[1],
        }

        def process(example):
            """
            Tokenize the example text by encoding it into token IDs.
            """
            question = example["question"]
            choices = example["choices"]
            answer = dict(zip(choices["label"], choices["text"])).get(
                example["answerKey"]
            )

            concatenated_text = f"{question} {answer}"
            # print(concatenated_text)
            ids = tokenize_with_pad(
                text=concatenated_text, pad_to_multiple=pad_to_multiple
            )
            return {"ids": ids, "len": len(ids)}

        # Tokenize and map the dataset
        tokenized = {}
        for split, dset in data_dict.items():
            tokenized[split] = dset.map(
                process,
                remove_columns=["question", "choices", "answerKey"],
                desc=f"Tokenizing {split} split",
                num_proc=num_proc,
            )

        # Concatenate all the token IDs into one large binary file per split
        for split, dset in tokenized.items():
            # Save token IDs length
            len_arr = np.array(dset["len"], dtype=np.uint16)
            with open(os.path.join(ARC_E_DATA_PATH, f"{split}.len"), "wb") as f:
                np.save(f, len_arr)
            # Total number of tokens
            arr_len = np.sum(dset["len"])
            filename = os.path.join(ARC_E_DATA_PATH, f"{split}.bin")
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 10

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
                batch = (
                    dset.shard(
                        num_shards=total_batches, index=batch_idx, contiguous=True
                    )
                    .with_format("numpy")
                    .to_dict()
                )
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    # Load tokenized binary files for training, validation
    train_data = np.memmap(
        os.path.join(ARC_E_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(ARC_E_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r"
    )

    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.uint16))
        val_data = torch.tensor(np.array(val_data, dtype=np.uint16))
    print(f"Benchmark ARC-Easy: train[{len(train_data)}] | val[{len(val_data)}]")
    return {
        "train": train_data,
        "train_len": np.load(os.path.join(ARC_E_DATA_PATH, "train.len")),
        "val": val_data,
        "val_len": np.load(os.path.join(ARC_E_DATA_PATH, "val.len")),
    }


def get_arc_challenge(num_proc=10, return_torch=False, pad_to_multiple=1024):
    """
    Load and process the AI2-ARC dataset.
    Tokenize the text and store it in binary format for efficient loading.
    """
    if not os.path.exists(os.path.join(ARC_C_DATA_PATH, "val.bin")):
        os.makedirs(ARC_C_DATA_PATH, exist_ok=True)

        # Load the AI2-ARC dataset from Hugging Face Datasets
        dataset = load_dataset(
            "allenai/ai2_arc", "ARC-Challenge", split=["train", "test"]
        )
        data_dict = {
            "train": dataset[0],
            "val": dataset[1],
        }

        def process(example):
            """
            Tokenize the example text by encoding it into token IDs.
            """
            question = example["question"]
            choices = example["choices"]
            answer = dict(zip(choices["label"], choices["text"])).get(
                example["answerKey"]
            )

            concatenated_text = f"{question} {answer}"
            # print(concatenated_text)
            ids = tokenize_with_pad(
                text=concatenated_text, pad_to_multiple=pad_to_multiple
            )
            return {"ids": ids, "len": len(ids)}

        # Tokenize and map the dataset
        tokenized = {}
        for split, dset in data_dict.items():
            tokenized[split] = dset.map(
                process,
                remove_columns=["question", "choices", "answerKey"],
                desc=f"Tokenizing {split} split",
                num_proc=num_proc,
            )

        # Concatenate all the token IDs into one large binary file per split
        for split, dset in tokenized.items():
            # Save token IDs length
            len_arr = np.array(dset["len"], dtype=np.uint16)
            with open(os.path.join(ARC_C_DATA_PATH, f"{split}.len"), "wb") as f:
                np.save(f, len_arr)
            # Total number of tokens
            arr_len = np.sum(dset["len"])
            filename = os.path.join(ARC_C_DATA_PATH, f"{split}.bin")
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 10

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
                batch = (
                    dset.shard(
                        num_shards=total_batches, index=batch_idx, contiguous=True
                    )
                    .with_format("numpy")
                    .to_dict()
                )
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    # Load tokenized binary files for training, validation
    train_data = np.memmap(
        os.path.join(ARC_C_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(ARC_C_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r"
    )

    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.uint16))
        val_data = torch.tensor(np.array(val_data, dtype=np.uint16))
    print(f"Benchmark ARC-Challenge: train[{len(train_data)}] | val[{len(val_data)}]")
    return {
        "train": train_data,
        "train_len": np.load(os.path.join(ARC_C_DATA_PATH, "train.len")),
        "val": val_data,
        "val_len": np.load(os.path.join(ARC_C_DATA_PATH, "val.len")),
    }


def get_hellaswag(num_proc=10, return_torch=False, pad_to_multiple=1024):
    """
    Load and process the HellaSwag dataset.
    Tokenize the text and store it in binary format for efficient loading.
    """
    if not os.path.exists(os.path.join(HELLASWAG_DATA_PATH, "val.bin")):
        os.makedirs(HELLASWAG_DATA_PATH, exist_ok=True)

        # Load the HellaSwag dataset from Hugging Face Datasets
        dataset = load_dataset("Rowan/hellaswag", split=["train", "validation"])
        data_dict = {
            "train": dataset[0],
            "val": dataset[1],
        }

        def process(example):
            """
            Tokenize the example text by encoding it into token IDs.
            """
            context = example["ctx"]
            ending_options = example["endings"]
            answer = ending_options[int(example["label"])]
            concatenated_text = f"{context} {answer}"
            # print(concatenated_text)
            ids = tokenize_with_pad(
                text=concatenated_text, pad_to_multiple=pad_to_multiple
            )
            return {"ids": ids, "len": len(ids)}

        # Tokenize and map the dataset
        tokenized = {}
        for split, dset in data_dict.items():
            tokenized[split] = dset.map(
                process,
                remove_columns=["ctx", "endings", "label"],
                desc=f"Tokenizing {split} split",
                num_proc=num_proc,
            )

        # Concatenate all the token IDs into one large binary file per split
        for split, dset in tokenized.items():
            # Save token IDs length
            len_arr = np.array(dset["len"], dtype=np.uint16)
            with open(os.path.join(HELLASWAG_DATA_PATH, f"{split}.len"), "wb") as f:
                np.save(f, len_arr)
            # Total number of tokens
            arr_len = np.sum(dset["len"])
            filename = os.path.join(HELLASWAG_DATA_PATH, f"{split}.bin")
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 10

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
                batch = (
                    dset.shard(
                        num_shards=total_batches, index=batch_idx, contiguous=True
                    )
                    .with_format("numpy")
                    .to_dict()
                )
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    # Load tokenized binary files for training, validation
    train_data = np.memmap(
        os.path.join(HELLASWAG_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(HELLASWAG_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r"
    )

    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.uint16))
        val_data = torch.tensor(np.array(val_data, dtype=np.uint16))

    print(f"Benchmark Hellaswag: train[{len(train_data)}] | val[{len(val_data)}]")
    return {
        "train": train_data,
        "train_len": np.load(os.path.join(HELLASWAG_DATA_PATH, "train.len")),
        "val": val_data,
        "val_len": np.load(os.path.join(HELLASWAG_DATA_PATH, "val.len")),
    }


def get_logiqa(num_proc=10, return_torch=False, pad_to_multiple=1024):
    """
    Load and process the LogiQA dataset.
    Tokenize the text and store it in binary format for efficient loading.
    """
    if not os.path.exists(os.path.join(LOGIQA_DATA_PATH, "val.bin")):
        os.makedirs(LOGIQA_DATA_PATH, exist_ok=True)

        # Load the LogiQA dataset from HuggingFace Datasets
        dataset = load_dataset("lucasmccabe/logiqa", split=["train", "test"])
        data_dict = {
            "train": dataset[0],
            "val": dataset[1],
        }

        def process(example):
            """
            Tokenize the example text by encoding it into token IDs.
            """
            context = example["context"]
            query = example["query"]
            correct_option = example["correct_option"]
            options = example["options"]
            answer = options[correct_option]
            concatenated_text = f"{context} {query} {answer}"
            # concatenated_text = f"Context: {context} Query: {query} Answer: {answer}" # TODO : try this ?
            ids = tokenize_with_pad(
                text=concatenated_text, pad_to_multiple=pad_to_multiple
            )
            return {"ids": ids, "len": len(ids)}

        # Tokenize and map the dataset
        tokenized = {}
        for split, dset in data_dict.items():
            tokenized[split] = dset.map(
                process,
                remove_columns=["context", "query", "correct_option", "options"],
                desc=f"Tokenizing {split} split",
                num_proc=num_proc,
            )

        # Concatenate all the token IDs into one large binary file per split
        for split, dset in tokenized.items():
            # Save token IDs length
            len_arr = np.array(dset["len"], dtype=np.uint16)
            with open(os.path.join(LOGIQA_DATA_PATH, f"{split}.len"), "wb") as f:
                np.save(f, len_arr)
            # total number of tokens
            arr_len = np.sum(dset["len"])
            filename = os.path.join(LOGIQA_DATA_PATH, f"{split}.bin")
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 10

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
                batch = (
                    dset.shard(
                        num_shards=total_batches, index=batch_idx, contiguous=True
                    )
                    .with_format("numpy")
                    .to_dict()
                )
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    # Load tokenized binary files for training, validation
    train_data = np.memmap(
        os.path.join(LOGIQA_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(LOGIQA_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r"
    )

    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.uint16))
        val_data = torch.tensor(np.array(val_data, dtype=np.uint16))
    print(f"Benchmark LogiQA: train[{len(train_data)}] | val[{len(val_data)}]")
    return {
        "train": train_data,
        "train_len": np.load(os.path.join(LOGIQA_DATA_PATH, "train.len")),
        "val": val_data,
        "val_len": np.load(os.path.join(LOGIQA_DATA_PATH, "val.len")),
    }


def get_piqa(num_proc=10, return_torch=False, pad_to_multiple=1024):
    """
    Load and process the PIQA dataset.
    Tokenize the text and store it in binary format for efficient loading.
    """
    if not os.path.exists(os.path.join(PIQA_DATA_PATH, "val.bin")):
        os.makedirs(PIQA_DATA_PATH, exist_ok=True)

        # Load the PIQA dataset from HuggingFace Datasets
        dataset = load_dataset("piqa", split=["train", "test"])

        data_dict = {
            "train": dataset[0],
            "val": dataset[1],
        }

        def process(example):
            """
            Tokenize the example text by encoding it into token IDs.
            """
            goal = example["goal"]
            sols = [example["sol1"], example["sol2"]]
            label = example["label"]
            sol = sols[label]
            concatenated_text = f"{goal} {sol}"  # only include the correct solution
            ids = tokenize_with_pad(
                text=concatenated_text, pad_to_multiple=pad_to_multiple
            )
            return {"ids": ids, "len": len(ids)}

        # Tokenize and map the dataset
        tokenized = {}
        for split, dset in data_dict.items():
            tokenized[split] = dset.map(
                process,
                remove_columns=["goal", "sol1", "sol2", "label"],
                desc=f"Tokenizing {split} split",
                num_proc=num_proc,
            )

        # Concatenate all the token IDs into one large binary file per split
        for split, dset in tokenized.items():
            # Save token IDs length
            len_arr = np.array(dset["len"], dtype=np.uint16)
            with open(os.path.join(PIQA_DATA_PATH, f"{split}.len"), "wb") as f:
                np.save(f, len_arr)
            # total number of tokens
            arr_len = np.sum(dset["len"])
            filename = os.path.join(PIQA_DATA_PATH, f"{split}.bin")
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 10

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
                batch = (
                    dset.shard(
                        num_shards=total_batches, index=batch_idx, contiguous=True
                    )
                    .with_format("numpy")
                    .to_dict()
                )
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    # Load tokenized binary files for training, validation
    train_data = np.memmap(
        os.path.join(PIQA_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(PIQA_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r"
    )
    print(f"Benchmark PIQA: train[{len(train_data)}] | val[{len(val_data)}]")
    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.uint16))
        val_data = torch.tensor(np.array(val_data, dtype=np.uint16))

    return {
        "train": train_data,
        "train_len": np.load(os.path.join(PIQA_DATA_PATH, "train.len")),
        "val": val_data,
        "val_len": np.load(os.path.join(PIQA_DATA_PATH, "val.len")),
    }


def get_sciq(num_proc=10, return_torch=False, pad_to_multiple=1024):
    """
    Load and process the SciQ dataset.
    Tokenize the text and store it in binary format for efficient loading.
    """
    if not os.path.exists(os.path.join(SCIQ_DATA_PATH, "val.bin")):
        os.makedirs(SCIQ_DATA_PATH, exist_ok=True)

        # Load the SciQ dataset from HuggingFace Datasets
        dataset = load_dataset("sciq", split=["train", "test"])

        data_dict = {
            "train": dataset[0],
            "val": dataset[1],
        }

        def process(example):
            """
            Tokenize the example text by encoding it into token IDs.
            """
            question = example["question"]
            answer = example["correct_answer"]
            # explanation = example.get('support', '')  # Explanation text (optional)
            explantation = example["support"]
            # concatenated_text = f"Question: {question} Answer: {answer} {explantation}" #TODO: Try this?
            concatenated_text = f"{explantation}{question}{answer}"
            ids = tokenize_with_pad(
                text=concatenated_text, pad_to_multiple=pad_to_multiple
            )
            return {"ids": ids, "len": len(ids)}

        # Tokenize and map the dataset
        tokenized = {}
        for split, dset in data_dict.items():
            tokenized[split] = dset.map(
                process,
                remove_columns=[
                    "question",
                    "distractor1",
                    "distractor2",
                    "distractor3",
                    "correct_answer",
                    "support",
                ],
                desc=f"Tokenizing {split} split",
                num_proc=num_proc,
            )

        # Concatenate all the token IDs into one large binary file per split
        for split, dset in tokenized.items():
            # Save token IDs length
            len_arr = np.array(dset["len"], dtype=np.uint16)
            with open(os.path.join(SCIQ_DATA_PATH, f"{split}.len"), "wb") as f:
                np.save(f, len_arr)
            # total number of tokens
            arr_len = np.sum(dset["len"])
            filename = os.path.join(SCIQ_DATA_PATH, f"{split}.bin")
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 10

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
                batch = (
                    dset.shard(
                        num_shards=total_batches, index=batch_idx, contiguous=True
                    )
                    .with_format("numpy")
                    .to_dict()
                )
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    # Load tokenized binary files for training, validation, and test
    train_data = np.memmap(
        os.path.join(SCIQ_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(SCIQ_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r"
    )
    print(f"Benchmark SCIQ: train[{len(train_data)}] | val[{len(val_data)}]")

    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.uint16))
        val_data = torch.tensor(np.array(val_data, dtype=np.uint16))

    return {
        "train": train_data,
        "train_len": np.load(os.path.join(SCIQ_DATA_PATH, "train.len")),
        "val": val_data,
        "val_len": np.load(os.path.join(SCIQ_DATA_PATH, "val.len")),
    }


SUPPORTED_TASK_MAP = {
    "arc_easy": get_arc_easy,
    "arc_challenge": get_arc_challenge,
    "hellaswag": get_hellaswag,
    "logiqa": get_logiqa,
    "piqa": get_piqa,
    "sciq": get_sciq,
    "humaneval": get_humaneval,
    "gsm8k": get_gsm8k,
    "kodcode": get_kodcode,
    "mathqa": get_mathqa,
    "medqa": get_medqa,
}
