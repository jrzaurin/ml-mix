import json
from pathlib import Path

import tiktoken
import torch
from sft_utils import GPTModel, generate, text_to_token_ids, token_ids_to_text
from torch.utils.data import DataLoader, Dataset


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            prompt = format_input(entry)
            rejected_response = entry["rejected"]
            chosen_response = entry["chosen"]

            prompt_tokens = tokenizer.encode(prompt)
            chosen_full_text = f"{prompt}\n\n### Response:\n{chosen_response}"
            rejected_full_text = f"{prompt}\n\n### Response:\n{rejected_response}"
            chosen_full_tokens = tokenizer.encode(chosen_full_text)
            rejected_full_tokens = tokenizer.encode(rejected_full_text)

            self.encoded_texts.append(
                {
                    "prompt": prompt_tokens,
                    "chosen": chosen_full_tokens,
                    "rejected": rejected_full_tokens,
                }
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    allowed_max_length=None,
    mask_prompt_tokens=True,
    device="cpu",
):
    # Initialize lists to hold batch data
    batch_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "rejected_mask": [],
        "chosen_mask": [],
    }

    # Determine the longest sequence to set a common padding length
    max_length_common = 0
    if batch:
        for key in ["chosen", "rejected"]:
            current_max = max(len(item[key]) + 1 for item in batch)
            max_length_common = max(max_length_common, current_max)

    # Process each item in the batch
    for item in batch:
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)

        for key in ["chosen", "rejected"]:
            # Adjust padding according to the common maximum length
            sequence = item[key]
            padded = sequence + [pad_token_id] * (max_length_common - len(sequence))
            mask = torch.ones(len(padded)).bool()

            # Set mask for all padding tokens to False
            mask[len(sequence) :] = False

            # Set mask for all input tokens to False
            # +2 sets the 2 newline ("\n") tokens before "### Response" to False
            if mask_prompt_tokens:
                mask[: prompt.shape[0] + 2] = False

            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(mask)

    # Final processing
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        # Stack all sequences into a tensor for the given key
        tensor_stack = torch.stack(batch_data[key])

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]

        # Move to the specified device
        batch_data[key] = tensor_stack.to(device)

    return batch_data


def decode_tokens_from_batch(token_ids, tokenizer):
    ids_in_python_list = token_ids.flatten().tolist()
    return tokenizer.decode(ids_in_python_list)


if __name__ == "__main__":

    models_path = Path("models")
    finetuned_model_name = "gpt2-medium355M-sft-standalone.pth"

    generated_data_path = Path("generated_data")
    file_path = "instruction-data-with-preference.json"

    with open(generated_data_path / file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]

    tokenizer = tiktoken.get_encoding("gpt2")

    # example_data = data[:2]

    # example_dataset = PreferenceDataset(example_data, tokenizer)

    # example_dataloader = DataLoader(
    #     example_dataset,
    #     batch_size=2,
    #     collate_fn=custom_collate_fn,
    #     shuffle=False
    # )

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_dataset = PreferenceDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    val_dataset = PreferenceDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    test_dataset = PreferenceDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    # from the sft script
    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True,  # Query-key-value bias
    }
    model_configs = {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}
    BASE_CONFIG.update(model_configs)

    model = GPTModel(BASE_CONFIG)

    model.load_state_dict(
        torch.load(
            models_path / finetuned_model_name,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    )
    model.eval()

    # torch.manual_seed(123)

    # prompt = """Below is an instruction that describes a task. Write a response
    # that appropriately completes the request.

    # ### Instruction:
    # Convert the active sentence to passive: 'The chef cooks the meal every day.'
    # """

    # token_ids = generate(
    #     model=model,
    #     idx=text_to_token_ids(prompt, tokenizer),
    #     max_new_tokens=35,
    #     context_size=BASE_CONFIG["context_length"],
    #     eos_id=50256
    # )

    # response = token_ids_to_text(token_ids, tokenizer)
    # print(response)
