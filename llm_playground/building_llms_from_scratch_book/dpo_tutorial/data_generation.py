# Code here taken from the book "Building LLMs from Scratch"
import json
import random
import urllib.request
from pathlib import Path

from tqdm import tqdm


def query_model(prompt, model="llama3.1", url="http://localhost:11434/api/chat"):
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "seed": 123,
            "temperature": 0,
        },
    }

    payload = json.dumps(data).encode("utf-8")

    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def generate_model_responses(json_data):

    for i, entry in enumerate(tqdm(json_data, desc="Writing entries")):
        politeness = random.choice(["polite", "impolite"])
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"slightly rewrite the output to be more {politeness}."
            "Keep the modification minimal."
            "Only return return the generated response and nothing else."
        )
        response = query_model(prompt)

        if politeness == "polite":
            json_data[i]["chosen"] = response
            json_data[i]["rejected"] = entry["output"]
        else:
            json_data[i]["rejected"] = response
            json_data[i]["chosen"] = entry["output"]


if __name__ == "__main__":

    data_path = Path("data")

    generated_data_path = Path("generated_data")
    if not generated_data_path.exists():
        generated_data_path.mkdir()

    json_filename = "instruction-data.json"
    with open(data_path / json_filename, "r") as file:
        json_data = json.load(file)

    generate_model_responses(json_data)

    with open(
        generated_data_path / "instruction-data-with-preference.json", "w"
    ) as file:
        json.dump(json_data, file, indent=4)
