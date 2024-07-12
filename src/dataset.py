import os
import re
import csv
import json
import random
import dotenv
from typing import Literal
from loguru import logger


if not dotenv.load_dotenv():
    print("No .env file found, please rename the .env.template file to .env and fill in the required fields")


def transform_to_completion(entry: dict) -> dict:
    prompt = (
        f"模擬作為香港連登討論區的一名用戶來回覆貼文，該論壇的用戶以其{os.getenv('SYSTEM_MESSAGE_ADJECTIVE')}的語言風格著稱。"
        f"你在回應貼文時，請保持諷刺、幽默和創意，以模仿連登用戶的交流方式和語言風格。"
        f"注意使用地道和符合文法的粵語表達，並合理地穿插俚語、粗口和流行網絡表情，來增強語言的真實性和吸引力。"
        f"\n\n"
        f'貼文類別：「{entry["category"]}」\n'
        f'貼文標題：「{entry["title"]}」\n'
        f'貼文內容：「{entry["content"]}」\n'
    )
    completion = entry['msg']

    return {'prompt': prompt, 'completion': completion}


def transform_to_chat(entry: dict) -> dict:
    system_message = (
        f"模擬作為香港連登討論區的一名用戶來回覆貼文，該論壇的用戶以其{os.getenv('SYSTEM_MESSAGE_ADJECTIVE')}的語言風格著稱。"
        f"你在回應貼文時，請保持諷刺、幽默和創意，以模仿連登用戶的交流方式和語言風格。"
        f"注意使用地道和符合文法的粵語表達，並合理地穿插俚語、粗口和流行網絡表情，來增強語言的真實性和吸引力。"
    )

    user_message = (
        f"貼文類別：「{entry['category']}」\n"
        f"貼文標題：「{entry['title']}」\n"
        f"貼文內容：「{entry['content']}」\n"
    )

    assistant_message = entry['msg']

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]

    return {"messages": messages}


def is_valid(entry: dict) -> bool:
    is_hot = (int(entry["reaction_count"]) >= 20)
    is_positive = (int(entry['like_count']) / (int(entry['dislike_count']) + 1) - 1 >= 1)
    is_not_quoting = not ("quote" in entry['content'])
    is_not_self_replying = (entry["author"] != entry["replier"])
    is_not_sensitive = not any([word in (entry['title'] + entry['content'] + entry['msg']) for word in os.getenv('SENSITIVE_WORDS').split(',')])
    has_no_embedding = "<pre>" not in entry['content']
    has_no_external_links = not ("http://" in entry["content"] or "https://" in entry["content"])
    not_only_asset = not bool(re.fullmatch(r'^\s*(<img src="/assets/faces/[^"]*" class="hkgmoji" />\s*)+$', entry['msg']))

    return all([
        is_hot, 
        is_positive,
        is_not_quoting,
        is_not_self_replying,
        is_not_sensitive,
        has_no_embedding,
        has_no_external_links,
        not_only_asset,
    ])


def validate_entries(entries: list[dict]) -> list[dict]:
    return [entry for entry in entries if is_valid(entry)]


def transform_entries(entries: list[dict], format: Literal['chat', 'completion']) -> list[dict]:
    if format == 'chat':
        return [transform_to_chat(entry) for entry in entries]
    elif format == 'completion':
        return [transform_to_completion(entry) for entry in entries]


def remove_long_entries(entries: list[dict], format: Literal['chat', 'completion']) -> list[dict]:
    if format == 'chat':
        return [entry for entry in entries if sum(len(message['content']) for message in entry['messages']) <= 2048]
    elif format == 'completion':
        return [entry for entry in entries if len(entry['prompt']) + len(entry['completion']) <= 2048]


def transform_all(format: Literal['chat', 'completion', 'text']):
    entries_folder = './data/lihkg/'
    entries = []

    for file in os.listdir(entries_folder):
        with open(f'{entries_folder}/{file}') as f:
            rows = csv.DictReader(f)
            entries += [row for row in rows]
    
    num_raw_entries = len(entries)

    entries = validate_entries(entries)
    entries = transform_entries(entries, format)
    entries = remove_long_entries(entries, format)

    num_transformed_entries = len(entries)

    os.makedirs(f'./dataset/{format}/', exist_ok=True)

    with open(f'./dataset/{format}/all.jsonl', 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    test_entries = entries[int(num_transformed_entries * 0.9):]
    entries = entries[:int(num_transformed_entries * 0.9)]

    random.shuffle(test_entries)
    random.shuffle(entries)

    train_entries = entries[:int(num_transformed_entries * 0.8)]
    valid_entries = entries[int(num_transformed_entries * 0.8):]

    with open(f'./dataset/{format}/train.jsonl', 'w', encoding='utf-8') as f:
        for entry in train_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    with open(f'./dataset/{format}/valid.jsonl', 'w', encoding='utf-8') as f:
        for entry in valid_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    with open(f'./dataset/{format}/test.jsonl', 'w', encoding='utf-8') as f:
        for entry in test_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logger.success(f"Transformed {num_transformed_entries}/{num_raw_entries} entries and saved to ./dataset/{format}/")


if __name__ == '__main__':
    transform_all('completion')
