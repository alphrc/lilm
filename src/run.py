#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import dotenv
import json
import random
from loguru import logger
from termcolor import colored

import transformers
import mlx
from mlx_lm import load, generate

from data import Thread, get_headers, get


if not dotenv.load_dotenv():
    print("No .env file found, please rename the .env.template file to .env and fill in the required fields")

logger.remove()


class LiLM:
    def __init__(self, model_name_or_path: str):
        transformers.logging.set_verbosity_error()

        self.model_name = model_name_or_path.removeprefix('model/')

        print(colored(f"Loading model {self.model_name}...", "light_yellow"))

        with open(f"{model_name_or_path}/tokenizer_config.json", "r") as f:
            tokenizer_config = json.load(f)
        self.model, self.tokenizer = load(
            model_name_or_path, 
            adapter_path=None, 
            tokenizer_config=tokenizer_config
        )

        self.max_tokens = 128
        self.temp = 0.7
        self.top_p = 0.7

    @property
    def system_message(self) -> str:
        return (
            f"模擬香港連登討論區的用戶來回覆以下貼文，該討論區的用戶以其{os.getenv('SYSTEM_MESSAGE_ADJECTIVE')}的語言風格著稱。"
            f"你在回應貼文時，請保持諷刺、幽默和創意，以模仿連登用戶的交流方式和語言風格。"
            f"注意使用地道和符合文法的粵語表達，注意文句通順，並合理地穿插俚語、粗口和流行網絡表情，來增強語言的真實性和吸引力。"
            f"{os.getenv('SYSTEM_MESSAGE_RESTRICTION')}"
        )
    
    def process_assets(self, response: str) -> str:
        response = response.replace('<img src="/assets/faces/normal/', '<img src="/assets/faces/')
        response = response.replace('<img src="/assets/faces/', '<img src="https://lihkg.com/assets/faces/normal/')
        return response
    
    def print_section(self, title: str, content: str, color: str) -> None:
        print('\n' + colored(title + ":", color))
        print(content)

    def generate(self, user_message: str, save_path: str = None) -> None:
        self.print_section("POST", user_message.rstrip('\n'), 'green')

        mlx.core.random.seed(random.randint(0, 42))

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content":  user_message}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        response = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt = prompt,
            max_tokens=self.max_tokens,
            temp=self.temp,
            top_p=self.top_p,
            verbose=False,
        )
        response = self.process_assets(response)

        self.print_section('REPLY', response, 'blue')

        if save_path:
            with open(save_path, 'a') as f:
                f.write(json.dumps({'model_name': self.model_name, "post": user_message, "reply": response}, ensure_ascii=False) + '\n')


def get_prompt_from_thread(thread: dict) -> str:
    return (
        f"貼文類別：「{thread['category']}」\n"
        f"貼文標題：「{thread['title']}」\n"
        f"貼文內容：「{thread['content']}」\n"
    )


def get_thread_from_id(thread_id: int) -> dict:
    print(colored(f"Getting thread {thread_id}", "blue"))

    url = f"https://lihkg.com/api_v2/thread/{thread_id}/page/1?order=reply_time"
    headers = get_headers(referer = f'https://lihkg.com/thread/{thread_id}/page/1')

    response = get(url=url, headers=headers, use_proxy=False)

    thread = Thread(response['response'])

    return {
        "category": thread.category.name,
        "title": thread.title,
        "content": thread.item_data[0].msg
    }


def get_thread_id_from_url(url: str) -> dict:
    try:
        if "thread/" in url:
            return url.split("thread/")[1].split("/")[0]
        elif "lih.kg" in url:
            return url.split("lih.kg/")[1]
    except (ValueError, IndexError):
        print(colored("Invalid URL format.", "red"))
        return None


def interactive(pretrained_model_name_or_path: str) -> None:
    lilm = LiLM(pretrained_model_name_or_path)

    while True:
        try:
            user_input = input(colored("\n" + "Enter thread URL or ID: ", 'light_yellow'))
            thread_id = int(user_input) if user_input.isdigit() else get_thread_id_from_url(user_input)
            thread = get_thread_from_id(thread_id)
            prompt = get_prompt_from_thread(thread)
            print(colored(f"Generating response for thread...", "blue"))
            lilm.generate(prompt, save_path='results.jsonl')
        except KeyboardInterrupt:
            quit()
        except Exception as e:
            print(f"{e}")


def test(pretrained_model_name_or_path: str) -> None:
    lilm = LiLM(pretrained_model_name_or_path)

    with open("dataset/chat/test.jsonl", "r") as f:
        for line in f:
            prompt = json.loads(line)['messages'][1]['content']
            lilm.generate(prompt, save_path='results.jsonl')


def demo(pretrained_model_name_or_path: str) -> None:
    lilm = LiLM(pretrained_model_name_or_path)

    with open('demo/threads.jsonl', 'r') as f:
        for line in f:
            thread = json.loads(line)
            lilm.generate(get_prompt_from_thread(thread), save_path='demo/results.jsonl')


def introduction():
    print(colored("Welcome to LiLM by alphrc!\n", 'green'))
    print(f"* This language model generates responses in the style of the Hong Kong LIHKG forum.")
    print(f"* For more information, updates, or to contribute to the project, visit: {colored('https://github.com/alphrc', 'cyan')}")
    print(f"* Press CTRL+C to quit at any time.")
    print()


def select_mode():
    print(colored('Available modes:', 'light_yellow'))
    print(colored('(1) Demo', 'white'))
    print(colored('(2) Test', 'white'))
    print(colored('(3) Interactive', 'white'))
    while True:
        mode_id = input(colored('Enter mode ID: ', 'light_yellow'))
        if mode_id.isdigit() and int(mode_id) in range(1, 4):
            break
        print(colored("Invalid mode ID. Please enter a valid mode ID.", 'red'))
    print()
    return mode_id


def select_model() -> str:
    print(colored("Available models:", 'light_yellow'))
    if len(os.listdir("model")) == 0:
        print(colored("No models found in the model directory. Please fuse a model and try again.", 'red'))
        quit()
    for i, model in enumerate(sorted([model for model in os.listdir("model") if model != ".gitkeep"])):
        print(colored(f"({i+1}) {model}", 'white'))
    while True:
        model_id = input(colored(f'Enter model ID: ', 'light_yellow'))
        if model_id.isdigit() and int(model_id) in range(1, len(os.listdir("model")) + 1):
            break
        print(colored("Invalid model ID. Please enter a valid model ID.", 'red'))
    print()
    return "model/" + sorted(os.listdir("model"))[int(model_id) - 1]


def run(mode: str, model: str) -> None:
    {"1": demo, "2": test, "3": interactive}[mode](model)


if __name__ == "__main__":
    introduction()
    mode = select_mode()
    model = select_model()
    run(mode, model)
