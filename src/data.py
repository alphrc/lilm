from __future__ import annotations

import os
import dotenv
import csv
import json
import requests
from typing import Literal
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed


if not dotenv.load_dotenv():
    print("No .env file found, please rename the .env.template file to .env and fill in the required fields")


class Thread:
    allow_create_child_thread: bool
    cat_id: int
    category: Category
    create_time: int
    dislike_count: int
    display_vote: bool
    first_post_id: str
    is_adu: bool
    is_bookmarked: bool
    is_hot: bool
    is_replied: bool
    item_data: list[Post]
    last_reply_time: int
    last_reply_user_id: int
    like_count: int
    max_reply: int
    max_reply_dislike_count: int
    max_reply_like_count: int
    no_of_reply: int
    no_of_uni_user_reply: int
    page: str
    parent_thread_id: str
    remark: Remark
    reply_dislike_count: int
    reply_like_count: int
    status: int
    sub_cat_id: int
    thread_id: str
    title: str
    total_page: int
    user: User
    user_gender: str
    user_id: str
    user_nickname: str
    vote_status: str

    def __init__(self, data: dict):
        if data is not None:
            for key, value in data.items():
                if key == 'category':
                    setattr(self, key, Category(value))
                elif key == 'item_data':
                    setattr(self, key, [Post(item) for item in value])
                elif key == 'remark':
                    setattr(self, key, Remark(value))
                elif key == 'user':
                    setattr(self, key, User(value))
                else:
                    setattr(self, key, value)

    def rank_by(self, order: Literal['reply_time', 'score']) -> None:
        if order == 'reply_time':
            self.item_data.sort(key=lambda x: x.reply_time)
        elif order == 'score':
            self.item_data.sort(key=lambda x: x.vote_score)

    def get_valid_entries(self) -> list[dict]:
        thread_entry = {
            'cat_id': self.cat_id,
            'category': self.category.name,
            'thread_id': self.thread_id,
            'title': self.title,
            'author': self.user_nickname,
            'content': self.item_data[0].msg,
            'thread_vote_score': self.item_data[0].vote_score,
            'thread_reaction_count': self.item_data[0].like_count + self.item_data[0].dislike_count,
        }
        entries = []
        for post in self.item_data:
            if post.is_valid():
                entries.append(thread_entry | post.as_entry())
        return entries
    
    def is_valid(self) -> bool:
        return len(self.item_data) > 1 and not self.item_data[0].user_nickname == self.item_data[1].user_nickname

    def save(self) -> None:
        if not self.is_valid():
            return
        
        entries = self.get_valid_entries()
        filename = f"./data/lihkg/{self.thread_id}.csv"

        if entries:
            keys = entries[0].keys()
            with open(filename, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=keys)
                writer.writeheader()
                writer.writerows(entries)

        logger.success(f"Saved {len(entries)} entries from thread {self.thread_id}")

    def __add__(self, other: Thread) -> Thread:
        self.item_data += other.item_data
        return self


class Category:
    cat_id: str
    name: str
    postable: bool

    def __init__(self, data: dict):
        if data is not None:
            for key, value in data.items():
                setattr(self, key, value)


class Post:
    dislike_count: int
    display_vote: bool
    is_minimized_keywords: bool
    like_count: bool
    low_quality: bool
    msg: str
    msg_num: int
    no_of_quote: int
    page: int
    post_id:str
    quote: Post
    quote_post_id: str
    remark: str
    reply_time: int
    status: int
    thread_id: str
    user: User
    user_gender: str
    user_nickname: str
    vote_score: int

    def __init__(self, data: dict):
        if data is not None:
            for key, value in data.items():
                if key == 'user':
                    setattr(self, key, User(value))
                if key == 'msg':
                    setattr(self, key, value.replace('<br />', ''))
                else:
                    setattr(self, key, value)
    
    def is_hot(self) -> bool:
        return self.like_count + self.dislike_count >= 10
    
    def contains_external_link(self) -> bool:
        return 'http://' in self.msg or 'https://' in self.msg
    
    def is_valid(self) -> bool:
        return self.is_hot() and not self.contains_external_link()

    def as_entry(self) -> dict:
        return {
            'post_id': self.post_id,
            'msg_num': self.msg_num,
            'replier': self.user_nickname,
            'msg': self.msg,
            'like_count': self.like_count,
            'dislike_count': self.dislike_count,
            'vote_score': self.vote_score,
            'reaction_count': self.like_count + self.dislike_count,
            'quote_post_id': self.quote_post_id if hasattr(self, 'quote') else None,
        }


class Remark:
    last_reply_count: int
    no_of_uni_not_push_post: int

    def __init__(self, data: dict):
        if data is not None:
            for key, value in data.items():
                setattr(self, key, value)


class User:
    gender: str
    is_blocked: bool
    is_disappear: bool
    is_following: bool
    is_newbie: bool
    level: int
    level_name: str
    nickname: str
    status: int
    user_id: str

    def __init__(self, data: dict):
        if data is not None:
            for key, value in data.items():
                setattr(self, key, value)


def get_headers(referer: str) -> dict[str, str]:
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Host': 'lihkg.com',
        'Referer': referer,
        'User-Agent': os.getenv("HEADERS_USER_AGENT"),
    }
    return headers


def get_api_key(disabled_key: str = None) -> str:
    global __api_key_cache

    if '__api_key_cache' not in globals():
        __api_key_cache = None

    if not disabled_key and __api_key_cache:
        return __api_key_cache

    with open('api_keys.json') as f:
        keys = json.load(f)
    if disabled_key and keys[disabled_key]['available']:
        keys[disabled_key]['available'] = False
        with open('api_keys.json', 'w') as f:
            json.dump(keys, f, indent=4)
    for key, value in keys.items():
        if value['available']:
            __api_key_cache = key
            return key
    return None


def get_proxies(disabled_key: str = None) -> dict[str, str]:
    api_key = get_api_key(disabled_key)
    return api_key and {
        "api_key": api_key,
        "http": os.getenv("PROXY_HTTP").format(api_key=api_key),
        "https": os.getenv("PROXY_HTTPS").format(api_key=api_key),
    }


def get(url: str, headers: dict, use_proxy: bool) -> dict:
    proxies = get_proxies() if use_proxy else None
    verify = os.getenv("VERIFY")

    while proxies:
        try:
            response = requests.get(url=url, headers=headers, proxies=proxies, verify=verify).json()
            return response
        except requests.exceptions.ProxyError:
            logger.warning(f"API key {proxies['api_key']} is disabled. Trying again with a new API key")
            proxies = get_proxies(disabled_key=proxies['api_key'])
        except Exception as e:
            logger.error(f"API Key: {proxies['api_key']} | URL: {url} | Error: {e}")
    else:
        try:
            return requests.get(url=url, headers=headers).json()
        except Exception as e:
            logger.error(f"API Key: {proxies['api_key']} | URL: {url} | Error: {e}")


def get_thread_page(thread_id: int, page: int, use_proxy: bool) -> Thread:
    url = f"https://lihkg.com/api_v2/thread/{thread_id}/page/{page}?order=reply_time"
    headers = get_headers(referer = f'https://lihkg.com/thread/{thread_id}/page/1')
    response = get(url, headers, use_proxy)
    thread = Thread(response['response'])
    return thread


def scrape_thread(thread_id: int, use_proxy: bool) -> None:
    logger.info(f"Getting thread {thread_id}")

    thread = get_thread_page(thread_id, 1, use_proxy)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_thread_page, thread_id, page) for page in range(2, thread.total_page + 1)]
        for future in as_completed(futures):
            thread += future.result()

    logger.info(f"Got all {thread.total_page} pages of thread {thread_id} with {len(thread.item_data)} posts")

    thread.rank_by('reply_time')
    thread.save()
    del thread


def scrape_thread_range(start_id: int, end_id: int, use_proxy: bool) -> None:
    logger.info(f"Scraping threads from {start_id} to {end_id}")

    with ThreadPoolExecutor() as executor:
        for thread_id in range(start_id, end_id + 1):
            executor.submit(scrape_thread, thread_id, use_proxy)


def main():
    scrape_thread_range(start_id=3669900, end_id=3733123, use_proxy=True)


if __name__ == "__main__":
    main()
