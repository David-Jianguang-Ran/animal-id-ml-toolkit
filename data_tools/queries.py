import requests
import json
import time
import os
import shutil

from .settings import BASE_URL, BACK_OFF_TIMER, DELAY_PER_REQUEST
from .utils import fancy_print

# these functions are for querying SOCIAL MEDIA!
def user_id_to_media_query(user_id, first=20, __end_cursor=None) -> dict:
    # make a get request
    # what happens when an anonymous user queries private users? No user to timeline media is returned
    # build url
    url = BASE_URL
    params = {
        "query_hash" : "56a7068fea504063273cc2120ffd54f3",
        "variables" : json.dumps({
            "id" : int(user_id),
            "first" : first,
            "after" : __end_cursor
        })
    }

    fancy_print(f"GET {url}", verbosity=2)

    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()['data']['user']['edge_owner_to_timeline_media']
    elif response.status_code == 429:
        fancy_print(f"429 Backing off for {BACK_OFF_TIMER} seconds")
    else:
        fancy_print(response.status_code)


def username_to_media_query(__username: str):
    # can't turn pages here... >_<
    url = f"https://www.instagram.com/{__username}/?__a=1"
    fancy_print(f"GET {url}", verbosity=2)
    try:
        response = requests.get(url)
    except:
        fancy_print("connection_refused")
        return

    if response.status_code == 200:
        try:
            return response.json()["graphql"]["user"]['edge_owner_to_timeline_media']["edges"]
        except (KeyError, json.JSONDecodeError):
            fancy_print("request_failed")
            return
    elif response.status_code == 429:
        fancy_print(response.status_code)
        time.sleep(BACK_OFF_TIMER)
    else:
        fancy_print(response.status_code)


def shortcode_media_query(__shortcode):
    # make a get request
    # what happens when an anonymous user queries private users? No user to timeline media is returned
    # build url
    url = BASE_URL
    params = {
        "query_hash" : "eaffee8f3c9c089c9904a5915a898814",
        "variables" : json.dumps({
            "shortcode" : __shortcode,
        })
    }

    fancy_print(f"GET {url} {params}", verbosity=2)
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()['data']['shortcode_media']
    elif response.status_code == 429:
        fancy_print(response.status_code)
        time.sleep(BACK_OFF_TIMER)
    else:
        fancy_print(response.status_code)
        return {}


def hashtag_to_media_query(__hashtag, __end_cursor=None):
    # build url
    url = BASE_URL
    params = {
        "query_hash" : "9b498c08113f1e09617a1703c22b2f32",
        "variables" : json.dumps({
            "tag_name" : __hashtag,
            "first" : 75,
            "after" : __end_cursor
        })
    }

    fancy_print(f"GET {url}", verbosity=2)
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()["data"]["hashtag"]
    elif response.status_code == 429:
        fancy_print(response.status_code)
        time.sleep(BACK_OFF_TIMER)
    else:
        fancy_print(response.status_code)


def id_to_username_query(__id):
    time.sleep(DELAY_PER_REQUEST * 2)
    url = BASE_URL
    params = {
        "query_hash": "56a7068fea504063273cc2120ffd54f3",
        "variables" : json.dumps({
            "id": int(__id),
            "first": 2
        })
    }

    fancy_print(f"GET {url} params {params}", verbosity=2)
    try:
        response = requests.get(url, params=params)
    except:
        fancy_print("connection failed")
        time.sleep(BACK_OFF_TIMER)
        return

    if response.status_code == 200:
        try:
            return response.json()['data']['user']['edge_owner_to_timeline_media']["edges"][0]["node"]['owner']['username']
        except (KeyError, ValueError, TypeError, IndexError, json.JSONDecodeError):
            fancy_print("request failed")
    elif response.status_code == 429:
        fancy_print(response.status_code)
        time.sleep(BACK_OFF_TIMER)
    else:
        fancy_print(response.status_code)



