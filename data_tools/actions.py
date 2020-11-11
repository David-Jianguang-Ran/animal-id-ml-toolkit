import pandas as pd
import numpy as np
import time
import requests
import pickle
import uuid

from PIL import Image

from .queries import hashtag_to_media_query, username_to_media_query, user_id_to_media_query, id_to_username_query
from .settings import *

from .utils import string_contains, load_and_combine_dataset


def find_id_by_hashtag(hashtag, user_count):
    ids = set()
    tag_start_time = time.time()
    print(f"discovering by tag {hashtag}")
    starting_lead_count = len(ids)
    end_cursor = None
    request_count = 0
    while len(ids) - starting_lead_count < user_count and request_count < MAX_REQUEST_PER_TAG:
        hashtag_dict = hashtag_to_media_query(hashtag, end_cursor)
        request_count += 1
        # stop iteration if no next page
        if hashtag_dict is None:
            break
        # record owner_id_found
        for each_media in hashtag_dict['edge_hashtag_to_media']['edges']:
            ids.add(each_media['node']["owner"]['id'])

        end_cursor = hashtag_dict['edge_hashtag_to_media']["page_info"]["end_cursor"]

        print(f"request# {request_count}, found {len(ids) - starting_lead_count}/{user_count}")

    print(f"found <{len(ids)}> users for tag <{hashtag}, time spent {time.time() - tag_start_time:.3} seconds, requests made {request_count}>")
    return ids


def get_media_url_from_user(user: dict) -> dict:
    # this checks timeline media and returns all media record in a dict of arrays,
    # basically a table

    # first let's make our internal animal ids id
    # for example:
    # a cat's id for user 1000 when we are looking for dog, cat
    # would be 1000.5
    animal_ids = {each_name: float(user['id']) + i / len(INCLUDED_SUBJECTS) for i, each_name in enumerate(INCLUDED_SUBJECTS)}

    record = {
        "id": [],
        "src": []
    }

    # iterate through all timeline media
    for each_media in user["edge_owner_to_timeline_media"]["edges"]:

        data = each_media['node']

        # ignore video media
        if data["is_video"]:
            continue

        if data['accessibility_caption'] is not None:
            contains_subject = string_contains(data['accessibility_caption'], INCLUDED_SUBJECTS)
        else:
            continue

        # ignore photos that dont contain subjects or too many subjects
        if len(contains_subject) != 1:
            continue

        # save the thumbnail of desired size to record
        try:
            thumb_nail = data["thumbnail_resources"][2]
        except (KeyError, IndexError):
            continue

        if thumb_nail["config_width"] == thumb_nail["config_height"] == IMAGE_SIZE:
            # append this media to record
            record['id'].append(animal_ids[contains_subject[0]])
            record['src'].append(thumb_nail["src"])

    return record


def discover_images(ids, file_path):
    # get a set of shortcodes for each user id
    for each_userid in ids:

        shortcodes = []
        user_start_time = time.time()
        print(f"discoverting images by userid{each_userid}")

        # keep requesting until image limit or request limit
        starting_code_count = len(shortcodes)
        end_cursor = None
        request_count = 0
        while len(shortcodes) - starting_code_count < MAX_IMAGE_PER_USER and request_count < MAX_REQUEST_PER_USER:
            media_data = user_id_to_media_query(each_userid, first=75, __end_cursor=end_cursor)
            request_count += 1

            # bail out if not 200
            if media_data is None:
                break

            # get shortcode
            for each_node in media_data["edges"]:
                shortcodes.append(each_node['node']['shortcode'])

        # combine data with existing record
        try:
            existing = pd.read_csv(file_path)
        except FileNotFoundError:
            existing = pd.DataFrame()
        existing = existing.append(pd.DataFrame(data=shortcodes, columns=["shortcode"]),ignore_index=True)
        # save to disk
        existing.to_csv(file_path, index=False)


def download_username_by_hashtags(hashtags: [], users_per_tag=300, output_name=True, found_prev=None):
    # we name our dataset with a truncated uuid4
    identifier = str(uuid.uuid4())[:8]
    # record start time
    start_time = time.time()

    # iterate through tags and save unique user_ids
    if not found_prev:
        found = set()
    else:
        found = found_prev

    for each_tag in hashtags:
        found_ids = find_id_by_hashtag(each_tag, users_per_tag)

        # saving just user_id is super easy
        if not output_name:
            found = found.union(found_ids)
        else:
            found_names = {id_to_username_query(each_id)for each_id in found_ids}
            found = found.union(found_names)

            # cache found username in a pickle file
            with open(f"./dev_data/temp/found_names_{each_tag}.pickle", "wb") as file:
                pickle.dump(found_names, file)

        time.sleep(BACK_OFF_TIMER)

    # save file to disk
    frame = pd.DataFrame(data=list(found), columns=["username" if output_name else "user_id"])
    frame.to_csv(f"./dev_data/auto_targeted_accounts_{identifier}.csv")
    print(f"found {frame.shape[0]} users, time spent {time.time() - start_time:.3} seconds, file id:{identifier}")


def download_dataset_from_usernames(usernames, chunk_num):
    labels = {
        "animal_id": [],
        "image_id": [],
    }
    __images = pd.DataFrame()
    id_suffix = ID_SUFFIX

    for each_name in usernames:
        # ignore accounts that contain 'and' in name,
        # they are usually multiple animal accounts
        if not isinstance(each_name, str):
            continue
        elif string_contains(each_name, ["and"]):
            continue
        media_data = username_to_media_query(each_name)
        to_download = {}

        if media_data is None:
            continue

        # collect urls and labels for download
        for media in media_data:
            data = media['node']

            # filter by accessibility tags
            if data['accessibility_caption'] is not None:
                contains_subject = string_contains(data['accessibility_caption'], INCLUDED_SUBJECTS)
            else:
                continue

            # ignore photos that dont contain subjects or too many subjects
            if len(contains_subject) != 1:
                continue

            # check for thumbnail existance and download
            try:
                thumb_nail = data["thumbnail_resources"][IMAGE_SIZE[0]]
            except (KeyError, IndexError):
                continue

            if thumb_nail["config_width"] == thumb_nail["config_height"] == IMAGE_SIZE[1]:
                to_download[thumb_nail['src']] = (float(data['owner']['id']) + id_suffix[contains_subject[0]], data["shortcode"])

        # only download when min photo count is met
        if len(to_download) > 3:
            for url, d in to_download.items():
                animal_id, shortcode = d
                # if there is any sort of connection error, ignore and continue
                try:
                    image = Image.open(requests.get(url, stream=True).raw)
                except:
                    print(f"connection failed")
                    continue

                image = np.array(image).reshape([-1])

                if image.shape[0] == IMAGE_SIZE[1] * IMAGE_SIZE[1] * 3:
                    # save image and label
                    labels['image_id'].append(shortcode)
                    labels['animal_id'].append(animal_id)

                    __images[shortcode] = np.array(image).reshape([-1])
        else:
            time.sleep(DELAY_PER_REQUEST)

    # save data
    __images.to_parquet(f"./data/dataset_{chunk_num}.parquet", engine="pyarrow")
    labels = pd.DataFrame(labels)
    labels.to_csv(f"./data/dataset_labels_{chunk_num}.csv")
