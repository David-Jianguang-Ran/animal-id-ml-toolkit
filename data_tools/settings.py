# some helpful globals
BASE_URL = "https://www.instagram.com/graphql/query/"
INCLUDED_SUBJECTS = ["dog", "cat"]
ID_SUFFIX = {each_name: i / len(INCLUDED_SUBJECTS) for i, each_name in enumerate(INCLUDED_SUBJECTS)}

IMAGE_SIZE = (1,240)
DELAY_PER_USER = 0.1
DELAY_PER_REQUEST = 1
BACK_OFF_TIMER = 60

MAX_REQUEST_PER_USER = 10
MAX_IMAGE_PER_USER = 100
MIN_IMAGE_PER_USER = 3

MAX_REQUEST_PER_TAG = 50
USERS_PER_TAG = 1000

VERBOSITY = 1

# processing related
DBSCAN_EPS = 10.
DBSCAN_MIN_SAMP = 3

# api query_id / hashes
# 17888483320059182 id to media
# 9b498c08113f1e09617a1703c22b2f32 hashtag to media
