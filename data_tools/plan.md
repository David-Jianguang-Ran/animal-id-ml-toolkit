# Data

### loose thoughts
* should there be a minimal image count per animal?
* should internal_id be a float or uuid?  
float will be easier for ML computation later on but uuid is so easy to deal with here??  
*I guess we can just use instagram user id and add a float to it for now*
* what about multiple animals sharing the same photos? __Can a dog be two dogs!?__
* I wonder if the classes (individual animal in our case) will be too imbalanced? 
If i have to guess the number of images per account is distributed logarithmically  
* How do I discover accounts ?
* __DIRECTORY STRUCTURE NEEDS CLEAN UP!!__
* we really don't like accounts that have multiple animals in them,  
I'm thinking filtering username by keyword 'and', hopefully this excludes accounts with two animals

* omg found the api endpoint for media query
    ```
    https://www.instagram.com/graphql/query/?query_id=17888483320059182&variables={%22id%22:%2238436887401%22,%22first%22:20,%22after%22:null}
    # somehow query_id for media only query per user is static!!
  
    e.FEED_QUERY_ID = "e3ae866f8b31b11595884f0c509f3ec5",
    e.FEED_PAGE_EXTRAS_QUERY_ID = "5b71e1f0e14e2ce11199f10da87357ec",
    e.SUGGESTED_USER_COUNT_QUERY_ID = "09bb2c060bd093088daac1906a1f1d53",
    e.SUL_QUERY_ID = "ed2e3ff5ae8b96717476b62ef06ed8cc",
  
    56a7068fea504063273cc2120ffd54f3 owner to media, username using id
    f2405b236d85e8296cf30347c9f08c2a same as above
    d4d88dc1500312af6f937f7b804c68c3 id to username through reel
  
    ```
* looks using the accessibility_captions isn't very reliable and it makes querying user images difficult,  
should i just get all image from user and run them through open computer vision models? I need the model to be retained from 

#### Filtering images by content

The accessibility_captions added by instagram are very useful in reducing irrelevant image downloads, I'd like to keep it  
__HOWEVER__ using the owner to media query doesn't not include accessibility tags  
The only query that has accessibility tags are shortcode media or username url with ?__a=1, and I can't do pagination on it

__Maybe__  
Maybe we can collect all record of images from user_id,  
when we need to download images, do a shortcode media query and only download if accessibility tags are met  
This has the advantage of being easy to parallelize across many difference cloud machines  
so we don't hit query limit from one machine  
This is probably worth going into in a later section


#### Account discovery

* __by hand__  
yeah you heard me, by hand. we had to wrangle up 50 usernames by hand for testing

* __by hashtag__  
potentially useful hashtags:  
dogmom, doglover, 

### description
150 x 150 rgb cat and dog images downloaded from public instagram accounts  
the labels will track the identity of each animal,  
the goal is to train a encoder for individual animal recognition  

### storage structure 
* labels.csv  
one row per image,   
columns:  
    * animal_id  
    * image_serial  <- actual path to file on disk is data/subject_id/image_serial
    * source_url
 
* data directory
images are stored here under data/subject_id/image_serial

# Scripts

