# Encoder Model


### loose thoughts

* pre-trained models behave just like a single conv layer  
i.e. output shape is linearly depended on input shape.  
This is useful for sliding window object detection.

* choosing the right alpha seems to be very important for triplet loss,
high alpha leads either to embedding space collapse or poor stable loss (poor performance)

* it is also worth noting optimal alpha seems to depend on sample size too.  
with more data seems to do better with lower alpha

* the number of dimensions of the embedding has doesn't seem to impact performace, (2d2e8357) 
this could be really good or bad, maybe current performance is just bad either way,  
although it would be super nice to have only a 2d embedding and be able to take advantage of existing GIS libraries 
