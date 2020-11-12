# Animal Id ML Toolkit
This is collection of python code for developing image based animal id systems.

## What's all in here?

Main functionalities
* Interacting with social media API _data_tools.actions_  
* Do object detection and crop image based on detection results. _data_tools.processing_ 
* Simple interactive python utility for manually identifying images _data_tools.processing_   
* Metric for measuring recall of identity embeddings _model_tools.encoder.proximity_hits_  
* Keras.Callback based embedding visualization tool _model_tools.encoder_
* Keras.Sequence based data generators for images and embeddings _model_tools.generators_  

## What's to come?

* Better documentation obviously  
* global verbosity control  
* Cython compilation for library  
* Iterative data cleaning / continuous training cycle  
