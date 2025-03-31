# Software Abstract

This project involves the implementation of a deep learning model for image detection and comprehension. The model uses tensorflow and more specifically the Mirrored Strategy that allows the training to be parallelized across multiple GPUs. This was the best choice since Ray RLib was not running properly with my code. TensorFlow serves as a programming tool and a middleware that facilitiates the deep learning model development and deployment. This can be used for Medical Imaging, Physics Simulations, and Autonomous Systems. In this case it is being used for xray image classification.

# Installation

1. Download the folder to the HPCC
2. Open a terminal in the HPCC
3. Navigate to the directory
4. Unzip the Dataset.csv in Data_Entry_2017_v2020.zip
5. Type "sbatch run.sb"

This should load all of the modules and download the libraries needed for the python file to run.

# References
1. https://arxiv.org/abs/1712.02029
2. https://www.tensorflow.org/api_docs/python/tf/distributeMirroredStrategy
3. https://www.byteplus.com/en/topic/499036?title=tensorflow-mirroredstrategy-example-mastering-distributed-training
