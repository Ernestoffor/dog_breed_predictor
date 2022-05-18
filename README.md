[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Description

This is a Convolutional Neural Networks (CNN) project. In this project,  a pipeline  is built that is used within a web app to process real-world, user-supplied images.  Given an image of a dog, the  algorithm  developed will identify an estimate of the dogs’s breed.  If supplied an image of a human, the algorithm will identify the resembling dog breed. Finally, if the uploaded image is neither a dog nor human image is selected, the app
would display the image and ask you to upload an image of a dog or human.


A sample of the web app built in the project is shown below.

!['image'](/app/static/dog_app.png)


Along with exploring state-of-the-art CNN models for classification, this project will make important design decisions about the user experience for the  app.  The goal is that project involves breaking down a series of models designed to perform various tasks in a data processing pipeline.  Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.  The imperfect solution will nonetheless create a fun user experience!

## Project Steps and Results:

A dataset comprising of **8351** total dog images with **6680** images used as the training, **835** as validation and **836** test dog images. It also contains **133 total dog categories**. Another dataset comprising **13233 total human images** was also used for developing human face detector. The following procedures were carried out:

1. A pre-trained human face detector was extracted and used in conjuction with open CV python library to develop a human face detector algorithm used in the project. The face detector algorithm detects all samples of 100 human images  fed to it as human faces but also detected 11 dog images as human faces. 

2. A pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006)  model is used to detect dogs in images. along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image. The dog detector was able to completely distinguish between dogs and human faces.

3. A Convolutional Neural Network (CNN)  was created to classify dog breeds. A first attempt was to use 4 layers with different filters and Dropouts. The accuracy of this crude model was **8.6124%**. This was too low and unacceptable and led to exploration of the use of transfer learning in building the model used for the web app.

4. The first model from transfer learning uses the the pre-trained **VGG-16** model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to the model. Only the global average pooling layer was added and a fully connected layer, where the latter contains one node for each dog category (that is 133 in all) and is equipped with a softmax activation function. This led to a test accuracy of **38.9952%**. This was a lot much more better than the former.

5. Rather than relying on the last model which was still not effective, another model from transfer learning uses the  pre-trained **VGG-19** model as a fixed feature extractor, where the last convolutional output of VGG-19 is fed as input to the model. Only the global average pooling layer was added and a fully connected layer, where the latter contains one node for each dog category (that is 133 in all) and is equipped with a softmax activation function. This led to a test accuracy of **62.6794%**. This was a lot much more better than the previous model.

6. Another model that uses transfer learning is developed using the pre-trained **Resnet-50** model as a fixed feature extractor, where the last convolutional output of Resnet-50 is fed as input to the model. Only the global average pooling layer was added and a fully connected layer, where the latter contains one node for each dog category (that is 133 in all) and is equipped with a softmax activation function. This led to a test accuracy of **73.8038%%**. This was a lot much more better than the last model and the best of all the models. This was used in building the web app.

7. With the model fully developed, the next stage is an algorithm that would use the trained model to make predictions of breeds of dogs. The algorithm would first check if an image is a dog or human. If it is a dog image, the algorithm predicts and outputs the breed of the dog. On the other hand, if it is human image, it would report it as a human image but would predict and output the breed of the most likely resemblance of the human image.  If it was neither a dog nor a human image, the algorithm would detect that it is not either human or dog.

8. Finally, a web app was developed for users' interactions with the model. This was done using a python flask framework. In the app, users can upload any image and get results as described in the algorithm of step 7. 


##  Instructions on run the web app.

1. Change directory to the /app folder.
2. Install the necessary software as follows:

``` pip install -r requirements.txt```

3. Run:

	```python run.py```

3. Go to the following endpoint in your favourite web browser:

	```http://127.0.0.1:5000/ ```

4. Interact with the web app there.




### Project Instructions of File Usage
In order to reproduce the project. Do the following. 
1. Clone the repository
```	
git clone https://github.com/Ernestoffor/dog_breed_predictor.git
cd dog-project
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog_breed_predictor/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog_breed_predictor/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog_breed_predictor/bottleneck_features`.

5.  __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

6.  **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```

7. **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name dog-project python=3.5
	source activate dog-project
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.5
	activate dog-project
	pip install -r requirements/requirements.txt
	```
	
8.  **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```
	
9. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

10.  **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog_breed_predictor` environment. 
```
python -m ipykernel install --user --name dog_breed_predictor --display-name "dog_breed_predictor"
```

11. Open the notebook.
```
jupyter notebook dog_app.ipynb
```

12. (Optional) **If you are running the project on your local machine (and not using AWS)**, before running code, change the kernel to match the dog_breed_predictor environment by using the drop-down menu (**Kernel > Change kernel > dog_breed_predictor**). 



