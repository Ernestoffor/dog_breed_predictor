import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.utils import np_utils
from numpy import loadtxt
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
#from keras.applications.resnet50 import ResNet50
import numpy as np
from keras.preprocessing import image                  
from tqdm import tqdm
import cv2
from keras import backend as K
import io
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from extract_bottleneck_features import *
import matplotlib.pyplot as plt    
import pandas as pd    
import flask  
from PIL import Image 

def load_resnet50_model(model_path):
	"""
	A procedure to load an already trained Resnet50 model for this app.
	This model was trained using transfer learning.
	INPUT:
		model_path ->str: absolute path to the trained model
	OUTPUT:
		Resnet50_model -> array: the trained model
	"""
	Resnet50_model = load_model(model_path)
	#print(Resnet50_model.summary())
	return Resnet50_model
Resnet50_model =  load_model('../saved_models/weights.best.Resnet50.hdf5')

ResNet50_model_ = ResNet50(weights='imagenet')
#graph2 = tf.get_default_graph()

ResNet50__ = ResNet50(weights='imagenet', include_top=False, pooling="avg")

def path_to_tensor(img_path):
	# loads RGB image as PIL.Image.Image type
	img = image.load_img(img_path, target_size=(224, 224))
	# convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
	x = image.img_to_array(img)
	# convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
	

	return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
	list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
	return np.vstack(list_of_tensors)

def pretrained_Resnet50_model():
	"""
	A pre-trained model for over 1000 images
	"""
	ResNet50_model_ = ResNet50(weights='imagenet', include_top=False, pooling="avg")
	return ResNet50_model_

#ResNet50_model_ = pretrained_Resnet50_model()

def prepare_image(img_path, target= (224,224)):

	# if the image mode is not RGB, convert it
	image = flask.request.files[img_path].read()
	image = Image.open(io.BytesIO(image))
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

def predict_probab(img_path):
	image = prepare_image(img_path)
	# classify the input image and then initialize the list
	# of predictions to return 
	preds = Resnet50_model.predict(image)
	results = imagenet_utils.decode_predictions(preds)
	predictions = []
	for (imagenetID, label, prob) in results[0]:
		# Convert the probability to % 
		r = {"label": label.replace('_', ' ').title(), "probability": round(float(prob)*100, 3)}
		predictions.append(r)
	df = pd.DataFrame(data=np.zeros((5, 2)),
					  columns=['Breeds', 'Confidence Level'],
					  index=np.linspace(1, 5, 5, dtype=int))

	for idx, p in enumerate(predictions):
		i = 0
		link = 'https://en.wikipedia.org/wiki/' + \
			p['label'].lower().replace(' ', '_')
		df.iloc[idx,
				0] = f'<a href="{link}" target="_blank">{p[0].title()}</a>'
		df.iloc[idx, 1] = p['probability']
		if i ==2:
			break
	#df_html = df.to_html(escape=False)

	return df


def ResNet50_predict_labels(img_path):
	# returns prediction vector for image located at img_path
	img = preprocess_input(path_to_tensor(img_path))
	predictions = ''
	#with graph2.as_default():
	predictions = ResNet50_model_.predict(img)

	return np.argmax(predictions)


def dog_detector(img_path):
	prediction = ResNet50_predict_labels(img_path)
	return ((prediction <= 268) & (prediction >= 151)) 


def face_detector(img_path):
	# extract pre-trained face detector
	face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
	img = cv2.imread(img_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray)
	return len(faces) > 0

# list of dog_names extracted from the attached jupyter notebook

dog_names = ['ages/train/001.Affenpinscher', 'ages/train/002.Afghan_hound', 'ages/train/003.Airedale_terrier', 'ages/train/004.Akita', 'ages/train/005.Alaskan_malamute', 'ages/train/006.American_eskimo_dog', 'ages/train/007.American_foxhound', 'ages/train/008.American_staffordshire_terrier', 'ages/train/009.American_water_spaniel', 'ages/train/010.Anatolian_shepherd_dog', 'ages/train/011.Australian_cattle_dog', 'ages/train/012.Australian_shepherd', 'ages/train/013.Australian_terrier', 'ages/train/014.Basenji', 'ages/train/015.Basset_hound', 'ages/train/016.Beagle', 'ages/train/017.Bearded_collie', 'ages/train/018.Beauceron', 'ages/train/019.Bedlington_terrier', 'ages/train/020.Belgian_malinois', 'ages/train/021.Belgian_sheepdog', 'ages/train/022.Belgian_tervuren', 'ages/train/023.Bernese_mountain_dog', 'ages/train/024.Bichon_frise', 'ages/train/025.Black_and_tan_coonhound', 'ages/train/026.Black_russian_terrier', 'ages/train/027.Bloodhound', 'ages/train/028.Bluetick_coonhound', 'ages/train/029.Border_collie', 'ages/train/030.Border_terrier', 'ages/train/031.Borzoi', 'ages/train/032.Boston_terrier', 'ages/train/033.Bouvier_des_flandres', 'ages/train/034.Boxer', 'ages/train/035.Boykin_spaniel', 'ages/train/036.Briard', 'ages/train/037.Brittany', 'ages/train/038.Brussels_griffon', 'ages/train/039.Bull_terrier', 'ages/train/040.Bulldog', 'ages/train/041.Bullmastiff', 'ages/train/042.Cairn_terrier', 'ages/train/043.Canaan_dog', 'ages/train/044.Cane_corso', 'ages/train/045.Cardigan_welsh_corgi', 'ages/train/046.Cavalier_king_charles_spaniel', 'ages/train/047.Chesapeake_bay_retriever', 'ages/train/048.Chihuahua', 'ages/train/049.Chinese_crested', 'ages/train/050.Chinese_shar-pei', 'ages/train/051.Chow_chow', 'ages/train/052.Clumber_spaniel', 'ages/train/053.Cocker_spaniel', 'ages/train/054.Collie', 'ages/train/055.Curly-coated_retriever', 'ages/train/056.Dachshund', 'ages/train/057.Dalmatian', 'ages/train/058.Dandie_dinmont_terrier', 'ages/train/059.Doberman_pinscher', 'ages/train/060.Dogue_de_bordeaux', 'ages/train/061.English_cocker_spaniel', 'ages/train/062.English_setter', 'ages/train/063.English_springer_spaniel', 'ages/train/064.English_toy_spaniel', 'ages/train/065.Entlebucher_mountain_dog', 'ages/train/066.Field_spaniel', 'ages/train/067.Finnish_spitz', 'ages/train/068.Flat-coated_retriever', 'ages/train/069.French_bulldog', 'ages/train/070.German_pinscher', 'ages/train/071.German_shepherd_dog', 'ages/train/072.German_shorthaired_pointer', 'ages/train/073.German_wirehaired_pointer', 'ages/train/074.Giant_schnauzer', 'ages/train/075.Glen_of_imaal_terrier', 'ages/train/076.Golden_retriever', 'ages/train/077.Gordon_setter', 'ages/train/078.Great_dane', 'ages/train/079.Great_pyrenees', 'ages/train/080.Greater_swiss_mountain_dog', 'ages/train/081.Greyhound', 'ages/train/082.Havanese', 'ages/train/083.Ibizan_hound', 'ages/train/084.Icelandic_sheepdog', 'ages/train/085.Irish_red_and_white_setter', 'ages/train/086.Irish_setter', 'ages/train/087.Irish_terrier', 'ages/train/088.Irish_water_spaniel', 'ages/train/089.Irish_wolfhound', 'ages/train/090.Italian_greyhound', 'ages/train/091.Japanese_chin', 'ages/train/092.Keeshond', 'ages/train/093.Kerry_blue_terrier', 'ages/train/094.Komondor', 'ages/train/095.Kuvasz', 'ages/train/096.Labrador_retriever', 'ages/train/097.Lakeland_terrier', 'ages/train/098.Leonberger', 'ages/train/099.Lhasa_apso', 'ages/train/100.Lowchen', 'ages/train/101.Maltese', 'ages/train/102.Manchester_terrier', 'ages/train/103.Mastiff', 'ages/train/104.Miniature_schnauzer', 'ages/train/105.Neapolitan_mastiff', 'ages/train/106.Newfoundland', 'ages/train/107.Norfolk_terrier', 'ages/train/108.Norwegian_buhund', 'ages/train/109.Norwegian_elkhound', 'ages/train/110.Norwegian_lundehund', 'ages/train/111.Norwich_terrier', 'ages/train/112.Nova_scotia_duck_tolling_retriever', 'ages/train/113.Old_english_sheepdog', 'ages/train/114.Otterhound', 'ages/train/115.Papillon', 'ages/train/116.Parson_russell_terrier', 'ages/train/117.Pekingese', 'ages/train/118.Pembroke_welsh_corgi', 'ages/train/119.Petit_basset_griffon_vendeen', 'ages/train/120.Pharaoh_hound', 'ages/train/121.Plott', 'ages/train/122.Pointer', 'ages/train/123.Pomeranian', 'ages/train/124.Poodle', 'ages/train/125.Portuguese_water_dog', 'ages/train/126.Saint_bernard', 'ages/train/127.Silky_terrier', 'ages/train/128.Smooth_fox_terrier', 'ages/train/129.Tibetan_mastiff', 'ages/train/130.Welsh_springer_spaniel', 'ages/train/131.Wirehaired_pointing_griffon', 'ages/train/132.Xoloitzcuintli', 'ages/train/133.Yorkshire_terrier']

def extract_Resnet50(tensor):
	#K.clear_session()
	output = ''
	#with graph3.as_default():
	output = ResNet50__.predict(preprocess_input(tensor))
	#K.clear_session()
	return output

def predict_breed(img_path):
	# extract bottleneck features
	bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
	# obtain predicted vector
	bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
	bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
	#print(bottleneck_feature.shape)
	predicted_vector = ''
	predicted_vector = Resnet50_model.predict(bottleneck_feature)
	# return dog breed that is predicted by the model
	return dog_names[np.argmax(predicted_vector)]


def predict_dog_breed(image_filepath):
	if dog_detector(image_filepath):
		pred = predict_breed(image_filepath).split('.')[1]
		# Replace '_' with ' '
		pred = pred.replace('_', ' ')
		# capitalize each word
		link = 'https://en.wikipedia.org/wiki/' + \
			pred.lower().replace(' ', '_')
		pred = pred.title()
		text1 = 'Your Image Selection is a Dog.'
		text2 =  'The Breed of the Dog is: '
		print(pred)
		return (pred, text1, text2, link)
	elif face_detector(image_filepath):
		pred = predict_breed(image_filepath).split('.')[1]
		# Replace '_' with ' '
		pred = pred.replace('_', ' ')
		# capitalize each word
		link = 'https://en.wikipedia.org/wiki/' + \
			pred.lower().replace(' ', '_')
		pred = pred.title()
		text1 = 'Your Image Selection is a Human Being.'
		text2 =  'The corresponding Dog Breed of This Human is'
		
		return (pred, text1, text2, link)
	else:

		link = "/"
		pred = "Upload another image."
		text1 = "Ups!!! Check your selection. "
		text2 = "This is neither a dog nor a human image."
		return (pred, text1, text2, link)
	
	