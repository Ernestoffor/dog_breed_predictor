import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SubmitField
from flask_uploads import configure_uploads, patch_request_class
from utility_resources import (image_set, save_image, get_url)
from functions import *


app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SECRET_KEY'] = '!!!!Very hard to crack!!!!'
app.config['UPLOADED_IMAGES_DEST'] = os.path.join("static", "images")
bootstrap = Bootstrap(app)
configure_uploads(app, (image_set))
patch_request_class(app, 5 * 1024 * 1024)


class ImageForm(FlaskForm):
    """
    Form for image upload
    """
    file = FileField('Check Your Dog Breed By Uploading its image',
                     validators=[FileAllowed(['jpg', 'jpeg', 'png', 'gif', 'svg'])])
    submit = SubmitField('Upload')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = ImageForm()
    url = get_url('default-dog.jpg')
    image_path = image_set.path(filename='default-dog.jpg', folder='dogs')
    pred = predict_dog_breed(image_path)
    if form.validate_on_submit() and form.file.data:
        image_path, url = save_image(image=form.file.data)
        pred  = predict_dog_breed(image_path)
    return render_template('index.html',
                           image_path=url,
                           prediction=pred,
                           form=form)

if __name__ == "__main__":
	app.run()

