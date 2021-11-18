from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import pickle
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#import WSGIServer
#from gevent.pywsgi import WSGIServer
from PIL import Image
import psycopg2 #pip install psycopg2 
import psycopg2.extras
import torchvision
import torch
import numpy as np
MODEL_PATH = 'models/covid_classifier.h5'
resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.fc = torch.nn.Linear(in_features=512, out_features=3)

resnet18.load_state_dict(torch.load(MODEL_PATH))
resnet18.eval()       
class_names = ['normal', 'viral', 'covid']
#resnet18 = torchvision.models.resnet18(pretrained=True)
train_transform = torchvision.transforms.Compose([
torchvision.transforms.Resize(size=(224, 224)),
torchvision.transforms.RandomHorizontalFlip(),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform = torchvision.transforms.Compose([
torchvision.transforms.Resize(size=(224, 224)),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
def predict_image_class(image_path):
    image = Image.open(image_path).convert('RGB')
    image = test_transform(image)
    # Please note that the transform is defined already in a previous code cell
    image = image.unsqueeze(0)
    output = resnet18(image)[0]
    probabilities = torch.nn.Softmax(dim=0)(output)
    probabilities = probabilities.cpu().detach().numpy()
    predicted_class_index = np.argmax(probabilities)
    predicted_class_name = class_names[predicted_class_index]
    return probabilities, predicted_class_index, predicted_class_name


print('Model loaded. Check http://127.0.0.1:5000/')
app = Flask(__name__)
     
app.secret_key = "cairocoders-ednalan"
     
DB_HOST = "localhost"
DB_NAME = "CC_photo"
DB_USER = "bob"
DB_PASS = "admin"
     
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
  
UPLOAD_FOLDER = 'static/uploads/'
  
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
  
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
  
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
      
  
@app.route('/')
def home():
    return render_template('index.html')
  
@app.route('/', methods=['POST'])
def upload_image():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
 
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        probabilities, predicted_class_index, predicted_class_name = predict_image_class(file)
        #print('Probabilities:', probabilities)
        #print('Predicted class index:', predicted_class_index)
        #print('Predicted class name:', predicted_class_name)
        #probabilities.values.astype(int)
        #predicted_class = predicted_class_name.tolist()
        prob = float(np.mean(probabilities))
        name=request.form["name"]
        age=request.form["age"]
        city=request.form["city"]
        state=request.form["state"]
        pincode=request.form["pincode"]
        mobile=request.form["mobile"]
        gender=request.form["gender"]
        bloodgroup=request.form["bloodgroup"]
        cursor.execute("INSERT INTO db_cc_photo (img , name , age, city, state, pincode, mobile, gender, bloodgroup ,predicted_class_name, probabilities) VALUES (%s ,%s ,%s ,%s ,%s ,%s ,%s ,%s ,%s ,%s ,%s)", (filename, name , age, city, state, pincode, mobile, gender, bloodgroup, predicted_class_name, prob) )
        conn.commit() 
        #file_path = file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Make prediction
        
        #display(image_path)
        

        return render_template('index.html', prediction_text='Covid-19 Detection Result is : {}'.format(predicted_class_name , prob), prob_text='And Probability is : {}'.format( prob) , filename=filename) 
 
        #flash('Image successfully uploaded and displayed below')
        #return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)




@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

  
if __name__ == "__main__":
    app.debug = True
    app.run()