import tensorflow as tf
import flask
import flask.scaffold
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
from tensorflow.keras import models
import numpy as np
import imageio
import os
from keras.preprocessing import image

flask_app = flask.Flask(__name__)
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "ML React App", 
		  description = "Predict results using a trained model")

name_space = app.namespace('prediction', description='Prediction APIs')

model = app.model('Prediction params', 
				  {'file': fields.String(required = True, 
				  							   description="Image to predict", 
    					  				 	   help="Image cannot be blank")})

# classifier = joblib.load('classifier.joblib')
loaded_model = models.load_model('experiment6 with batch norm.h5')
# graph = tf.get_default_graph()

@name_space.route("/")
class MainClass(Resource):

	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response

	@app.expect(model)		
	def post(self):
		#try: 				
		# imagefile = request.files.get('file','')
		imagefile = request.files['file']
		filename = werkzeug.utils.secure_filename(imagefile.filename)
		print("\nReceived image file name : " + imagefile.filename)
		imagefile.save(filename)
		img = image.load_img(filename, target_size=(100, 100))
		img_array = image.img_to_array(img).astype('float32')/255
		img_array = np.expand_dims(img_array, axis=0)
		# x = preprocess_input(x)

		print('predicting...')
# 		with graph.as_default():
		predictions = loaded_model.predict(img_array)
		print('done')
		print('predicted', predictions)
		# decode the results into a list of tuples (class, description, probability)
		# (one such list for each sample in the batch)
		class_names = ['benign', 'malignant']
		score = tf.nn.softmax(predictions[0])
		print(score)
		predicted_label = str("Gambar ini kemungkinan besar tergolong {} dengan tingkat kepercayaan {:.1f}%."
    		.format(class_names[np.argmax(score)], 100 * np.max(score)))
		print('Predicted:', )
		class1 = None
		score1 = None
		class2 = None
		score2 = None
		if(class_names[np.argmax(score)]  == 'benign' and class_names[np.argmin(score)] == 'malignant'):
			class1 = class_names[np.argmax(score)]
			score1 = 100 * np.max(score)
			class2 = class_names[np.argmin(score)]
			score2 = 100 * np.min(score)
		elif (class_names[np.argmax(score)] == 'malignant' and class_names[np.argmin(score)] == 'benign'):
			class1 = class_names[np.argmin(score)]
			score1 = 100 * np.min(score)
			class2 = class_names[np.argmax(score)]
			score2 = 100 * np.max(score)
		result = [
			{
				"class" : class1, 
				"score" : score1
			}, 
			{
				"class" : class2, 
				"score" : score2
			},
			{
				"predicted_label" : predicted_label
			}
		]
		response = jsonify({
			"statusCode": 200,
			"status": "Prediction made",
			"result": result #predicted_label
		})

		response.headers.add('Access-Control-Allow-Origin', '*')
		return response
flask_app.run(host="0.0.0.0", port=os.environ.get('PORT', 5000), debug=True)