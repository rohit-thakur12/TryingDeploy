from flask import Flask,render_template,url_for,request
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
model = tf.keras.models.load_model('detech.h5')

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        data = tokenizer.texts_to_sequences(message)
        padded = pad_sequences(data, maxlen=100, padding='post', truncating='post')
		my_prediction = model.predict(padded)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
