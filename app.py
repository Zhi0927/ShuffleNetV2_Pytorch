from flask import Flask, request, render_template, jsonify
from models.ShuffleV2 import *
import predict

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/about")
def about_page():
    return "Please subscribe  Artificial Intelligence Hub..!!!"


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':       
        img = request.files['my_image']
        if img is None or img.filename == "":
            return "Error': No file"
        if not allowed_file(img.filename):
            return 'Error: formate not supported'

        img_path = "static/" + img.filename
        img.save(img_path)
        try:
            result_ = predict.predict(img_path, verbose= True)
            result = result_["Class"]
            return render_template("index.html", prediction=result, img_path=img_path)
        except Exception as err:
            return str(err)




@app.route('/predict', methods= ['POST'])
def get_output_json():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'formate not supported'})

        try:
            img_path = "static/" + file.filename
            file.save(img_path)
            result_ = predict.predict(img_path, verbose= True)
            return jsonify(result_)
        except Exception as err:
            return jsonify({'error': str(err)})




if __name__ == '__main__':
    app.run(port=8123, debug=False)

        
        
        
# http://127.0.0.1:8123/

















































