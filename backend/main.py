from flask import Flask, request, jsonify
import random
from werkzeug.utils import secure_filename

app = Flask(__name__)
@app.route('/uploadImage', methods=['POST'])
def uploadImage():
    try:
        data= '''
        |  a  |  b  |  c  |  d  |  f  |    
        |  1  |  1  |  1  |  1  |  1  |  
        |  1  |  1  |  1  |  1  |  0  |   
        |  1  |  1  |  1  |  0  |  1  | 
        |  1  |  1  |  1  |  0  |  0  | 
        |  1  |  1  |  0  |  1  |  1  | 
        |  1  |  1  |  0  |  1  |  0  | 
        |  1  |  1  |  0  |  0  |  1  |
        |  1  |  1  |  0  |  0  |  0  | 
        |  1  |  0  |  1  |  1  |  1  | 
        |  1  |  0  |  1  |  1  |  0  |  
        |  1  |  0  |  1  |  0  |  1  |  
        |  1  |  1  |  1  |  1  |  1  |  
        |  1  |  1  |  1  |  1  |  0  |   
        |  1  |  1  |  1  |  0  |  1  | 
        |  1  |  1  |  1  |  0  |  0  | 
        |  1  |  1  |  0  |  1  |  1  | 
        |  1  |  1  |  0  |  1  |  0  | 
        |  1  |  1  |  0  |  0  |  1  |
        |  1  |  1  |  0  |  0  |  0  | 
        |  1  |  0  |  1  |  1  |  1  | 
        |  1  |  0  |  1  |  1  |  0  |  
        |  1  |  0  |  1  |  0  |  1  |  
        '''
        if 'image' not in request.files:
            raise ValueError("No 'image' file part in the request")

        imageFile = request.files['image']

        if imageFile.filename == '':
            raise ValueError("No selected file")

        filename = secure_filename(imageFile.filename)
        five_digit_number = random.randint(10000, 99999)
        imageFile.save("./uploaded/" + str(five_digit_number) + filename)
        type_process = request.form.get('type')
        print(type_process)
        return jsonify({"status": "200", "result": "Image uploaded successfully",
                        "data":data
                        })
    except Exception as e:
        return jsonify({"status": "Error", "error": str(e)})








if __name__ == '__main__':
    app.run(debug=True, threaded=True)