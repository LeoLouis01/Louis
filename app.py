import base64
from flask import Flask, request, jsonify, render_template
import face_recognition
import numpy as np
from PIL import Image
import os
import io
import cv2
from werkzeug.utils import secure_filename
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import datetime
import requests
import json

app = Flask(__name__)
# 设置图片的保存路径
UPLOAD_FOLDER = 'backend/temp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

face_model = load_model('model/fas.h5') #方案一的模型
face_detector = MTCNN()

known_faces = []
known_names = []
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_known_faces():
    global known_faces, known_names  # 指定使用全局变量
    known_faces = []
    known_names = []
    for filename in os.listdir('backend/images'):
        img = face_recognition.load_image_file(f'backend/images/{filename}')
        encodings = face_recognition.face_encodings(img)
        if encodings:
            encoding = encodings[0]  # 只取第一个面部编码
            known_faces.append(encoding)
            known_names.append(filename.split('.')[0])  # 假设文件名是人的名字
        else:
            print(f"No faces found in {filename}.")  # 打印没有找到面部的图片名称

def liveness_detection_scheme1(image):
    img = cv2.resize(image, (224, 224))  # 根据模型输入尺寸调整图像大小
    img = (img - 127.5) / 127.5  # 归一化图像
    img = np.expand_dims(img, axis=0)  # 增加批次维度
    score = face_model.predict(img)[0][0]
    print(score)
    return score > 0.7

def liveness_detection_scheme2(image):
    headers = {'content-type': 'application/json'}
    # 百度API活体检测逻辑
    url = "https://aip.baidubce.com/rest/2.0/face/v3/faceverify?access_token={'填写你获取的api'}"      #注意修改access_token
    b64_data = base64.b64encode(open("backend/temp/capture.jpg",'rb').read())
    b64_str = str(b64_data, 'utf-8')
    payload = [
        {
            "image":b64_str,
            "image_type":"BASE64"
        }
    ]
    payload = json.dumps(payload)
    res = requests.post(url, data=payload, headers=headers).json()
    if res['result']['face_liveness'] > 0.9:
        return True
    else:
        return False

def save_attendance_record(name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = f"{timestamp} - {name}\n"
    os.makedirs('data', exist_ok=True)
    with open('backend/record.txt', 'a') as file:
        file.write(record)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_attendance', methods=['POST'])
def check_attendance():
    if 'image' not in request.files or 'detection_scheme' not in request.form:
        return jsonify(message="缺少图像文件或检测方案"), 400

    file = request.files['image']
    detection_scheme = request.form['detection_scheme']

    if file and allowed_file(file.filename):
        content_type = file.content_type
        extension = ''
        if content_type == 'image/jpeg':
            extension = '.jpg'
        elif content_type == 'image/png':
            extension = '.png'

        filename = secure_filename(file.filename.rsplit('.', 1)[0]) + extension
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        print(f"Image saved to {save_path}")

        try:
            # 将文件指针复位到文件开头
            file.seek(0)
            img_stream = io.BytesIO(file.read())
            img = Image.open(img_stream)
            img = np.array(img)

            # 根据选择的方案进行活体检测
            if detection_scheme == 'scheme1':
                is_live = liveness_detection_scheme1(img)
            elif detection_scheme == 'scheme2':
                is_live = liveness_detection_scheme2(img)
            else:
                return jsonify(message="Invalid detection scheme"), 400

            if is_live:
                face_locations = face_recognition.face_locations(img)
                face_encodings = face_recognition.face_encodings(img, face_locations)

                names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_faces, face_encoding)
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_names[first_match_index]
                        names.append(name)
                        save_attendance_record(name)
                    else:
                        names.append("Unknown")

                return jsonify(names=names)
            else:
                return jsonify(message="未检测到真实人脸"), 400

        except Exception as e:
            print(f"处理图像时发生错误: {e}")
            return jsonify(message="处理图像时发生错误"), 500
    else:
        return jsonify(message="文件类型不允许或无效的文件"), 400


if __name__ == '__main__':
    load_known_faces()  # 调用函数加载已知的面部
    app.run(debug=True)
