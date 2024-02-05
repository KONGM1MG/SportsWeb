# coding:utf-8

from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import time

from datetime import timedelta

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.SEND_FILE_MAX_AGE_DEFAULT = timedelta(seconds=1)
# 保存边界框
bounding_boxes = []

@app.route('/index', methods=['POST', 'GET'])  # 添加路由
def upload():
    # 全局变量
    global bounding_boxes
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)
        # 清空边界框
        bounding_boxes = []
        print(f.filename)
        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        # cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)

        return render_template('image.html', userinput=user_input, val1=time.time())

    return render_template('index.html')

@app.route('/processed_image')
def processed_image():
    # Display the processed image
    return send_file('./static/images/processed_image.jpg', mimetype='image/jpg')

@app.route('/update_bounding_boxes', methods=['POST'])
def update_bounding_boxes():
    global bounding_boxes

    # Receive bounding box coordinates from the client
    data = request.get_json()
    bounding_boxes.append(data)

    return jsonify({"status": "success"})

if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=8080, debug=True)