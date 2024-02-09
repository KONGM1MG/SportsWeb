import os
from flask import Flask, render_template, request, flash, make_response, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from PIL import Image


app = Flask(__name__)
app.config['SECRET_KEY'] = '123456'


# 允许上传type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', "PNG", "JPG", 'JPEG'])  # 大写的.JPG是不允许的
# 用于存储矩形框数据的全局变量或容器
rectangles = []

# check type
def allowed_file(filename):
    return '.' in filename and filename.split('.', 1)[1] in ALLOWED_EXTENSIONS
    # 圆括号中的1是分割次数


# upload path
UPLOAD_FOLDER = './images'


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """"目前只支持上传英文名"""
    if request.method == 'POST':
        # 获取上传文件
        file = request.files['file']
        print(dir(file))
        # 检查文件对象是否存在且合法
        if file and allowed_file(file.filename):  # 哪里规定file都有什么属性
            filename = secure_filename(file.filename)  # 把汉字文件名抹掉了，所以下面多一道检查
            if filename != file.filename:
                flash("only support ASCII name")
                return render_template('upload.html')
            # save
            try:
                file.save(os.path.join(UPLOAD_FOLDER, filename))  # 现在似乎不会出现重复上传同名文件的问题
            except FileNotFoundError:
                os.mkdir(UPLOAD_FOLDER)
                file.save(os.path.join(UPLOAD_FOLDER, filename))
            return redirect(url_for('update', fileName=filename))
        else:
            return 'Upload Failed'
    else:  # GET方法
        return render_template('upload.html')


def render_photo_as_page(filename):
    """每次调用都将上传的图片复制到static中"""
    img = Image.open(os.path.join(UPLOAD_FOLDER, filename))  # 上传文件夹和static分离
    img.save(os.path.join('./static/images', filename))  # 这里要求jpg还是png都必须保存成png，因为html里是写死的
    result = {}
    result["fileName"] = filename
    return result


@app.route('/upload/<path:fileName>', methods=['POST', 'GET'])
def update(fileName):
    """输入url加载图片，并返回预测值；上传图片，也会重定向到这里"""
    result = render_photo_as_page(fileName)
    return render_template('show.html', fname='images/' + fileName, result=result)


@app.route('/save_rectangles', methods=['POST'])
def save_rectangles():
    # 获取从前端发送过来的 JSON 数据
    data = request.json

    # 在这里你可以对接收到的矩形数据进行处理
    # 例如，将其保存到全局变量或容器中
    rectangles.extend(data)  # 直接将接收到的数据添加到列表中
    print(rectangles)
    # 返回一个成功的响应
    return jsonify({'message': 'Rectangles saved successfully.'}), 200


@app.route('/get_rectangles', methods=['GET'])
def get_rectangles():
    # 返回存储的矩形框数据给前端
    return jsonify({'rectangles': rectangles}), 200

if __name__ == '__main__':
    app.run(debug=True)
