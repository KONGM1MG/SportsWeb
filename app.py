from flask import Flask, render_template, request, flash, make_response, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import argparse
from segment_anything import sam_model_registry, SamPredictor
import os
import torch
import PIL
import matplotlib.pyplot as plt
import numpy as np
import cv2
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


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


parser = argparse.ArgumentParser(description="auto mask generator")
args = parser.parse_args()
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# upload path
UPLOAD_FOLDER = './images'


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """"目前只支持上传英文名"""
    if request.method == 'POST':
        # 清空上次的矩形框数据
        rectangles.clear()
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
    # 例如，将其保存到全局变量或容器中
    rectangles.extend(data)  # 直接将接收到的数据添加到列表中
    print(rectangles)
    # 返回一个成功的响应
    return jsonify({'message': 'Rectangles saved successfully.'}), 200


@app.route('/get_rectangles', methods=['GET'])
def get_rectangles():
    # 返回存储的矩形框数据给前端
    return jsonify({'rectangles': rectangles}), 200


@app.route('/predict', methods=['POST'])
def predict(fileName):
    """预测"""
    # 读取图片
    # file = request.files['file']
    # fileName = file.filename
    print(fileName)
    img = cv2.imread(os.path.join(UPLOAD_FOLDER, fileName))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img)
    box = []
    for rect in rectangles:
        box.append([rect['x'], rect['y'], rect['x'] + rect['width'], rect['y'] + rect['height']])
    input_boxes = torch.tensor(box, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, img.shape[:2])
    masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
    )
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    # plt.imshow(img)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    plt.savefig(args.sav_path + '_mask_', fileName)
    plt.imshow(img)
    plt.axis('on')
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box in input_boxes:
        show_box(box.cpu().numpy(), plt.gca())
    plt.savefig(args.sav_path + '_maskwithbox_', fileName)


if __name__ == '__main__':
    app.run(debug=True)
