from flask import Blueprint, render_template, request, redirect, url_for
from flask import Blueprint, request, jsonify
import os
from PIL import Image
import base64
import io
from flask_cors import CORS

api = Blueprint('api', __name__)
CORS(api, resources={
    r"/v1/*": {
        "origins": ["http://localhost:8080"],  # 替换成你的前端地址
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

@api.route('/v1/predict', methods=['POST'])
def predict():
    print("predict")
    try:
        # 检查是否有文件
        if 'image' not in request.files:
            return jsonify({'error': '没有找到图片文件'}), 400
            
        file = request.files['image']
        
        # 检查文件是否为空
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
            
        # 确保目录存在
        save_dir = 'D:/lib'
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存图片
        img = Image.open(file.stream)
        
        # 转换为PNG格式
        if img.format != 'PNG':
            img = img.convert('RGB')
        
        # 生成文件名（这里用时间戳作为示例）
        import time
        filename = f"image_{int(time.time())}.png"
        save_path = os.path.join(save_dir, filename)
        img.save(save_path, 'PNG')
        
        # 将处理后的图片转为base64（模拟返回处理结果）
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # 返回数据
        response = {
            'image': img_str,
            'parameters': [float(i) for i in range(75)]
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api.route('/v1/test', methods=['GET'])
def test():
    print("测试连接")
    return jsonify({'message': 'connection successful'})

