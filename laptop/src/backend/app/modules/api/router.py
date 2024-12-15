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
        
        # 修改返回数据的格式: 可以返回更多参数, 只需按照这种方式组织数据就可以
        parameters = [
            {'参数 1': 23.5},
            {'参数 2': 25.1},
            {'参数 3': 22.8},
            {'参数 4': 23.5},
            {'参数 5': 25.1},
            {'参数 6': 22.8},
            {'参数 7': 23.5},
            {'参数 8': 25.1},
            {'参数 9': 22.8},
            {'参数10': 23.5},
            # ... 其他参数
        ]
        
        response = {
            'image': img_str,
            'parameters': parameters
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api.route('/v1/test', methods=['GET'])
def test():
    print("测试连接")
    return jsonify({'message': 'connection successful'})

