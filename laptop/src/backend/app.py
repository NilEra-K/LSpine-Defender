from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 导入蓝图
# from auth.routes import auth
# from blog.routes import blog
from app.modules.api.router import api

# 注册蓝图
# app.register_blueprint(auth, url_prefix='/auth')
# app.register_blueprint(blog, url_prefix='/blog')
app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='127.0.0.1', port=16020, debug=True)