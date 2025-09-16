# app.py 的安全登录版本
from dotenv import load_dotenv
load_dotenv()  # 这行必须在导入其他模块之前！

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import sqlite3
import json
import os
import secrets

app = Flask(__name__)
CORS(app)

# 安全的密钥生成
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'photos'), exist_ok=True)

# 数据库初始化
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # 创建用户表（新增）
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # 创建地点表（添加user_id字段）
    c.execute('''
        CREATE TABLE IF NOT EXISTS places (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,
            lng REAL NOT NULL,
            type TEXT NOT NULL,
            name TEXT,
            note TEXT,
            rating INTEGER,
            photo_url TEXT,
            created_by TEXT,
            user_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            visited_at DATE,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # 创建留言表
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            place_id INTEGER,
            author TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (place_id) REFERENCES places (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# 创建默认用户（首次运行时）
def create_default_user():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # 检查是否已存在默认用户
    c.execute("SELECT id FROM users WHERE username = ?", ('339233',))
    if not c.fetchone():
        # 使用环境变量中的密码，如果没有则使用随机密码
        default_password = os.environ.get('DEFAULT_PASSWORD', secrets.token_urlsafe(16))
        password_hash = generate_password_hash(default_password)
        
        c.execute('''
            INSERT INTO users (username, password_hash, display_name)
            VALUES (?, ?, ?)
        ''', ('339233', password_hash, '刘等等'))
        
        conn.commit()
        
        # 如果使用了随机密码，打印出来
        if 'DEFAULT_PASSWORD' not in os.environ:
            print(f"默认用户已创建！")
            print(f"用户名: 339233")
            print(f"密码: {default_password}")
            print(f"请保存此密码并在 .env 文件中设置 DEFAULT_PASSWORD")
    
    conn.close()

# 登录装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

# 路由：登录页面
@app.route('/login')
def login_page():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return '''
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>登录 - 我们的探店地图</title>
        <!-- 替换 app.py 中登录页面的 <style> 部分 -->
        <style>
            body {
                margin: 0;
                padding: 0;
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                /* 使用本地图片作为背景 */
                background: url('/static/images/login-bg.png') center/cover no-repeat;
                /* 添加一个半透明遮罩层，让文字更清晰 */
                position: relative;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            /* 可选：添加遮罩层效果 */
            body::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.3); /* 黑色半透明遮罩 */
                z-index: 1;
            }
            
            .login-container {
                background: rgba(255, 255, 255, 0.95);
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
                width: 100%;
                max-width: 400px;
                /* 确保登录框在遮罩层上方 */
                position: relative;
                z-index: 2;
                /* 添加毛玻璃效果 */
                backdrop-filter: blur(10px);
            }
            
            .login-title {
                text-align: center;
                color: #764ba2;
                margin-bottom: 30px;
                font-size: 24px;
            }
            
            .login-emoji {
                text-align: center;
                font-size: 60px;
                margin-bottom: 20px;
                /* 添加动画效果 */
                animation: heartbeat 1.5s ease-in-out infinite;
            }
            
            @keyframes heartbeat {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.1); }
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            .form-label {
                display: block;
                margin-bottom: 8px;
                color: #666;
                font-size: 14px;
            }
            
            .form-input {
                width: 100%;
                padding: 12px;
                border: 2px solid #f093fb;
                border-radius: 10px;
                font-size: 16px;
                box-sizing: border-box;
                transition: all 0.3s;
            }
            
            .form-input:focus {
                outline: none;
                border-color: #764ba2;
                box-shadow: 0 0 10px rgba(240, 147, 251, 0.3);
            }
            
            .login-btn {
                width: 100%;
                padding: 14px;
                background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                cursor: pointer;
                transition: all 0.3s;
                position: relative;
                overflow: hidden;
            }
            
            .login-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 20px rgba(240, 147, 251, 0.4);
            }
            
            .error-msg {
                color: #f5576c;
                text-align: center;
                margin-top: 10px;
                display: none;
                animation: shake 0.5s;
            }
            
            @keyframes shake {
                0%, 100% { transform: translateX(0); }
                25% { transform: translateX(-10px); }
                75% { transform: translateX(10px); }
            }
            
            /* 响应式设计 */
            @media (max-width: 500px) {
                .login-container {
                    margin: 20px;
                    padding: 30px 20px;
                }
            }
        </style>
    </head>
    <body>
        <div class="login-container">
            <div class="login-emoji">
                <img src="/static/images/title_log.png" style="width: 80px; height: 80px; border-radius: 10px;">
            </div>
            <h2 class="login-title">欢迎来到秘密基地哇～</h2>
            <form id="loginForm">
                <div class="form-group">
                    <label class="form-label">账号</label>
                    <input type="text" class="form-input" id="username" required>
                </div>
                <div class="form-group">
                    <label class="form-label">密码</label>
                    <input type="password" class="form-input" id="password" required>
                </div>
                <button type="submit" class="login-btn">登录</button>
                <div class="error-msg" id="errorMsg"></div>
            </form>
        </div>
        
        <script>
            document.getElementById('loginForm').onsubmit = async (e) => {
                e.preventDefault();
                const username = document.getElementById('username').value;
                const password = document.getElementById('password').value;
                
                try {
                    const response = await fetch('/api/login', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ username, password })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        window.location.href = '/';
                    } else {
                        document.getElementById('errorMsg').style.display = 'block';
                        document.getElementById('errorMsg').textContent = data.error || '登录失败';
                    }
                } catch (error) {
                    document.getElementById('errorMsg').style.display = 'block';
                    document.getElementById('errorMsg').textContent = '网络错误，请重试';
                }
            };
        </script>
    </body>
    </html>
    '''

# 路由：主页面（需要登录）
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('index.html')

# API: 登录
@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': '请输入账号和密码'}), 400
    
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT id, password_hash, display_name FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    
    if user and check_password_hash(user[1], password):
        session.permanent = True
        session['user_id'] = user[0]
        session['username'] = username
        session['display_name'] = user[2]
        
        # 更新最后登录时间
        c.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user[0],))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'display_name': user[2]})
    
    conn.close()
    return jsonify({'error': '账号或密码错误'}), 401

# API: 登出
@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'success': True})

# API: 获取当前用户信息
@app.route('/api/user')
@login_required
def get_user():
    return jsonify({
        'username': session.get('username'),
        'display_name': session.get('display_name')
    })

# API: 获取所有地点（需要登录）
@app.route('/api/places', methods=['GET'])
@login_required
def get_places():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    # 只获取当前用户的地点
    c.execute('SELECT * FROM places WHERE user_id = ? ORDER BY created_at DESC', 
              (session['user_id'],))
    places = []
    for row in c.fetchall():
        places.append({
            'id': row[0],
            'lat': row[1],
            'lng': row[2],
            'type': row[3],
            'name': row[4],
            'note': row[5],
            'rating': row[6],
            'photo_url': row[7],
            'created_by': row[8],
            'created_at': row[10],
            'visited_at': row[11]
        })
    conn.close()
    return jsonify(places)

# API: 添加新地点（需要登录）
@app.route('/api/places', methods=['POST'])
@login_required
def add_place():
    data = request.json
    
    # 验证输入
    if not data.get('lat') or not data.get('lng'):
        return jsonify({'error': 'Missing coordinates'}), 400
    
    if data.get('type') not in ['heart', 'paw']:
        return jsonify({'error': 'Invalid type'}), 400
    
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO places (lat, lng, type, name, note, rating, created_by, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['lat'],
        data['lng'],
        data['type'],
        data.get('name', ''),
        data.get('note', ''),
        data.get('rating', 0),
        session.get('display_name', '匿名'),
        session['user_id']
    ))
    place_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return jsonify({'id': place_id, 'success': True})

# API: 更新地点（需要登录且是创建者）
@app.route('/api/places/<int:place_id>', methods=['PUT'])
@login_required
def update_place(place_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # 检查是否是该用户的地点
    c.execute("SELECT user_id FROM places WHERE id = ?", (place_id,))
    place = c.fetchone()
    if not place or place[0] != session['user_id']:
        conn.close()
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.json
    updates = []
    values = []
    for key in ['name', 'note', 'rating', 'visited_at']:
        if key in data:
            updates.append(f"{key} = ?")
            values.append(data[key])
    
    if updates:
        values.append(place_id)
        c.execute(f"UPDATE places SET {', '.join(updates)} WHERE id = ?", values)
        conn.commit()
    
    conn.close()
    return jsonify({'success': True})

# API: 删除地点（需要登录且是创建者）
@app.route('/api/places/<int:place_id>', methods=['DELETE'])
@login_required
def delete_place(place_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # 检查是否是该用户的地点
    c.execute("SELECT user_id FROM places WHERE id = ?", (place_id,))
    place = c.fetchone()
    if not place or place[0] != session['user_id']:
        conn.close()
        return jsonify({'error': 'Unauthorized'}), 403
    
    c.execute('DELETE FROM places WHERE id = ?', (place_id,))
    c.execute('DELETE FROM messages WHERE place_id = ?', (place_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


# API: 导出数据
@app.route('/api/export')
@login_required
def export_data():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # 获取用户的所有地点
    c.execute('''
        SELECT lat, lng, type, name, note, rating, created_at, visited_at 
        FROM places WHERE user_id = ?
        ORDER BY created_at DESC
    ''', (session['user_id'],))
    
    places = []
    for row in c.fetchall():
        places.append({
            'lat': row[0],
            'lng': row[1],
            'type': row[2],
            'name': row[3],
            'note': row[4],
            'rating': row[5],
            'created_at': row[6],
            'visited_at': row[7]
        })
    
    conn.close()
    
    # 返回 JSON 文件下载
    response = jsonify({
        'version': '1.0',
        'exported_at': datetime.now().isoformat(),
        'user': session.get('display_name'),
        'places': places
    })
    
    # 设置为文件下载
    response.headers['Content-Disposition'] = f'attachment; filename=love-map-backup-{datetime.now().strftime("%Y%m%d")}.json'
    return response

# API: 导入数据
@app.route('/api/import', methods=['POST'])
@login_required
def import_data():
    try:
        file = request.files.get('backup_file')
        if not file:
            return jsonify({'error': '请选择备份文件'}), 400
        
        # 读取 JSON
        data = json.load(file)
        places = data.get('places', [])
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        
        imported_count = 0
        for place in places:
            # 检查是否已存在（避免重复）
            c.execute('''
                SELECT id FROM places 
                WHERE lat = ? AND lng = ? AND name = ? AND user_id = ?
            ''', (place['lat'], place['lng'], place['name'], session['user_id']))
            
            if not c.fetchone():
                c.execute('''
                    INSERT INTO places (lat, lng, type, name, note, rating, created_by, user_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    place['lat'],
                    place['lng'],
                    place['type'],
                    place.get('name', ''),
                    place.get('note', ''),
                    place.get('rating', 0),
                    session.get('display_name'),
                    session['user_id'],
                    place.get('created_at', datetime.now().isoformat())
                ))
                imported_count += 1
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True, 
            'imported': imported_count,
            'total': len(places)
        })
        
    except Exception as e:
        return jsonify({'error': f'导入失败: {str(e)}'}), 500

# API: 获取统计数据
@app.route('/api/stats', methods=['GET'])
@login_required
def get_stats():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # 统计数据
    c.execute("SELECT COUNT(*) FROM places WHERE type = 'heart' AND user_id = ?", (session['user_id'],))
    want_to_go = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM places WHERE type = 'paw' AND user_id = ?", (session['user_id'],))
    visited = c.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        'want_to_go': want_to_go,
        'visited': visited,
        'completion_rate': round((visited / (want_to_go + visited) * 100) if (want_to_go + visited) > 0 else 0, 1)
    })

if __name__ == '__main__':
    init_db()
    create_default_user()
    app.run(debug=False, host='0.0.0.0', port=5001)