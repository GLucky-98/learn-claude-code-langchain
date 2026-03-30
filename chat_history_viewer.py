#!/usr/bin/env python3
"""
对话历史记录实时监控与可视化 Web 服务
监控指定文件夹中的新对话历史记录文件，并在 Web 界面实时显示
"""

import json
from datetime import datetime
import os

def messages_to_json(messages) -> str:
    """将消息列表转换为 JSON 字符串"""
    data = []
    for msg in messages:
        msg_dict = {
            "type": msg.type,
            "text": msg.text,
            "content": msg.content,
            "additional_kwargs": msg.additional_kwargs,
            "id": getattr(msg, "id", None),
        }
        if msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
            msg_dict["tool_calls"] = [tc for tc in msg.tool_calls]
        data.append(msg_dict)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./chat_history/{timestamp}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return filename

import signal
import sys
import threading
from pathlib import Path
from flask import Flask, render_template, Response
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ============ 配置区 ============
WATCH_FOLDER = "./chat_history"  # 监控的文件夹路径
HOST = "0.0.0.0"
PORT = 5000
# ================================

app = Flask(__name__)

# 全局存储
conversations = {}  # {filename: conversations}
lock = threading.Lock()
observer = None

class ChatHistoryHandler(FileSystemEventHandler):
    """处理对话历史记录文件的创建和修改事件"""

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            print(f"[+] 新文件创建: {event.src_path}")
            load_conversation(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            print(f"[~] 文件被修改: {event.src_path}")
            load_conversation(event.src_path)

def load_conversation(filepath):
    """加载或更新对话记录"""
    messages = []
    with lock:
        messages = process_conversation_file(filepath)

    if messages is not None:
        filename = os.path.basename(filepath)
        mtime = os.path.getmtime(filepath)
        with lock:
            conversations[filename] = {
                'messages': messages,
                'mtime': mtime,
                'filepath': filepath
            }

def process_conversation_file(filepath):
    """解析对话历史记录文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        messages = []
        for msg in data:
            msg_type = msg.get('type', 'unknown')
            content = msg.get('content', msg.get('text', ''))
            tool_calls = msg.get('tool_calls', [])

            messages.append({
                'type': msg_type,
                'content': content,
                'tool_calls': tool_calls,
                'id': msg.get('id'),
                'additional_kwargs': msg.get('additional_kwargs', {})
            })

        return messages
    except Exception as e:
        print(f"[!] 读取文件错误 ({filepath}): {e}")
        return None

def init_watchdog():
    """初始化文件监控"""
    global observer
    event_handler = ChatHistoryHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
    observer.start()
    print(f"[*] 开始监控文件夹: {WATCH_FOLDER}")
    return observer

def load_existing_files():
    """加载已存在的对话历史记录文件"""
    Path(WATCH_FOLDER).mkdir(exist_ok=True)
    for filepath in Path(WATCH_FOLDER).glob("*.json"):
        print(f"[*] 加载已有文件: {filepath}")
        load_conversation(str(filepath))

def shutdown_server():
    """优雅关闭服务器"""
    global observer
    print("\n[*] 正在停止文件监控...")
    if observer:
        observer.stop()
        observer.join()
    print("[*] 服务器已关闭，端口已释放")
    sys.exit(0)

def free_port(port):
    """尝试释放被占用的端口"""
    import subprocess
    try:
        # 找到占用端口的进程并杀掉
        result = subprocess.run(f"lsof -ti:{port} | xargs kill -9 2>/dev/null",
                               shell=True, capture_output=True)
        return True
    except Exception:
        return False

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/api/conversations')
def get_conversations():
    """获取所有对话记录"""
    with lock:
        result = {}
        for filename, data in conversations.items():
            result[filename] = {
                'messages': data['messages'],
                'mtime': datetime.fromtimestamp(data['mtime']).strftime('%Y-%m-%d %H:%M:%S')
            }
        return Response(
            json.dumps(result, ensure_ascii=False),
            mimetype='application/json'
        )

@app.route('/api/conversations/<filename>')
def get_conversation(filename):
    """获取指定对话记录"""
    with lock:
        if filename in conversations:
            data = conversations[filename]
            return Response(
                json.dumps({
                    'messages': data['messages'],
                    'mtime': datetime.fromtimestamp(data['mtime']).strftime('%Y-%m-%d %H:%M:%S')
                }, ensure_ascii=False),
                mimetype='application/json'
            )
        return Response(json.dumps({'error': 'Not found'}), status=404, mimetype='application/json')

def force_free_port(port):
    """强制释放端口"""
    import socket
    import subprocess
    # 先用 lsof 强制 kill
    subprocess.run(f"lsof -ti:{port} | xargs kill -9 2>/dev/null", shell=True)
    # 等待一下让端口释放
    import time
    time.sleep(0.5)
    # 再次尝试绑定
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', port))
        s.close()
        return True
    except OSError:
        return False

if __name__ == "__main__":
    # 注册信号处理，确保 Ctrl+C 时正确关闭
    signal.signal(signal.SIGINT, lambda s, f: shutdown_server())
    signal.signal(signal.SIGTERM, lambda s, f: shutdown_server())

    # 强制清理残留端口
    if not force_free_port(PORT):
        print(f"[!] 端口 {PORT} 无法释放，请手动执行: lsof -ti:{PORT} | xargs kill -9")
        sys.exit(1)

    # 加载已存在的文件
    load_existing_files()

    # 启动文件监控线程
    init_watchdog()

    # 启动 Flask 服务
    print(f"[*] Web 服务启动: http://{HOST}:{PORT}")
    print(f"[*] 按 Ctrl+C 停止服务")
    app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)
