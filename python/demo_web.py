#!/usr/bin/env python3
"""
Demo Web - Web界面演示

基于FastAPI和WebSocket的Web界面
"""

import asyncio
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import logging

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("请安装FastAPI: pip install fastapi uvicorn")

from cockpit_assistant import CockpitAssistant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能座舱助手</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            padding: 30px 0;
        }
        
        header h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        header p {
            color: #888;
            font-size: 1.1rem;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 20px;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
        
        .chat-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            height: 600px;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
            margin-bottom: 15px;
        }
        
        .message {
            margin-bottom: 15px;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            text-align: right;
        }
        
        .message.assistant {
            text-align: left;
        }
        
        .message-content {
            display: inline-block;
            padding: 12px 18px;
            border-radius: 18px;
            max-width: 80%;
            word-wrap: break-word;
        }
        
        .user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-bottom-right-radius: 4px;
        }
        
        .assistant .message-content {
            background: rgba(255, 255, 255, 0.1);
            border-bottom-left-radius: 4px;
        }
        
        .message-label {
            font-size: 0.75rem;
            color: #666;
            margin-bottom: 5px;
        }
        
        .function-call {
            background: rgba(0, 210, 255, 0.1);
            border-left: 3px solid #00d2ff;
            padding: 10px;
            margin-top: 10px;
            border-radius: 0 10px 10px 0;
            font-size: 0.85rem;
        }
        
        .input-area {
            display: flex;
            gap: 10px;
        }
        
        .input-area input {
            flex: 1;
            padding: 15px 20px;
            border: none;
            border-radius: 25px;
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            font-size: 1rem;
            outline: none;
            transition: background 0.3s;
        }
        
        .input-area input:focus {
            background: rgba(255, 255, 255, 0.15);
        }
        
        .input-area input::placeholder {
            color: #666;
        }
        
        .input-area button {
            padding: 15px 25px;
            border: none;
            border-radius: 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            font-size: 1rem;
            cursor: pointer;
            transition: transform 0.2s, opacity 0.2s;
        }
        
        .input-area button:hover {
            transform: scale(1.05);
        }
        
        .input-area button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .status-panel {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 20px;
        }
        
        .status-panel h3 {
            font-size: 1.1rem;
            margin-bottom: 15px;
            color: #00d2ff;
        }
        
        .status-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .status-item:last-child {
            border-bottom: none;
        }
        
        .status-label {
            color: #888;
        }
        
        .status-value {
            font-weight: 600;
        }
        
        .status-value.on {
            color: #4caf50;
        }
        
        .status-value.off {
            color: #f44336;
        }
        
        .quick-actions {
            margin-top: 20px;
        }
        
        .quick-actions h3 {
            font-size: 1.1rem;
            margin-bottom: 15px;
            color: #00d2ff;
        }
        
        .quick-btn {
            display: block;
            width: 100%;
            padding: 10px;
            margin-bottom: 8px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            background: transparent;
            color: #fff;
            font-size: 0.9rem;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .quick-btn:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        
        .typing-indicator {
            display: none;
            padding: 10px;
        }
        
        .typing-indicator.show {
            display: block;
        }
        
        .typing-indicator span {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #00d2ff;
            border-radius: 50%;
            margin: 0 2px;
            animation: bounce 1.4s infinite;
        }
        
        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes bounce {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.85rem;
        }
        
        .connection-status.connected {
            background: rgba(76, 175, 80, 0.2);
            color: #4caf50;
        }
        
        .connection-status.disconnected {
            background: rgba(244, 67, 54, 0.2);
            color: #f44336;
        }
    </style>
</head>
<body>
    <div class="connection-status disconnected" id="connectionStatus">
        断开连接
    </div>
    
    <div class="container">
        <header>
            <h1>🚗 智能座舱助手</h1>
            <p>Cockpit Assistant - AI驱动的车载交互系统</p>
        </header>
        
        <div class="main-content">
            <div class="chat-container">
                <div class="chat-messages" id="chatMessages">
                    <div class="message assistant">
                        <div class="message-label">助手</div>
                        <div class="message-content">
                            你好！我是智能座舱助手小智，可以帮你控制空调、车窗、导航、音乐等。有什么需要帮助的吗？
                        </div>
                    </div>
                </div>
                
                <div class="typing-indicator" id="typingIndicator">
                    <span></span><span></span><span></span>
                </div>
                
                <div class="input-area">
                    <input type="text" id="userInput" placeholder="输入消息..." autocomplete="off">
                    <button id="sendBtn" onclick="sendMessage()">发送</button>
                </div>
            </div>
            
            <div class="status-panel">
                <h3>🚙 车辆状态</h3>
                <div class="status-item">
                    <span class="status-label">空调</span>
                    <span class="status-value off" id="acStatus">关闭</span>
                </div>
                <div class="status-item">
                    <span class="status-label">温度</span>
                    <span class="status-value" id="acTemp">24°C</span>
                </div>
                <div class="status-item">
                    <span class="status-label">导航</span>
                    <span class="status-value" id="navStatus">未启动</span>
                </div>
                <div class="status-item">
                    <span class="status-label">音乐</span>
                    <span class="status-value off" id="musicStatus">停止</span>
                </div>
                <div class="status-item">
                    <span class="status-label">电量</span>
                    <span class="status-value" id="batteryStatus">78%</span>
                </div>
                <div class="status-item">
                    <span class="status-label">续航</span>
                    <span class="status-value" id="rangeStatus">320km</span>
                </div>
                
                <div class="quick-actions">
                    <h3>⚡ 快捷指令</h3>
                    <button class="quick-btn" onclick="quickSend('把空调打开')">打开空调</button>
                    <button class="quick-btn" onclick="quickSend('查看车辆状态')">车辆状态</button>
                    <button class="quick-btn" onclick="quickSend('播放音乐')">播放音乐</button>
                    <button class="quick-btn" onclick="quickSend('打开全部车窗')">打开车窗</button>
                    <button class="quick-btn" onclick="quickSend('导航到最近的加油站')">最近加油站</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let isGenerating = false;
        let currentAssistantMessage = null;
        
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = () => {
                document.getElementById('connectionStatus').textContent = '已连接';
                document.getElementById('connectionStatus').className = 'connection-status connected';
            };
            
            ws.onclose = () => {
                document.getElementById('connectionStatus').textContent = '断开连接';
                document.getElementById('connectionStatus').className = 'connection-status disconnected';
                // 尝试重连
                setTimeout(connect, 3000);
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }
        
        function handleMessage(data) {
            if (data.type === 'token') {
                // 流式token
                if (!currentAssistantMessage) {
                    currentAssistantMessage = addMessage('assistant', '');
                }
                currentAssistantMessage.querySelector('.message-content').textContent += data.content;
                scrollToBottom();
            } else if (data.type === 'end') {
                // 生成结束
                isGenerating = false;
                document.getElementById('typingIndicator').classList.remove('show');
                document.getElementById('sendBtn').disabled = false;
                currentAssistantMessage = null;
            } else if (data.type === 'function_call') {
                // 函数调用
                if (currentAssistantMessage) {
                    const fcDiv = document.createElement('div');
                    fcDiv.className = 'function-call';
                    fcDiv.textContent = `🔧 ${data.name}: ${data.result}`;
                    currentAssistantMessage.appendChild(fcDiv);
                }
            } else if (data.type === 'status') {
                // 更新状态
                updateStatus(data.status);
            }
        }
        
        function addMessage(role, content) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            const labelDiv = document.createElement('div');
            labelDiv.className = 'message-label';
            labelDiv.textContent = role === 'user' ? '你' : '助手';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            
            messageDiv.appendChild(labelDiv);
            messageDiv.appendChild(contentDiv);
            messagesDiv.appendChild(messageDiv);
            
            scrollToBottom();
            return messageDiv;
        }
        
        function scrollToBottom() {
            const messagesDiv = document.getElementById('chatMessages');
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            
            if (!message || isGenerating || !ws || ws.readyState !== WebSocket.OPEN) {
                return;
            }
            
            // 添加用户消息
            addMessage('user', message);
            input.value = '';
            
            // 发送到服务器
            ws.send(JSON.stringify({ type: 'chat', content: message }));
            
            // 显示加载状态
            isGenerating = true;
            document.getElementById('typingIndicator').classList.add('show');
            document.getElementById('sendBtn').disabled = true;
        }
        
        function quickSend(message) {
            document.getElementById('userInput').value = message;
            sendMessage();
        }
        
        function updateStatus(status) {
            if (status.ac) {
                document.getElementById('acStatus').textContent = status.ac.on ? '开启' : '关闭';
                document.getElementById('acStatus').className = 'status-value ' + (status.ac.on ? 'on' : 'off');
                document.getElementById('acTemp').textContent = status.ac.temperature + '°C';
            }
            if (status.navigation) {
                document.getElementById('navStatus').textContent = status.navigation.active ? 
                    (status.navigation.destination || '导航中') : '未启动';
            }
            if (status.music) {
                document.getElementById('musicStatus').textContent = status.music.playing ? '播放中' : '停止';
                document.getElementById('musicStatus').className = 'status-value ' + (status.music.playing ? 'on' : 'off');
            }
            if (status.battery !== undefined) {
                document.getElementById('batteryStatus').textContent = status.battery + '%';
            }
            if (status.range !== undefined) {
                document.getElementById('rangeStatus').textContent = status.range + 'km';
            }
        }
        
        // 回车发送
        document.getElementById('userInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // 连接WebSocket
        connect();
    </script>
</body>
</html>
"""


# 创建FastAPI应用
if HAS_FASTAPI:
    app = FastAPI(title="智能座舱助手")
    assistant: Optional[CockpitAssistant] = None
    
    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML_TEMPLATE
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        global assistant
        
        await websocket.accept()
        logger.info("WebSocket connected")
        
        try:
            # 发送初始状态
            if assistant:
                status = assistant.get_vehicle_state()
                await websocket.send_json({"type": "status", "status": status})
            
            while True:
                # 接收消息
                data = await websocket.receive_json()
                
                if data.get("type") == "chat":
                    content = data.get("content", "")
                    
                    if assistant:
                        # 流式响应
                        async for token in assistant.chat(content):
                            await websocket.send_json({
                                "type": "token",
                                "content": token
                            })
                        
                        # 发送结束标记
                        await websocket.send_json({"type": "end"})
                        
                        # 更新状态
                        status = assistant.get_vehicle_state()
                        await websocket.send_json({"type": "status", "status": status})
                    else:
                        await websocket.send_json({
                            "type": "token",
                            "content": "助手未初始化"
                        })
                        await websocket.send_json({"type": "end"})
                
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")


def main(model_path: str, host: str = "0.0.0.0", port: int = 8000,
         n_ctx: int = 4096, n_gpu_layers: int = 35):
    """主函数"""
    global assistant
    
    if not HAS_FASTAPI:
        print("请安装依赖: pip install fastapi uvicorn")
        return
    
    print(f"正在加载模型: {model_path}")
    
    try:
        assistant = CockpitAssistant(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers
        )
        print("模型加载成功！")
    except Exception as e:
        print(f"警告: 模型加载失败 ({e})，将使用模拟模式")
        assistant = CockpitAssistant("mock_model.gguf")
    
    print(f"\n启动Web服务器: http://{host}:{port}")
    print("按 Ctrl+C 停止服务器\n")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能座舱助手 - Web演示")
    parser.add_argument(
        "model_path",
        nargs="?",
        default="models/qwen2.5-7b-instruct-q4_k_m.gguf",
        help="模型文件路径"
    )
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("-c", "--ctx", type=int, default=4096, help="上下文长度")
    parser.add_argument("-g", "--gpu-layers", type=int, default=35, help="GPU层数")
    
    args = parser.parse_args()
    main(args.model_path, args.host, args.port, args.ctx, args.gpu_layers)
