#!/usr/bin/env python3
"""
Demo Web - Web界面演示

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
    from fastapi.responses import HTMLResponse, FileResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("请安装FastAPI: pip install fastapi uvicorn")

from cockpit_assistant import CockpitAssistant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Friday · 智能座舱</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            /* 现代浅色配色 - 清爽简洁 */
            --bg-base: #f8fafc;
            --bg-warm: #f1f5f9;
            --bg-card: #ffffff;
            --bg-elevated: #ffffff;
            
            --accent: #3b82f6;
            --accent-light: #60a5fa;
            --accent-dark: #2563eb;
            --accent-soft: rgba(59, 130, 246, 0.08);
            --accent-medium: rgba(59, 130, 246, 0.15);
            
            --gradient-brand: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
            --gradient-soft: linear-gradient(135deg, rgba(59, 130, 246, 0.06) 0%, rgba(99, 102, 241, 0.06) 100%);
            
            --text-primary: #1e293b;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
            
            --border: rgba(148, 163, 184, 0.2);
            --border-strong: rgba(148, 163, 184, 0.3);
            --success: #10b981;
            --success-soft: rgba(16, 185, 129, 0.1);
            --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.06);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
            --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.02);
            --shadow-glow: 0 0 40px rgba(59, 130, 246, 0.08);
            
            --radius-sm: 12px;
            --radius-md: 16px;
            --radius-lg: 24px;
            --radius-xl: 32px;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        html { font-size: 15px; }

        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-base);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }

        body::before {
            content: '';
            position: fixed;
            inset: 0;
            background: 
                radial-gradient(ellipse 80% 50% at 10% -20%, rgba(59, 130, 246, 0.05), transparent 50%),
                radial-gradient(ellipse 60% 40% at 90% 100%, rgba(99, 102, 241, 0.04), transparent 50%),
                radial-gradient(ellipse 40% 30% at 50% 50%, rgba(59, 130, 246, 0.02), transparent 50%);
            pointer-events: none;
        }

        .app {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-rows: auto 1fr;
            min-height: 100vh;
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 32px;
        }

        /* ===== 顶部导航 ===== */
        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 24px 0;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 18px;
        }

        /* ===== 顶部动态LOGO - 使用GIF ===== */
        .brand-logo {
            width: 64px;
            height: 64px;
            border-radius: var(--radius-md);
            overflow: hidden;
            position: relative;
            flex-shrink: 0;
            background: var(--bg-warm);
        }

        .brand-logo img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .brand-text h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            background: var(--gradient-brand);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.1;
        }

        .brand-text span {
            font-size: 0.7rem;
            color: var(--text-muted);
            font-weight: 500;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            margin-top: 4px;
            display: block;
        }

        .connection-badge {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 20px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 100px;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-secondary);
            box-shadow: var(--shadow-sm);
            transition: all 0.3s ease;
        }

        .connection-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-muted);
            transition: all 0.3s ease;
        }

        .connection-badge.connected {
            border-color: rgba(16, 185, 129, 0.3);
            background: var(--success-soft);
        }

        .connection-badge.connected .connection-dot {
            background: var(--success);
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.6);
            animation: dotPulse 2s ease-in-out infinite;
        }

        @keyframes dotPulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.2); }
        }

        .main {
            display: grid;
            grid-template-columns: 1fr 360px;
            gap: 28px;
            padding: 12px 0 40px;
            align-items: start;
        }

        @media (max-width: 980px) {
            .main { grid-template-columns: 1fr; }
            .sidebar { order: -1; }
            .app { padding: 0 20px; }
        }

        /* ===== 聊天面板 ===== */
        .chat-panel {
            background: var(--bg-card);
            border-radius: var(--radius-xl);
            border: 1px solid var(--border);
            box-shadow: var(--shadow-lg), var(--shadow-glow);
            display: flex;
            flex-direction: column;
            height: calc(100vh - 160px);
            min-height: 520px;
            overflow: hidden;
        }

        .chat-header {
            padding: 20px 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 16px;
            background: var(--bg-elevated);
        }

        /* 助手头像 - 使用PNG */
        .chat-avatar {
            width: 52px;
            height: 52px;
            border-radius: var(--radius-md);
            overflow: hidden;
            position: relative;
            box-shadow: var(--shadow-md);
            flex-shrink: 0;
            background: var(--bg-warm);
        }

        .chat-avatar img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .chat-avatar::after {
            content: '';
            position: absolute;
            inset: 0;
            border-radius: inherit;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.1);
        }

        .chat-info h2 {
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .chat-info p {
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-top: 2px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .chat-info p::before {
            content: '';
            width: 6px;
            height: 6px;
            background: var(--success);
            border-radius: 50%;
            animation: dotPulse 2s ease-in-out infinite;
        }

        /* 消息区 */
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            scroll-behavior: smooth;
        }

        .chat-messages::-webkit-scrollbar { width: 6px; }
        .chat-messages::-webkit-scrollbar-track { background: transparent; }
        .chat-messages::-webkit-scrollbar-thumb { 
            background: var(--border-strong); 
            border-radius: 10px; 
        }
        .chat-messages::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }

        .message {
            margin-bottom: 20px;
            animation: msgSlide 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes msgSlide {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
        }

        .message.assistant {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }

        .message-meta {
            font-size: 0.68rem;
            color: var(--text-muted);
            margin-bottom: 6px;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .message-bubble {
            max-width: 72%;
            padding: 16px 20px;
            border-radius: var(--radius-lg);
            font-size: 0.95rem;
            line-height: 1.65;
            word-wrap: break-word;
        }

        .user .message-bubble {
            background: var(--accent);
            color: #fff;
            border-bottom-right-radius: 6px;
            box-shadow: 0 2px 12px rgba(59, 130, 246, 0.25);
        }

        .assistant .message-bubble {
            background: var(--bg-warm);
            border: 1px solid var(--border);
            border-bottom-left-radius: 6px;
            color: var(--text-primary);
        }

        .function-tag {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 12px;
            padding: 10px 14px;
            background: var(--accent-soft);
            border: 1px solid var(--accent-medium);
            border-radius: var(--radius-sm);
            font-size: 0.78rem;
            color: var(--accent);
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
        }

        .function-tag svg {
            width: 14px;
            height: 14px;
            stroke: var(--accent);
        }

        /* 输入区 */
        .chat-input-wrap {
            padding: 20px 24px;
            border-top: 1px solid var(--border);
            background: var(--bg-elevated);
        }

        .typing-indicator {
            display: none;
            padding: 0 24px 14px;
            gap: 6px;
            align-items: center;
        }

        .typing-indicator.show { display: flex; }

        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: var(--accent);
            border-radius: 50%;
            animation: typingBounce 1.4s ease-in-out infinite;
        }

        .typing-indicator span:nth-child(2) { animation-delay: 0.15s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.3s; }

        @keyframes typingBounce {
            0%, 80%, 100% { transform: scale(0.7); opacity: 0.4; }
            40% { transform: scale(1); opacity: 1; }
        }

        .input-row {
            display: flex;
            gap: 12px;
        }

        .input-field {
            flex: 1;
            background: var(--bg-warm);
            border: 2px solid transparent;
            border-radius: var(--radius-md);
            padding: 16px 20px;
            color: var(--text-primary);
            font-size: 0.95rem;
            font-family: inherit;
            outline: none;
            transition: all 0.25s ease;
        }

        .input-field::placeholder { color: var(--text-muted); }

        .input-field:focus {
            background: var(--bg-card);
            border-color: var(--accent);
            box-shadow: 0 0 0 4px var(--accent-soft);
        }

        .send-btn {
            background: var(--accent);
            border: none;
            border-radius: var(--radius-md);
            padding: 0 28px;
            color: #fff;
            font-size: 0.9rem;
            font-weight: 600;
            font-family: inherit;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .send-btn:hover {
            background: var(--accent-dark);
        }

        .send-btn:active { 
            background: var(--accent-dark);
            transform: scale(0.98);
        }

        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            background: var(--accent);
        }

        .send-btn svg { width: 17px; height: 17px; }

        /* ===== 侧边栏 ===== */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        .card {
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            border: 1px solid var(--border);
            box-shadow: var(--shadow-md);
            overflow: hidden;
        }

        .card-header {
            padding: 18px 22px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 12px;
            background: var(--bg-elevated);
        }

        .card-icon {
            width: 36px;
            height: 36px;
            background: var(--gradient-soft);
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
        }

        .card-header h3 {
            font-size: 0.92rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .card-body { padding: 18px 22px; }

        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .stat-box {
            background: var(--bg-warm);
            border-radius: var(--radius-sm);
            padding: 14px 16px;
            border: 1px solid var(--border);
            transition: all 0.25s ease;
        }

        .stat-box:hover {
            border-color: var(--accent-medium);
            background: var(--accent-soft);
        }

        .stat-label {
            font-size: 0.68rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .stat-value {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-primary);
            font-variant-numeric: tabular-nums;
        }

        .stat-value.active { color: var(--success); }
        .stat-value.inactive { color: var(--text-muted); }

        /* 电量指示器 */
        .battery-section {
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid var(--border);
        }

        .battery-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 10px;
        }

        .battery-header span {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        .battery-header strong {
            font-size: 1.4rem;
            color: var(--text-primary);
            font-weight: 700;
        }

        .battery-bar {
            height: 12px;
            background: var(--bg-warm);
            border-radius: 100px;
            overflow: hidden;
            border: 1px solid var(--border);
        }

        .battery-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981 0%, #34d399 50%, #6ee7b7 100%);
            border-radius: 100px;
            transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
        }

        .battery-fill::after {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
            animation: batteryShine 2.5s ease-in-out infinite;
        }

        @keyframes batteryShine {
            0%, 100% { transform: translateX(-100%); }
            50% { transform: translateX(100%); }
        }

        .battery-meta {
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
            font-size: 0.72rem;
            color: var(--text-muted);
        }

        /* 快捷按钮 */
        .quick-grid {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .quick-btn {
            background: var(--bg-warm);
            border: 1.5px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 14px 16px;
            color: var(--text-primary);
            font-size: 0.88rem;
            font-family: inherit;
            font-weight: 500;
            cursor: pointer;
            text-align: left;
            transition: all 0.25s ease;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .quick-btn:hover {
            background: var(--accent-soft);
            border-color: var(--accent);
            color: var(--accent);
            transform: translateX(4px);
        }

        .quick-btn .icon {
            width: 32px;
            height: 32px;
            background: var(--bg-card);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            transition: all 0.25s ease;
            box-shadow: var(--shadow-sm);
        }

        .quick-btn:hover .icon {
            background: var(--bg-card);
            transform: scale(1.1) rotate(-5deg);
            box-shadow: var(--shadow-md);
        }
    </style>
</head>
<body>
    <div class="app">
        <nav class="topbar">
            <div class="brand">
                <div class="brand-logo">
                    <img src="/static/brand_logo.gif" alt="Friday Logo">
                </div>
                <div class="brand-text">
                    <h1>Friday</h1>
                    <span>Intelligent Cockpit</span>
                </div>
            </div>
            <div class="connection-badge" id="connectionStatus">
                <div class="connection-dot"></div>
                <span>连接中...</span>
            </div>
        </nav>

        <main class="main">
            <section class="chat-panel">
                <div class="chat-header">
                    <div class="chat-avatar">
                        <img src="/static/assistant_avatar.png" alt="Friday助手">
                    </div>
                    <div class="chat-info">
                        <h2>Friday 助手</h2>
                        <p>在线 · 随时为您效劳</p>
                    </div>
                </div>

                <div class="chat-messages" id="chatMessages">
                    <div class="message assistant">
                        <div class="message-meta">Friday</div>
                        <div class="message-bubble">您好，我是您的智能座舱助手 Friday。需要我为您调节车内环境、规划路线，还是来点音乐？</div>
                    </div>
                </div>

                <div class="typing-indicator" id="typingIndicator">
                    <span></span><span></span><span></span>
                </div>

                <div class="chat-input-wrap">
                    <div class="input-row">
                        <input type="text" class="input-field" id="userInput" placeholder="输入指令或问题..." autocomplete="off">
                        <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                            发送
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                <line x1="22" y1="2" x2="11" y2="13"></line>
                                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                            </svg>
                        </button>
                    </div>
                </div>
            </section>

            <aside class="sidebar">
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">🚗</div>
                        <h3>车辆状态</h3>
                    </div>
                    <div class="card-body">
                        <div class="stats-grid">
                            <div class="stat-box">
                                <div class="stat-label">空调系统</div>
                                <div class="stat-value inactive" id="acStatus">关闭</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">车内温度</div>
                                <div class="stat-value" id="acTemp">24°C</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">媒体播放</div>
                                <div class="stat-value inactive" id="musicStatus">停止</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">导航状态</div>
                                <div class="stat-value inactive" id="navStatus">未启动</div>
                            </div>
                        </div>

                        <div class="battery-section">
                            <div class="battery-header">
                                <span>电池电量</span>
                                <strong id="batteryStatus">78%</strong>
                            </div>
                            <div class="battery-bar">
                                <div class="battery-fill" id="batteryFill" style="width: 78%"></div>
                            </div>
                            <div class="battery-meta">
                                <span>预计续航</span>
                                <span id="rangeStatus">320 km</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">⚡</div>
                        <h3>快捷指令</h3>
                    </div>
                    <div class="card-body">
                        <div class="quick-grid">
                            <button class="quick-btn" onclick="quickSend('把空调打开')">
                                <span class="icon">❄️</span>启动空调
                            </button>
                            <button class="quick-btn" onclick="quickSend('查看车辆状态')">
                                <span class="icon">📊</span>车辆状态
                            </button>
                            <button class="quick-btn" onclick="quickSend('播放音乐')">
                                <span class="icon">🎵</span>播放音乐
                            </button>
                            <button class="quick-btn" onclick="quickSend('打开全部车窗')">
                                <span class="icon">🪟</span>开启车窗
                            </button>
                            <button class="quick-btn" onclick="quickSend('导航到最近的充电站')">
                                <span class="icon">🔋</span>最近充电站
                            </button>
                        </div>
                    </div>
                </div>
            </aside>
        </main>
    </div>

    <script>
        let ws = null;
        let isGenerating = false;
        let currentAssistantMessage = null;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                const badge = document.getElementById('connectionStatus');
                badge.classList.add('connected');    
                badge.querySelector('span').textContent = '已连接';
            };

            ws.onclose = () => {
                const badge = document.getElementById('connectionStatus');
                badge.classList.remove('connected');
                badge.querySelector('span').textContent = '断开连接';
                setTimeout(connect, 3000);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };

            ws.onerror = (error) => console.error('WebSocket error:', error);
        }

        function handleMessage(data) {
            if (data.type === 'token') {
                if (!currentAssistantMessage) {
                    currentAssistantMessage = addMessage('assistant', '');
                }
                currentAssistantMessage.querySelector('.message-bubble').textContent += data.content;
                scrollToBottom();
            } else if (data.type === 'end') {
                isGenerating = false;
                document.getElementById('typingIndicator').classList.remove('show');
                document.getElementById('sendBtn').disabled = false;
                currentAssistantMessage = null;
            } else if (data.type === 'function_call') {
                if (currentAssistantMessage) {
                    const fcDiv = document.createElement('div');
                    fcDiv.className = 'function-tag';
                    fcDiv.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path></svg>${data.name}: ${data.result}`;
                    currentAssistantMessage.appendChild(fcDiv);
                }
            } else if (data.type === 'status') {
                updateStatus(data.status);
            }
        }

        function addMessage(role, content) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.innerHTML = `<div class="message-meta">${role === 'user' ? '您' : 'Friday'}</div><div class="message-bubble">${content}</div>`;
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
            if (!message || isGenerating || !ws || ws.readyState !== WebSocket.OPEN) return;

            addMessage('user', message);
            input.value = '';
            ws.send(JSON.stringify({ type: 'chat', content: message }));

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
                const acEl = document.getElementById('acStatus');
                acEl.textContent = status.ac.on ? '运行中' : '关闭';
                acEl.className = 'stat-value ' + (status.ac.on ? 'active' : 'inactive');
                document.getElementById('acTemp').textContent = status.ac.temperature + '°C';
            }
            if (status.navigation) {
                const navEl = document.getElementById('navStatus');
                navEl.textContent = status.navigation.active ? (status.navigation.destination || '导航中') : '未启动';
                navEl.className = 'stat-value ' + (status.navigation.active ? 'active' : 'inactive');
            }
            if (status.music) {
                const musicEl = document.getElementById('musicStatus');
                musicEl.textContent = status.music.playing ? '播放中' : '停止';
                musicEl.className = 'stat-value ' + (status.music.playing ? 'active' : 'inactive');
            }
            if (status.battery !== undefined) {
                document.getElementById('batteryStatus').textContent = status.battery + '%';
                document.getElementById('batteryFill').style.width = status.battery + '%';
            }
            if (status.range !== undefined) {
                document.getElementById('rangeStatus').textContent = status.range + ' km';
            }
        }

        document.getElementById('userInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        connect();
    </script>
</body>
</html>
"""


if HAS_FASTAPI:
    app = FastAPI(title="智能座舱助手")
    assistant: Optional[CockpitAssistant] = None
    
    # 静态文件目录
    STATIC_DIR = Path(__file__).parent / "static"
    STATIC_DIR.mkdir(exist_ok=True)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML_TEMPLATE
    
    @app.get("/static/{filename}")
    async def serve_static(filename: str):
        file_path = STATIC_DIR / filename
        if file_path.exists():
            return FileResponse(file_path)
        return {"error": "File not found"}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        global assistant

        await websocket.accept()
        logger.info("WebSocket connected")

        try:
            if assistant:
                status = assistant.get_vehicle_state()
                await websocket.send_json({"type": "status", "status": status})

            while True:
                data = await websocket.receive_json()

                if data.get("type") == "chat":
                    content = data.get("content", "")
                    logger.info(f"Received chat message: {content[:50]}...")

                    if assistant:
                        try:
                            token_count = 0
                            async for token in assistant.chat(content):
                                token_count += 1
                                await websocket.send_json({"type": "token", "content": token})
                            logger.info(f"Chat completed, sent {token_count} tokens")
                            await websocket.send_json({"type": "end"})
                            status = assistant.get_vehicle_state()
                            await websocket.send_json({"type": "status", "status": status})
                        except Exception as e:
                            logger.error(f"Error in chat generation: {e}", exc_info=True)
                            await websocket.send_json({"type": "token", "content": f"错误: {str(e)}"})
                            await websocket.send_json({"type": "end"})
                    else:
                        await websocket.send_json({"type": "token", "content": "助手未初始化"})
                        await websocket.send_json({"type": "end"})

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")


def main(model_path: str, host: str = "0.0.0.0", port: int = 8000,
         n_ctx: int = 4096, n_gpu_layers: int = 0):
    global assistant

    if not HAS_FASTAPI:
        print("请安装依赖: pip install fastapi uvicorn")
        return

    print(f"正在加载模型: {model_path}")

    try:
        assistant = CockpitAssistant(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        print("模型加载成功！")
    except Exception as e:
        print(f"警告: 模型加载失败 ({e})，将使用模拟模式")
        assistant = CockpitAssistant("mock_model.gguf")

    print(f"\n启动Web服务器: http://{host}:{port}")
    print("按 Ctrl+C 停止服务器\n")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能座舱助手 - Web演示")
    parser.add_argument("model_path", nargs="?", default="models/qwen2.5-7b-instruct-q4_k_m.gguf", help="模型文件路径")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("-c", "--ctx", type=int, default=4096, help="上下文长度")
    parser.add_argument("-g", "--gpu-layers", type=int, default=0, help="GPU层数")

    args = parser.parse_args()
    main(args.model_path, args.host, args.port, args.ctx, args.gpu_layers)
