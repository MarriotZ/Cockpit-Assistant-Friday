#!/usr/bin/env python3
"""
Demo Web - Web界面演示 (语音增强版) - 音频处理修复版

修复内容：
1. 前端使用 MediaRecorder 替代 ScriptProcessorNode（不会丢帧）
2. 前端使用 OfflineAudioContext 进行高质量重采样（自动抗混叠滤波）
3. 后端使用 scipy.signal.resample_poly 进行验证重采样
4. 修复首次录音 WebM header 不完整的问题（添加最小录音时间保护）
"""

import asyncio
import json
import os
import sys
import argparse
import base64
import re
import io
import wave
import struct
from pathlib import Path
from typing import Optional
import logging
import numpy as np

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

# 尝试导入语音相关库
try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    print("提示: 安装 faster-whisper 可启用语音识别: pip install faster-whisper")

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False
    print("提示: 安装 edge-tts 可启用语音合成: pip install edge-tts")

# 尝试导入 scipy 用于高质量重采样
try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Friday · 语音智能助手</title>
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
            --warning: #f59e0b;
            --warning-soft: rgba(245, 158, 11, 0.1);
            --danger: #ef4444;
            --danger-soft: rgba(239, 68, 68, 0.1);
            
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
            padding: 20px 0;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 18px;
        }

        /* ===== 顶部动态LOGO - 统一透明背景 ===== */
        .brand-logo {
            width: 64px;
            height: 64px;
            border-radius: var(--radius-md);
            overflow: hidden;
            position: relative;
            flex-shrink: 0;
            background: transparent;
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

        .topbar-actions {
            display: flex;
            align-items: center;
            gap: 16px;
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
            grid-template-columns: 1fr 340px;
            gap: 24px;
            padding: 8px 0 24px;
            align-items: stretch;
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
            height: calc(100vh - 140px);
            min-height: 600px;
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

        /* 助手头像 - 统一透明背景 */
        .chat-avatar {
            width: 52px;
            height: 52px;
            border-radius: var(--radius-md);
            overflow: hidden;
            position: relative;
            box-shadow: var(--shadow-md);
            flex-shrink: 0;
            background: transparent;
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

        .btn-group {
            display: flex;
            gap: 8px;
        }

        .send-btn, .voice-btn {
            background: var(--accent);
            border: none;
            border-radius: var(--radius-md);
            padding: 0 20px;
            color: #fff;
            font-size: 0.9rem;
            font-weight: 600;
            font-family: inherit;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .send-btn:hover, .voice-btn:hover {
            background: var(--accent-dark);
        }

        .send-btn:active, .voice-btn:active { 
            background: var(--accent-dark);
            transform: scale(0.98);
        }

        .send-btn:disabled, .voice-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            background: var(--accent);
        }

        .send-btn svg, .voice-btn svg { width: 20px; height: 20px; }

        /* 语音按钮特殊状态 */
        .voice-btn {
            width: 52px;
            padding: 0;
            position: relative;
            overflow: hidden;
        }

        .voice-btn.listening {
            background: var(--danger);
            animation: voicePulse 1.5s ease-in-out infinite;
        }

        .voice-btn.processing {
            background: var(--warning);
        }

        @keyframes voicePulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
            50% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
        }

        /* ===== 语音助手浮层 ===== */
        .voice-overlay {
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(8px);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
        }

        .voice-overlay.active {
            opacity: 1;
            visibility: visible;
        }

        .voice-assistant {
            background: var(--bg-card);
            border-radius: var(--radius-xl);
            padding: 48px;
            text-align: center;
            max-width: 420px;
            width: 90%;
            box-shadow: var(--shadow-lg);
            transform: scale(0.9);
            transition: transform 0.3s ease;
        }

        .voice-overlay.active .voice-assistant {
            transform: scale(1);
        }

        .voice-avatar {
            width: 120px;
            height: 120px;
            margin: 0 auto 24px;
            border-radius: 50%;
            background: transparent;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }

        .voice-avatar img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .voice-avatar::before {
            content: '';
            position: absolute;
            inset: -8px;
            border-radius: 50%;
            border: 3px solid var(--accent);
            opacity: 0;
            animation: none;
        }

        .voice-overlay.listening .voice-avatar::before {
            opacity: 1;
            animation: avatarPulse 2s ease-in-out infinite;
        }

        @keyframes avatarPulse {
            0%, 100% { transform: scale(1); opacity: 0.8; }
            50% { transform: scale(1.1); opacity: 0.4; }
        }

        .voice-status {
            font-size: 1.4rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 8px;
        }

        .voice-hint {
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-bottom: 24px;
        }

        .voice-transcript {
            background: var(--bg-warm);
            border-radius: var(--radius-md);
            padding: 16px 20px;
            min-height: 60px;
            font-size: 1rem;
            color: var(--text-primary);
            margin-bottom: 24px;
            text-align: left;
        }

        .voice-transcript:empty::before {
            content: '等待语音输入...';
            color: var(--text-muted);
        }

        .voice-waveform {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            height: 48px;
            margin-bottom: 24px;
        }

        .voice-waveform span {
            width: 4px;
            height: 24px;
            background: var(--accent);
            border-radius: 2px;
            animation: waveform 1s ease-in-out infinite;
        }

        .voice-waveform span:nth-child(1) { animation-delay: 0s; }
        .voice-waveform span:nth-child(2) { animation-delay: 0.1s; }
        .voice-waveform span:nth-child(3) { animation-delay: 0.2s; }
        .voice-waveform span:nth-child(4) { animation-delay: 0.3s; }
        .voice-waveform span:nth-child(5) { animation-delay: 0.4s; }

        @keyframes waveform {
            0%, 100% { height: 12px; }
            50% { height: 40px; }
        }

        .voice-cancel {
            background: var(--bg-warm);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 12px 32px;
            color: var(--text-secondary);
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .voice-cancel:hover {
            background: var(--danger-soft);
            border-color: var(--danger);
            color: var(--danger);
        }

        /* ===== 唤醒词指示器 ===== */
        .wake-indicator {
            position: fixed;
            bottom: 32px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 100px;
            padding: 12px 24px;
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 0.85rem;
            color: var(--text-secondary);
            box-shadow: var(--shadow-lg);
            z-index: 100;
            transition: all 0.3s ease;
        }

        .wake-indicator.active {
            border-color: var(--success);
            background: var(--success-soft);
        }

        .wake-indicator .mic-icon {
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .wake-indicator .mic-icon svg {
            width: 100%;
            height: 100%;
        }

        .wake-indicator.active .mic-icon svg {
            color: var(--success);
            animation: micPulse 1.5s ease-in-out infinite;
        }

        @keyframes micPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.2); }
        }

        /* ===== 侧边栏 ===== */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
            max-height: calc(100vh - 140px);
            overflow-y: auto;
        }

        .sidebar::-webkit-scrollbar { width: 6px; }
        .sidebar::-webkit-scrollbar-track { background: transparent; }
        .sidebar::-webkit-scrollbar-thumb { 
            background: var(--border-strong); 
            border-radius: 10px; 
        }

        .card {
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            border: 1px solid var(--border);
            box-shadow: var(--shadow-md);
            overflow: hidden;
        }

        .card-header {
            padding: 14px 20px;
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

        .card-body { padding: 16px 20px; }

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
            gap: 8px;
        }

        .quick-btn {
            background: var(--bg-warm);
            border: 1.5px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 12px 14px;
            color: var(--text-primary);
            font-size: 0.85rem;
            font-family: inherit;
            font-weight: 500;
            cursor: pointer;
            text-align: left;
            transition: all 0.25s ease;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .quick-btn:hover {
            background: var(--accent-soft);
            border-color: var(--accent);
            color: var(--accent);
            transform: translateX(4px);
        }

        .quick-btn .icon {
            width: 28px;
            height: 28px;
            background: var(--bg-card);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            transition: all 0.25s ease;
            box-shadow: var(--shadow-sm);
        }

        .quick-btn:hover .icon {
            background: var(--bg-card);
            transform: scale(1.1) rotate(-5deg);
            box-shadow: var(--shadow-md);
        }

        /* 语音设置卡片 */
        .voice-settings {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .voice-toggle {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 14px;
            background: var(--bg-warm);
            border-radius: var(--radius-sm);
            border: 1px solid var(--border);
        }

        .voice-toggle-label {
            font-size: 0.85rem;
            color: var(--text-primary);
            font-weight: 500;
        }

        .toggle-switch {
            position: relative;
            width: 48px;
            height: 26px;
        }

        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }

        .toggle-slider {
            position: absolute;
            cursor: pointer;
            inset: 0;
            background: var(--border-strong);
            border-radius: 26px;
            transition: 0.3s;
        }

        .toggle-slider::before {
            content: '';
            position: absolute;
            width: 20px;
            height: 20px;
            left: 3px;
            bottom: 3px;
            background: white;
            border-radius: 50%;
            transition: 0.3s;
            box-shadow: var(--shadow-sm);
        }

        .toggle-switch input:checked + .toggle-slider {
            background: var(--accent);
        }

        .toggle-switch input:checked + .toggle-slider::before {
            transform: translateX(22px);
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
                    <span>Intelligent Cockpit System</span>
                </div>
            </div>
            <div class="topbar-actions">
                <div class="connection-badge" id="connectionStatus">
                    <div class="connection-dot"></div>
                    <span>连接中...</span>
                </div>
            </div>
        </nav>

        <main class="main">
            <section class="chat-panel">
                <div class="chat-header">
                    <div class="chat-avatar">
                        <img src="/static/assistant_avatar.png" alt="Friday助手">
                    </div>
                    <div class="chat-info">
                        <h2>Friday 你的智能助手</h2>
                        <p>在线 · 随时为您效劳</p>
                    </div>
                </div>

                <div class="chat-messages" id="chatMessages">
                    <div class="message assistant">
                        <div class="message-meta">Friday</div>
                        <div class="message-bubble">您好，我是您的智能座舱助手 Friday。您可以通过文字或语音与我交流，或者说 "Hey Friday" 唤醒我。需要我为您调节车内环境、规划路线，还是来点音乐？</div>
                    </div>
                </div>

                <div class="typing-indicator" id="typingIndicator">
                    <span></span><span></span><span></span>
                </div>

                <div class="chat-input-wrap">
                    <div class="input-row">
                        <input type="text" class="input-field" id="userInput" placeholder="输入指令或问题..." autocomplete="off">
                        <div class="btn-group">
                            <button class="voice-btn" id="voiceBtn" onclick="toggleVoice()" title="语音输入">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                                    <line x1="12" y1="19" x2="12" y2="23"></line>
                                    <line x1="8" y1="23" x2="16" y2="23"></line>
                                </svg>
                            </button>
                            <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                                发送
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                    <line x1="22" y1="2" x2="11" y2="13"></line>
                                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                                </svg>
                            </button>
                        </div>
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

                <!-- 语音设置卡片 -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">🎤</div>
                        <h3>语音设置</h3>
                    </div>
                    <div class="card-body">
                        <div class="voice-settings">
                            <div class="voice-toggle">
                                <span class="voice-toggle-label">唤醒词检测</span>
                                <label class="toggle-switch">
                                    <input type="checkbox" id="wakeWordToggle" onchange="toggleWakeWord(this.checked)">
                                    <span class="toggle-slider"></span>
                                </label>
                            </div>
                            <div class="voice-toggle">
                                <span class="voice-toggle-label">语音播报</span>
                                <label class="toggle-switch">
                                    <input type="checkbox" id="ttsToggle" checked onchange="toggleTTS(this.checked)">
                                    <span class="toggle-slider"></span>
                                </label>
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

    <!-- 语音助手浮层 -->
    <div class="voice-overlay" id="voiceOverlay">
        <div class="voice-assistant">
            <div class="voice-avatar">
                <img src="/static/assistant_avatar.png" alt="Friday">
            </div>
            <div class="voice-status" id="voiceStatus">正在聆听...</div>
            <div class="voice-hint" id="voiceHint">请说出您的指令</div>
            <div class="voice-waveform" id="voiceWaveform">
                <span></span><span></span><span></span><span></span><span></span>
            </div>
            <div class="voice-transcript" id="voiceTranscript"></div>
            <button class="voice-cancel" onclick="cancelVoice()">取消</button>
        </div>
    </div>

    <!-- 唤醒词指示器 -->
    <div class="wake-indicator" id="wakeIndicator" style="display: none;">
        <div class="mic-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="23"></line>
                <line x1="8" y1="23" x2="16" y2="23"></line>
            </svg>
        </div>
        <span>说 "Hey Friday" 唤醒</span>
    </div>

    <script>
        // ===== 全局状态 =====
        let ws = null;
        let isGenerating = false;
        let currentAssistantMessage = null;
        
        // 语音相关状态
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let wakeWordEnabled = false;
        let ttsEnabled = true;
        let recognition = null;
        
        // ★★★ 新增：录音时间追踪，解决首次录音 WebM header 不完整问题 ★★★
        let recordingStartTime = 0;
        const MIN_RECORDING_TIME = 800;  // 最小录音时间 800ms，确保 WebM header 完整
        
        // ===== WebSocket连接 =====
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
                document.getElementById('voiceBtn').disabled = false;
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
            } else if (data.type === 'asr_result') {
                handleASRResult(data.text);
            } else if (data.type === 'tts_audio') {
                if (ttsEnabled) {
                    playTTSAudio(data.audio);
                }
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
            ws.send(JSON.stringify({ type: 'chat', content: message, tts: ttsEnabled }));

            isGenerating = true;
            document.getElementById('typingIndicator').classList.add('show');
            document.getElementById('sendBtn').disabled = true;
            document.getElementById('voiceBtn').disabled = true;
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

        // ===== 语音功能（修复版 - 解决首次录音 WebM header 不完整问题）=====
        
        async function toggleVoice() {
            if (isRecording) {
                stopRecording();
            } else {
                await startRecording();
            }
        }

        /**
         * 使用 OfflineAudioContext 进行高质量重采样
         * 关键：浏览器会自动进行抗混叠滤波，避免频率混叠问题
         */
        async function resampleWithOfflineContext(audioBuffer, targetSampleRate) {
            const sourceSampleRate = audioBuffer.sampleRate;
            
            if (sourceSampleRate === targetSampleRate) {
                return audioBuffer.getChannelData(0);
            }
            
            const duration = audioBuffer.duration;
            const targetLength = Math.round(duration * targetSampleRate);
            
            // 创建离线上下文进行重采样（会自动低通滤波）
            const offlineCtx = new OfflineAudioContext(1, targetLength, targetSampleRate);
            
            const source = offlineCtx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(offlineCtx.destination);
            source.start(0);
            
            const renderedBuffer = await offlineCtx.startRendering();
            
            console.log(`OfflineAudioContext 重采样: ${sourceSampleRate}Hz -> ${targetSampleRate}Hz, ${renderedBuffer.length} 样本`);
            
            return renderedBuffer.getChannelData(0);
        }

        /**
         * ★★★ 修复后的开始录音函数 ★★★
         * 关键修改：不使用 timeslice 参数，让浏览器自己管理数据块
         */
        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    } 
                });
                
                // 显示语音浮层
                const overlay = document.getElementById('voiceOverlay');
                overlay.classList.add('active', 'listening');
                document.getElementById('voiceStatus').textContent = '正在聆听...';
                document.getElementById('voiceHint').textContent = '请说出您的指令';
                document.getElementById('voiceTranscript').textContent = '';
                
                const voiceBtn = document.getElementById('voiceBtn');
                voiceBtn.classList.add('listening');
                
                // 使用 MediaRecorder 录制（不会丢帧）
                audioChunks = [];
                
                // 选择支持的格式
                let mimeType = 'audio/webm;codecs=opus';
                if (!MediaRecorder.isTypeSupported(mimeType)) {
                    mimeType = 'audio/webm';
                }
                if (!MediaRecorder.isTypeSupported(mimeType)) {
                    mimeType = 'audio/mp4';
                }
                console.log('MediaRecorder 格式:', mimeType);
                
                mediaRecorder = new MediaRecorder(stream, { 
                    mimeType: mimeType,
                    audioBitsPerSecond: 128000
                });
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                        console.log(`收到音频块: ${event.data.size} bytes, 总计 ${audioChunks.length} 块`);
                    }
                };
                
                mediaRecorder.onstop = async () => {
                    console.log(`录音停止，共 ${audioChunks.length} 个音频块`);
                    stream.getTracks().forEach(track => track.stop());
                    await processRecordedAudio(mimeType);
                };
                
                // ★★★ 关键修复：不使用 timeslice 参数，让浏览器自己管理数据块 ★★★
                // 原来是 mediaRecorder.start(250); 会导致数据分片，首次录音 header 不完整
                mediaRecorder.start();
                
                // ★★★ 记录开始时间 ★★★
                recordingStartTime = Date.now();
                console.log('录音开始时间:', recordingStartTime);
                
                window.currentStream = stream;
                isRecording = true;
                
                // 8秒后自动停止
                window.recordingTimeout = setTimeout(() => {
                    if (isRecording) {
                        stopRecording();
                    }
                }, 8000);
                
            } catch (error) {
                console.error('无法访问麦克风:', error);
                alert('无法访问麦克风，请检查权限设置');
            }
        }

        /**
         * ★★★ 修复后的停止录音函数 ★★★
         * 关键修改：添加最小录音时间保护，确保 WebM header 完整
         */
        function stopRecording() {
            clearTimeout(window.recordingTimeout);
            
            // ★★★ 计算录音时长 ★★★
            const recordingDuration = Date.now() - recordingStartTime;
            console.log(`录音时长: ${recordingDuration}ms`);
            
            // ★★★ 关键修复：如果录音时间太短，等待一下再停止 ★★★
            if (recordingDuration < MIN_RECORDING_TIME) {
                const waitTime = MIN_RECORDING_TIME - recordingDuration;
                console.log(`录音时间不足，等待 ${waitTime}ms`);
                
                document.getElementById('voiceStatus').textContent = '正在处理...';
                
                setTimeout(() => {
                    doStopRecording();
                }, waitTime);
            } else {
                doStopRecording();
            }
        }
        
        /**
         * ★★★ 新增：实际执行停止录音 ★★★
         */
        function doStopRecording() {
            isRecording = false;
            
            const voiceBtn = document.getElementById('voiceBtn');
            voiceBtn.classList.remove('listening');
            voiceBtn.classList.add('processing');
            
            document.getElementById('voiceStatus').textContent = '正在处理...';
            document.getElementById('voiceOverlay').classList.remove('listening');
            
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                // ★★★ 关键修复：在停止前请求所有待处理的数据 ★★★
                try {
                    mediaRecorder.requestData();
                } catch (e) {
                    console.log('requestData 不支持或已无数据');
                }
                
                // 稍微延迟停止，确保 requestData 完成
                setTimeout(() => {
                    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                        mediaRecorder.stop();
                    }
                }, 100);
            } else {
                handleASRResult('');
            }
        }

        /**
         * ★★★ 修复后的处理录音函数 ★★★
         * 关键修改：添加数据大小检查
         */
        async function processRecordedAudio(mimeType) {
            if (audioChunks.length === 0) {
                console.warn('没有录制到音频数据');
                handleASRResult('');
                return;
            }
            
            try {
                // 合并音频块
                const audioBlob = new Blob(audioChunks, { type: mimeType });
                console.log('录制完成，大小:', audioBlob.size, 'bytes, 块数:', audioChunks.length);
                
                // ★★★ 关键修复：检查数据大小（WebM header 通常至少需要几百字节）★★★
                if (audioBlob.size < 500) {
                    console.warn('音频数据太小，可能不完整，尝试发送原始数据');
                    await sendRawAudio(audioBlob, mimeType);
                    return;
                }
                
                // 解码为 AudioBuffer
                const arrayBuffer = await audioBlob.arrayBuffer();
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                
                let audioBuffer;
                try {
                    audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                } catch (decodeError) {
                    console.error('浏览器解码失败:', decodeError);
                    console.log('尝试发送原始数据让后端处理...');
                    // 解码失败时发送原始数据让后端处理
                    await sendRawAudio(audioBlob, mimeType);
                    audioContext.close();
                    return;
                }
                
                console.log('解码成功:', audioBuffer.sampleRate, 'Hz,', audioBuffer.duration.toFixed(2), '秒');
                
                // 使用 OfflineAudioContext 重采样到 16kHz（关键：会自动低通滤波）
                const targetSampleRate = 16000;
                const resampledData = await resampleWithOfflineContext(audioBuffer, targetSampleRate);
                
                // 编码为 WAV
                const wavBlob = encodeWAV(resampledData, targetSampleRate);
                console.log('WAV 编码完成:', wavBlob.size, 'bytes');
                
                // 发送到服务器
                await sendAudioForRecognition(wavBlob);
                
                audioContext.close();
                
            } catch (error) {
                console.error('音频处理错误:', error);
                handleASRResult('');
            }
        }

        /**
         * 发送原始音频（当 decodeAudioData 失败时使用）
         */
        async function sendRawAudio(audioBlob, mimeType) {
            const reader = new FileReader();
            reader.onloadend = () => {
                const base64Audio = reader.result.split(',')[1];
                if (ws && ws.readyState === WebSocket.OPEN) {
                    const format = mimeType.includes('webm') ? 'webm' : 'mp4';
                    console.log('发送原始音频, 格式:', format, ', 大小:', audioBlob.size);
                    ws.send(JSON.stringify({
                        type: 'audio',
                        audio: base64Audio,
                        format: format
                    }));
                }
            };
            reader.readAsDataURL(audioBlob);
        }

        /**
         * 编码为 WAV 格式
         */
        function encodeWAV(samples, sampleRate) {
            const numChannels = 1;
            const bitsPerSample = 16;
            const bytesPerSample = bitsPerSample / 8;
            const blockAlign = numChannels * bytesPerSample;
            const byteRate = sampleRate * blockAlign;
            const dataSize = samples.length * bytesPerSample;
            const headerSize = 44;
            const totalSize = headerSize + dataSize;
            
            const buffer = new ArrayBuffer(totalSize);
            const view = new DataView(buffer);
            
            // RIFF header
            writeString(view, 0, 'RIFF');
            view.setUint32(4, totalSize - 8, true);
            writeString(view, 8, 'WAVE');
            
            // fmt chunk
            writeString(view, 12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);  // PCM
            view.setUint16(22, numChannels, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, byteRate, true);
            view.setUint16(32, blockAlign, true);
            view.setUint16(34, bitsPerSample, true);
            
            // data chunk
            writeString(view, 36, 'data');
            view.setUint32(40, dataSize, true);
            
            // 写入采样数据
            let offset = 44;
            for (let i = 0; i < samples.length; i++) {
                const sample = Math.max(-1, Math.min(1, samples[i]));
                const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
                view.setInt16(offset, intSample, true);
                offset += 2;
            }
            
            return new Blob([buffer], { type: 'audio/wav' });
        }

        function writeString(view, offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        }

        async function sendAudioForRecognition(audioBlob) {
            const reader = new FileReader();
            reader.onloadend = () => {
                const base64Audio = reader.result.split(',')[1];
                if (ws && ws.readyState === WebSocket.OPEN) {
                    console.log('发送 WAV 到服务器, base64 长度:', base64Audio.length);
                    ws.send(JSON.stringify({
                        type: 'audio',
                        audio: base64Audio,
                        format: 'wav'
                    }));
                }
            };
            reader.readAsDataURL(audioBlob);
        }

        function handleASRResult(text) {
            const voiceBtn = document.getElementById('voiceBtn');
            voiceBtn.classList.remove('processing');
            
            document.getElementById('voiceTranscript').textContent = text;
            
            if (text && text.trim()) {
                document.getElementById('voiceStatus').textContent = '已识别';
                
                setTimeout(() => {
                    document.getElementById('voiceOverlay').classList.remove('active');
                    
                    addMessage('user', text);
                    ws.send(JSON.stringify({ type: 'chat', content: text, tts: ttsEnabled }));
                    
                    isGenerating = true;
                    document.getElementById('typingIndicator').classList.add('show');
                    document.getElementById('sendBtn').disabled = true;
                    document.getElementById('voiceBtn').disabled = true;
                }, 800);
            } else {
                document.getElementById('voiceStatus').textContent = '未识别到语音';
                setTimeout(() => {
                    document.getElementById('voiceOverlay').classList.remove('active');
                }, 1500);
            }
        }

        /**
         * ★★★ 修复后的取消语音函数 ★★★
         * 关键修改：取消时不触发 onstop 处理
         */
        function cancelVoice() {
            clearTimeout(window.recordingTimeout);
            
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                // ★★★ 取消时不触发 onstop 处理 ★★★
                mediaRecorder.ondataavailable = null;
                mediaRecorder.onstop = null;
                mediaRecorder.stop();
            }
            if (window.currentStream) {
                window.currentStream.getTracks().forEach(track => track.stop());
            }
            
            isRecording = false;
            audioChunks = [];
            
            document.getElementById('voiceOverlay').classList.remove('active', 'listening');
            document.getElementById('voiceBtn').classList.remove('listening', 'processing');
        }

        function playTTSAudio(base64Audio) {
            try {
                const audioData = atob(base64Audio);
                const arrayBuffer = new ArrayBuffer(audioData.length);
                const view = new Uint8Array(arrayBuffer);
                for (let i = 0; i < audioData.length; i++) {
                    view[i] = audioData.charCodeAt(i);
                }
                
                const blob = new Blob([arrayBuffer], { type: 'audio/mp3' });
                const url = URL.createObjectURL(blob);
                const audio = new Audio(url);
                audio.play();
                audio.onended = () => URL.revokeObjectURL(url);
            } catch (error) {
                console.error('播放TTS音频失败:', error);
            }
        }

        // ===== 唤醒词检测 =====
        function toggleWakeWord(enabled) {
            wakeWordEnabled = enabled;
            const indicator = document.getElementById('wakeIndicator');
            
            if (enabled) {
                indicator.style.display = 'flex';
                startWakeWordDetection();
            } else {
                indicator.style.display = 'none';
                indicator.classList.remove('active');
                stopWakeWordDetection();
            }
        }

        function startWakeWordDetection() {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                console.warn('浏览器不支持语音识别');
                alert('您的浏览器不支持语音识别功能，请使用Chrome浏览器');
                document.getElementById('wakeWordToggle').checked = false;
                document.getElementById('wakeIndicator').style.display = 'none';
                return;
            }
            
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'zh-CN';
            
            recognition.onresult = (event) => {
                const indicator = document.getElementById('wakeIndicator');
                
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript.toLowerCase();
                    console.log('识别到:', transcript);
                    
                    if (transcript.includes('friday') || 
                        transcript.includes('hey friday') || 
                        transcript.includes('嘿friday') ||
                        transcript.includes('你好friday') ||
                        transcript.includes('嗨friday') ||
                        transcript.includes('弗莱德')) {
                        
                        indicator.classList.add('active');
                        recognition.stop();
                        
                        setTimeout(() => {
                            indicator.classList.remove('active');
                            startRecording();
                        }, 500);
                        
                        return;
                    }
                }
            };
            
            recognition.onerror = (event) => {
                console.error('唤醒词检测错误:', event.error);
                if (event.error !== 'no-speech' && wakeWordEnabled) {
                    setTimeout(startWakeWordDetection, 1000);
                }
            };
            
            recognition.onend = () => {
                if (wakeWordEnabled && !isRecording) {
                    setTimeout(() => {
                        if (wakeWordEnabled) {
                            try {
                                recognition.start();
                            } catch (e) {
                                console.log('重启唤醒词检测');
                            }
                        }
                    }, 500);
                }
            };
            
            try {
                recognition.start();
                console.log('唤醒词检测已启动');
            } catch (e) {
                console.error('启动唤醒词检测失败:', e);
            }
        }

        function stopWakeWordDetection() {
            if (recognition) {
                recognition.stop();
                recognition = null;
            }
        }

        function toggleTTS(enabled) {
            ttsEnabled = enabled;
        }

        document.getElementById('userInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        connect();
    </script>
</body>
</html>
"""


# ===== ASR引擎 =====
class ASREngine:
    """语音识别引擎"""
    
    def __init__(self, model_size: str = "small", device: str = "cuda"):
        self.model_size = model_size
        self.device = device
        self._model = None
        self._initialized = False
    
    def initialize(self):
        if self._initialized:
            return
        
        if HAS_WHISPER:
            logger.info(f"Loading Whisper model: {self.model_size}")
            
            device = self.device
            compute_type = "float16" if device == "cuda" else "int8"
            
            if device == "cuda":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        device = "cpu"
                        compute_type = "int8"
                except ImportError:
                    device = "cpu"
                    compute_type = "int8"
            
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type=compute_type
                )
                logger.info(f"Whisper model loaded on {device}")
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}")
                self._model = None
        
        self._initialized = True
    
    async def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        if not self._initialized:
            self.initialize()
        
        if self._model is None:
            return "语音识别不可用"
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._transcribe_sync,
            audio_data,
            sample_rate
        )
        return result
    
    def _transcribe_sync(self, audio_data: np.ndarray, sample_rate: int) -> str:
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            if audio_data.max() > 1.0:
                audio_data = audio_data / 32768.0
        
        segments, info = self._model.transcribe(
            audio_data,
            language="zh",
            beam_size=5
        )
        
        text = "".join([segment.text for segment in segments])
        return text.strip()


# ===== TTS引擎 =====
class TTSEngine:
    """语音合成引擎"""
    
    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural"):
        self.voice = voice
    
    async def synthesize(self, text: str) -> bytes:
        if not HAS_EDGE_TTS:
            return b""
        
        if not text.strip():
            return b""
        
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            audio_data = b""
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            return audio_data
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return b""


# ===== 音频处理（修复版）=====

def resample_audio_scipy(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """使用 scipy 进行高质量重采样（带抗混叠滤波）"""
    if from_rate == to_rate:
        return audio
    
    # 计算重采样参数
    gcd = np.gcd(from_rate, to_rate)
    up = to_rate // gcd
    down = from_rate // gcd
    
    # scipy.signal.resample_poly 会自动进行抗混叠滤波
    resampled = signal.resample_poly(audio, up, down)
    
    return resampled.astype(np.float32)


def resample_audio_linear(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """简单线性插值重采样（备用方案，质量较差）"""
    if from_rate == to_rate:
        return audio
    
    ratio = from_rate / to_rate
    new_length = int(len(audio) / ratio)
    
    old_indices = np.arange(len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)
    
    resampled = np.interp(new_indices, old_indices, audio)
    return resampled.astype(np.float32)


def convert_wav_to_numpy(wav_data: bytes) -> np.ndarray:
    """将WAV音频数据转换为numpy数组（修复版：增加采样率验证和重采样）"""
    try:
        if len(wav_data) < 44:
            logger.error("WAV data too short")
            return np.array([])
        
        wav_io = io.BytesIO(wav_data)
        with wave.open(wav_io, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            audio_bytes = wav_file.readframes(n_frames)
        
        logger.info(f"WAV 参数: {sample_rate}Hz, {n_channels}ch, {sample_width*8}bit, {n_frames} frames")
        
        if sample_width == 2:
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        elif sample_width == 1:
            audio_data = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.int16) - 128
            audio_data = audio_data * 256
        else:
            logger.error(f"Unsupported sample width: {sample_width}")
            return np.array([])
        
        if n_channels == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)
        
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # 关键修复：如果采样率不是 16kHz，进行高质量重采样
        if sample_rate != 16000:
            logger.info(f"后端重采样: {sample_rate}Hz -> 16000Hz")
            if HAS_SCIPY:
                audio_float = resample_audio_scipy(audio_float, sample_rate, 16000)
                logger.info("使用 scipy 高质量重采样")
            else:
                audio_float = resample_audio_linear(audio_float, sample_rate, 16000)
                logger.warning("使用线性插值重采样（建议安装 scipy: pip install scipy）")
        
        logger.info(f"转换完成: {len(audio_float)} 样本 ({len(audio_float)/16000:.2f}秒)")
        return audio_float
        
    except Exception as e:
        logger.error(f"WAV conversion error: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])


def convert_webm_to_numpy(webm_data: bytes) -> np.ndarray:
    """将 WebM 音频转换为 numpy 数组（需要 pydub + ffmpeg）"""
    try:
        from pydub import AudioSegment
        
        audio_io = io.BytesIO(webm_data)
        audio = AudioSegment.from_file(audio_io, format='webm')
        
        # 转换为单声道 16kHz
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # 转换为 numpy
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0
        
        logger.info(f"WebM 转换完成: {len(samples)} 样本 ({len(samples)/16000:.2f}秒)")
        return samples
        
    except ImportError:
        logger.error("需要 pydub 来处理 WebM: pip install pydub")
        logger.error("还需要安装 ffmpeg")
        return np.array([])
    except Exception as e:
        logger.error(f"WebM conversion error: {e}")
        return np.array([])


def clean_text_for_tts(text: str) -> str:
    """清理文本用于TTS"""
    cleaned = re.sub(r'\{[^}]+\}', '', text)
    cleaned = re.sub(r'[✅❌🔧📊🔋🛞🛢️📍🌡️🎵🪟❄️⚡]', '', cleaned)
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()


# ===== FastAPI应用 =====
if HAS_FASTAPI:
    app = FastAPI(title="智能座舱助手 - 语音增强版（修复）")
    assistant: Optional[CockpitAssistant] = None
    asr_engine: Optional[ASREngine] = None
    tts_engine: Optional[TTSEngine] = None
    
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
        global assistant, asr_engine, tts_engine

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
                    need_tts = data.get("tts", True)
                    logger.info(f"Received chat message: {content[:50]}...")

                    if assistant:
                        try:
                            response_text = ""
                            token_count = 0
                            async for token in assistant.chat(content):
                                token_count += 1
                                response_text += token
                                await websocket.send_json({"type": "token", "content": token})
                            
                            logger.info(f"Chat completed, sent {token_count} tokens")
                            await websocket.send_json({"type": "end"})
                            
                            status = assistant.get_vehicle_state()
                            await websocket.send_json({"type": "status", "status": status})
                            
                            if need_tts and tts_engine and HAS_EDGE_TTS:
                                clean_text = clean_text_for_tts(response_text)
                                if clean_text:
                                    audio_data = await tts_engine.synthesize(clean_text)
                                    if audio_data:
                                        audio_base64 = base64.b64encode(audio_data).decode()
                                        await websocket.send_json({
                                            "type": "tts_audio",
                                            "audio": audio_base64
                                        })
                            
                        except Exception as e:
                            logger.error(f"Error in chat generation: {e}", exc_info=True)
                            await websocket.send_json({"type": "token", "content": f"错误: {str(e)}"})
                            await websocket.send_json({"type": "end"})
                    else:
                        await websocket.send_json({"type": "token", "content": "助手未初始化"})
                        await websocket.send_json({"type": "end"})

                elif data.get("type") == "audio":
                    audio_base64 = data.get("audio", "")
                    audio_format = data.get("format", "wav")
                    
                    if audio_base64 and asr_engine:
                        try:
                            audio_bytes = base64.b64decode(audio_base64)
                            logger.info(f"Received audio: {len(audio_bytes)} bytes, format: {audio_format}")
                            
                            # 根据格式选择转换方法
                            if audio_format == 'wav':
                                audio_array = convert_wav_to_numpy(audio_bytes)
                            elif audio_format in ['webm', 'mp4']:
                                audio_array = convert_webm_to_numpy(audio_bytes)
                            else:
                                logger.warning(f"Unknown format: {audio_format}, trying wav")
                                audio_array = convert_wav_to_numpy(audio_bytes)
                            
                            if len(audio_array) > 0:
                                text = await asr_engine.transcribe(audio_array)
                                logger.info(f"ASR result: {text}")
                                await websocket.send_json({
                                    "type": "asr_result",
                                    "text": text
                                })
                            else:
                                logger.warning("Audio conversion returned empty array")
                                await websocket.send_json({
                                    "type": "asr_result",
                                    "text": ""
                                })
                        except Exception as e:
                            logger.error(f"ASR error: {e}", exc_info=True)
                            await websocket.send_json({
                                "type": "asr_result",
                                "text": ""
                            })
                    else:
                        await websocket.send_json({
                            "type": "asr_result",
                            "text": "语音识别不可用"
                        })

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")


def main(model_path: str, host: str = "0.0.0.0", port: int = 8000,
         n_ctx: int = 4096, n_gpu_layers: int = 0,
         asr_model: str = "small", tts_voice: str = "zh-CN-XiaoxiaoNeural"):
    global assistant, asr_engine, tts_engine

    if not HAS_FASTAPI:
        print("请安装依赖: pip install fastapi uvicorn")
        return

    print("=" * 60)
    print("  🚗 Friday 智能座舱助手")
    print("=" * 60)
    print(f"\n正在加载模型: {model_path}")

    try:
        assistant = CockpitAssistant(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        #print("✓ LLM模型加载成功！")
    except Exception as e:
        print(f"⚠ 警告: 模型加载失败 ({e})，将使用模拟模式")
        assistant = CockpitAssistant("mock_model.gguf")

    if HAS_WHISPER:
        #print(f"\n正在加载ASR模型: {asr_model}")
        asr_engine = ASREngine(model_size=asr_model, device="cuda" if n_gpu_layers > 0 else "cpu")
        try:
            asr_engine.initialize()
            #print("✓ ASR模型加载成功！")
        except Exception as e:
            print(f"⚠ ASR模型加载失败: {e}")
    else:
        print("\n⚠ ASR不可用 (请安装 faster-whisper)")
    
    if HAS_EDGE_TTS:
        tts_engine = TTSEngine(voice=tts_voice)
        #print(f"✓ TTS引擎就绪 (语音: {tts_voice})")
    else:
        print("⚠ TTS不可用 (请安装 edge-tts)")

    print(f"\n启动Web服务器: http://{host}:{port}")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能座舱助手 - Friday")
    parser.add_argument("model_path", nargs="?", default="models/qwen2.5-7b-instruct-q4_k_m.gguf", help="模型文件路径")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("-c", "--ctx", type=int, default=4096, help="上下文长度")
    parser.add_argument("-g", "--gpu-layers", type=int, default=0, help="GPU层数")
    parser.add_argument("--asr-model", default="small", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper ASR模型大小")
    parser.add_argument("--tts-voice", default="zh-CN-XiaoxiaoNeural",
                        help="TTS语音")

    args = parser.parse_args()
    main(args.model_path, args.host, args.port, args.ctx, args.gpu_layers, 
         args.asr_model, args.tts_voice)
