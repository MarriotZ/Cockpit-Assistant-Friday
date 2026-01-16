#!/usr/bin/env python3
"""
Demo Web - Web Interface Demo (Voice Enhanced Version) - Audio Processing Fixed

Fixes:
1. Frontend uses MediaRecorder instead of ScriptProcessorNode (no frame drops)
2. Frontend uses OfflineAudioContext for high-quality resampling (automatic anti-aliasing filter)
3. Backend uses scipy.signal.resample_poly for validation resampling
4. Fixed first recording WebM header incomplete issue (added minimum recording time protection)
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

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("Please install FastAPI: pip install fastapi uvicorn")

from cockpit_assistant import CockpitAssistant

# Try importing voice-related libraries
try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    print("Note: Install faster-whisper to enable speech recognition: pip install faster-whisper")

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False
    print("Note: Install edge-tts to enable speech synthesis: pip install edge-tts")

# Try importing scipy for high-quality resampling
try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Friday · Intelligent Cockpit System</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            /* Modern light color scheme - Clean and simple */
            --bg-base: #f8fafc;
            --bg-warm: #f1f5f9;
            --bg-card: #ffffff;
            --bg-elevated: #ffffff;
            --bg-glass: rgba(255, 255, 255, 0.85);
            
            --accent: #3b82f6;
            --accent-light: #60a5fa;
            --accent-dark: #2563eb;
            --accent-soft: rgba(59, 130, 246, 0.08);
            --accent-medium: rgba(59, 130, 246, 0.15);
            
            --gradient-brand: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
            --gradient-soft: linear-gradient(135deg, rgba(59, 130, 246, 0.06) 0%, rgba(99, 102, 241, 0.06) 100%);
            --gradient-cockpit: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
            
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
            --shadow-cockpit: 0 8px 32px rgba(0, 0, 0, 0.08);
            
            --radius-sm: 12px;
            --radius-md: 16px;
            --radius-lg: 24px;
            --radius-xl: 32px;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        html { font-size: 14px; }

        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--gradient-cockpit);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
            overflow: hidden;
        }

        /* Cockpit background texture */
        body::before {
            content: '';
            position: fixed;
            inset: 0;
            background: 
                radial-gradient(ellipse 100% 60% at 50% -10%, rgba(59, 130, 246, 0.06), transparent 50%),
                radial-gradient(ellipse 80% 40% at 20% 90%, rgba(99, 102, 241, 0.04), transparent 50%),
                radial-gradient(ellipse 80% 40% at 80% 90%, rgba(59, 130, 246, 0.04), transparent 50%);
            pointer-events: none;
        }

        /* ===== Cockpit main layout ===== */
        .cockpit {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-rows: auto 1fr auto;
            height: 100vh;
            max-width: 1600px;
            margin: 0 auto;
            padding: 16px 24px;
            gap: 16px;
        }

        /* ===== Top HUD area ===== */
        .hud-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 20px;
            background: var(--bg-glass);
            backdrop-filter: blur(20px);
            border-radius: var(--radius-lg);
            border: 1px solid var(--border);
            box-shadow: var(--shadow-md);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-logo {
            width: 48px;
            height: 48px;
            border-radius: var(--radius-sm);
            overflow: hidden;
            background: transparent;
        }

        .brand-logo img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .brand-text span {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.85rem;
            color: var(--text-secondary);
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        /* HUD indicators */
        .hud-indicators {
            display: flex;
            align-items: center;
            gap: 24px;
        }

        .hud-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2px;
        }

        .hud-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--text-primary);
            font-variant-numeric: tabular-nums;
            line-height: 1;
        }

        .hud-value.success { color: var(--success); }
        .hud-value.warning { color: var(--warning); }

        .hud-label {
            font-size: 0.65rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 600;
        }

        .connection-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--bg-warm);
            border: 1px solid var(--border);
            border-radius: 100px;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-secondary);
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

        /* ===== Main dashboard - Three column layout ===== */
        .dashboard {
            display: grid;
            grid-template-columns: 300px 1fr 300px;
            gap: 20px;
            min-height: 0;
        }

        @media (max-width: 1200px) {
            .dashboard { 
                grid-template-columns: 1fr; 
                grid-template-rows: auto 1fr auto;
            }
            .panel-left, .panel-right { 
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 12px;
            }
        }

        /* ===== Left control panel ===== */
        .panel-left {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        /* ===== Center console ===== */
        .center-console {
            display: flex;
            flex-direction: column;
            background: var(--bg-card);
            border-radius: var(--radius-xl);
            border: 1px solid var(--border);
            box-shadow: var(--shadow-cockpit);
            overflow: hidden;
            position: relative;
        }

        /* Center top decoration line */
        .center-console::before {
            content: '';
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 60%;
            height: 3px;
            background: var(--gradient-brand);
            border-radius: 0 0 4px 4px;
        }

        .console-header {
            padding: 20px 24px 16px;
            display: flex;
            align-items: center;
            gap: 16px;
            border-bottom: 1px solid var(--border);
        }

        .assistant-orb {
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: var(--gradient-soft);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            box-shadow: var(--shadow-md);
            overflow: hidden;
        }

        .assistant-orb img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .assistant-orb::after {
            content: '';
            position: absolute;
            inset: -2px;
            border-radius: 50%;
            border: 2px solid transparent;
            background: var(--gradient-brand) border-box;
            -webkit-mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
            mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            opacity: 0.5;
        }

        .assistant-info h2 {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .assistant-info p {
            font-size: 0.75rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .assistant-info p::before {
            content: '';
            width: 6px;
            height: 6px;
            background: var(--success);
            border-radius: 50%;
            animation: dotPulse 2s ease-in-out infinite;
        }

        /* Message area */
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px 24px;
            scroll-behavior: smooth;
            min-height: 200px;
        }

        .chat-messages::-webkit-scrollbar { width: 5px; }
        .chat-messages::-webkit-scrollbar-track { background: transparent; }
        .chat-messages::-webkit-scrollbar-thumb { 
            background: var(--border-strong); 
            border-radius: 10px; 
        }

        .message {
            margin-bottom: 16px;
            animation: msgSlide 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes msgSlide {
            from { opacity: 0; transform: translateY(12px); }
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
            font-size: 0.65rem;
            color: var(--text-muted);
            margin-bottom: 4px;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .message-bubble {
            max-width: 80%;
            padding: 14px 18px;
            border-radius: var(--radius-lg);
            font-size: 0.9rem;
            line-height: 1.6;
            word-wrap: break-word;
        }

        .user .message-bubble {
            background: var(--accent);
            color: #fff;
            border-bottom-right-radius: 6px;
            box-shadow: 0 2px 12px rgba(59, 130, 246, 0.2);
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
            gap: 6px;
            margin-top: 10px;
            padding: 8px 12px;
            background: var(--accent-soft);
            border: 1px solid var(--accent-medium);
            border-radius: var(--radius-sm);
            font-size: 0.72rem;
            color: var(--accent);
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
        }

        .function-tag svg {
            width: 12px;
            height: 12px;
            stroke: var(--accent);
        }

        /* Input area */
        .console-input {
            padding: 16px 20px;
            border-top: 1px solid var(--border);
            background: var(--bg-elevated);
        }

        .typing-indicator {
            display: none;
            padding: 0 20px 12px;
            gap: 5px;
            align-items: center;
        }

        .typing-indicator.show { display: flex; }

        .typing-indicator span {
            width: 7px;
            height: 7px;
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
            gap: 10px;
        }

        .input-field {
            flex: 1;
            background: var(--bg-warm);
            border: 2px solid transparent;
            border-radius: var(--radius-md);
            padding: 14px 18px;
            color: var(--text-primary);
            font-size: 0.9rem;
            font-family: inherit;
            outline: none;
            transition: all 0.25s ease;
        }

        .input-field::placeholder { color: var(--text-muted); }

        .input-field:focus {
            background: var(--bg-card);
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-soft);
        }

        .btn-group {
            display: flex;
            gap: 8px;
        }

        .send-btn, .voice-btn {
            background: var(--accent);
            border: none;
            border-radius: var(--radius-md);
            padding: 0 18px;
            color: #fff;
            font-size: 0.85rem;
            font-weight: 600;
            font-family: inherit;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .send-btn:hover, .voice-btn:hover {
            background: var(--accent-dark);
        }

        .send-btn:active, .voice-btn:active { 
            transform: scale(0.98);
        }

        .send-btn:disabled, .voice-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .send-btn svg, .voice-btn svg { width: 18px; height: 18px; }

        .voice-btn {
            width: 48px;
            padding: 0;
            position: relative;
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
            50% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        }

        /* ===== Right status panel ===== */
        .panel-right {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        /* ===== Common card style ===== */
        .widget {
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            border: 1px solid var(--border);
            box-shadow: var(--shadow-md);
            overflow: hidden;
        }

        .widget-header {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 10px;
            background: var(--bg-elevated);
        }

        .widget-icon {
            width: 32px;
            height: 32px;
            background: var(--gradient-soft);
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
        }

        .widget-header h3 {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .widget-body { padding: 14px 16px; }

        /* Vehicle status grid */
        .status-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }

        .status-item {
            background: var(--bg-warm);
            border-radius: var(--radius-sm);
            padding: 12px;
            border: 1px solid var(--border);
            transition: all 0.25s ease;
        }

        .status-item:hover {
            border-color: var(--accent-medium);
            background: var(--accent-soft);
        }

        .status-label {
            font-size: 0.65rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .status-value {
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .status-value.active { color: var(--success); }
        .status-value.inactive { color: var(--text-muted); }

        /* Battery gauge */
        .battery-gauge {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 4px 0;
        }

        .gauge-visual {
            position: relative;
            width: 64px;
            height: 64px;
        }

        .gauge-circle {
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background: conic-gradient(
                var(--success) calc(var(--percent, 78) * 3.6deg),
                var(--bg-warm) calc(var(--percent, 78) * 3.6deg)
            );
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .gauge-inner {
            width: 48px;
            height: 48px;
            background: var(--bg-card);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .gauge-info {
            flex: 1;
        }

        .gauge-info .label {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-bottom: 2px;
        }

        .gauge-info .value {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .gauge-info .sub {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 2px;
        }

        /* Quick actions */
        .quick-actions {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }

        .quick-btn {
            background: var(--bg-warm);
            border: 1.5px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 10px 12px;
            color: var(--text-primary);
            font-size: 0.78rem;
            font-family: inherit;
            font-weight: 500;
            cursor: pointer;
            text-align: center;
            transition: all 0.25s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
        }

        .quick-btn:hover {
            background: var(--accent-soft);
            border-color: var(--accent);
            color: var(--accent);
            transform: translateY(-2px);
        }

        .quick-btn .icon {
            font-size: 1.2rem;
        }

        /* Voice settings */
        .voice-settings {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .voice-toggle {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 12px;
            background: var(--bg-warm);
            border-radius: var(--radius-sm);
            border: 1px solid var(--border);
        }

        .voice-toggle-label {
            font-size: 0.8rem;
            color: var(--text-primary);
            font-weight: 500;
        }

        .toggle-switch {
            position: relative;
            width: 44px;
            height: 24px;
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
            border-radius: 24px;
            transition: 0.3s;
        }

        .toggle-slider::before {
            content: '';
            position: absolute;
            width: 18px;
            height: 18px;
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
            transform: translateX(20px);
        }

        /* ===== Bottom control bar ===== */
        .control-bar {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            padding: 12px 20px;
            background: var(--bg-glass);
            backdrop-filter: blur(20px);
            border-radius: var(--radius-lg);
            border: 1px solid var(--border);
        }

        .control-btn {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 4px;
            padding: 10px 20px;
            background: var(--bg-warm);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            cursor: pointer;
            transition: all 0.25s ease;
            min-width: 80px;
        }

        .control-btn:hover {
            background: var(--accent-soft);
            border-color: var(--accent);
        }

        .control-btn .icon {
            font-size: 1.3rem;
        }

        .control-btn .label {
            font-size: 0.7rem;
            color: var(--text-secondary);
            font-weight: 500;
        }

        .control-btn.active {
            background: var(--accent-soft);
            border-color: var(--accent);
        }

        .control-btn.active .label {
            color: var(--accent);
        }

        /* ===== Voice assistant overlay ===== */
        .voice-overlay {
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(12px);
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
            padding: 40px;
            text-align: center;
            max-width: 400px;
            width: 90%;
            box-shadow: var(--shadow-lg);
            transform: scale(0.9);
            transition: transform 0.3s ease;
        }

        .voice-overlay.active .voice-assistant {
            transform: scale(1);
        }

        .voice-avatar {
            width: 100px;
            height: 100px;
            margin: 0 auto 20px;
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
            inset: -6px;
            border-radius: 50%;
            border: 3px solid var(--accent);
            opacity: 0;
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
            font-size: 1.3rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 6px;
        }

        .voice-hint {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 20px;
        }

        .voice-transcript {
            background: var(--bg-warm);
            border-radius: var(--radius-md);
            padding: 14px 18px;
            min-height: 50px;
            font-size: 0.95rem;
            color: var(--text-primary);
            margin-bottom: 20px;
            text-align: left;
        }

        .voice-transcript:empty::before {
            content: 'Waiting for voice input...';
            color: var(--text-muted);
        }

        .voice-waveform {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            height: 40px;
            margin-bottom: 20px;
        }

        .voice-waveform span {
            width: 4px;
            height: 20px;
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
            0%, 100% { height: 10px; }
            50% { height: 36px; }
        }

        .voice-cancel {
            background: var(--bg-warm);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 10px 28px;
            color: var(--text-secondary);
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .voice-cancel:hover {
            background: var(--danger-soft);
            border-color: var(--danger);
            color: var(--danger);
        }

        /* Wake word indicator */
        .wake-indicator {
            position: fixed;
            bottom: 100px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 100px;
            padding: 10px 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.8rem;
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
            width: 18px;
            height: 18px;
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
    </style>
</head>
<body>
    <div class="cockpit">
        <!-- Top HUD -->
        <header class="hud-bar">
            <div class="brand">
                <div class="brand-logo">
                    <img src="/static/brand_logo.gif" alt="Friday Logo">
                </div>
                <div class="brand-text">
                    <span>Intelligent Cockpit System</span>
                </div>
            </div>
            
            <div class="hud-indicators">
                <div class="hud-item">
                    <div class="hud-value success" id="batteryStatus">78%</div>
                    <div class="hud-label">Battery</div>
                </div>
                <div class="hud-item">
                    <div class="hud-value" id="rangeStatus">320</div>
                    <div class="hud-label">Range km</div>
                </div>
                <div class="hud-item">
                    <div class="hud-value" id="acTemp">24°</div>
                    <div class="hud-label">Cabin Temp</div>
                </div>
            </div>
            
            <div class="connection-badge" id="connectionStatus">
                <div class="connection-dot"></div>
                <span>Connecting...</span>
            </div>
        </header>

        <!-- Main dashboard -->
        <main class="dashboard">
            <!-- Left panel -->
            <div class="panel-left">
                <div class="widget">
                    <div class="widget-header">
                        <div class="widget-icon">🚗</div>
                        <h3>Vehicle Control</h3>
                    </div>
                    <div class="widget-body">
                        <div class="status-grid">
                            <div class="status-item">
                                <div class="status-label">AC</div>
                                <div class="status-value inactive" id="acStatus">Off</div>
                            </div>
                            <div class="status-item">
                                <div class="status-label">Music</div>
                                <div class="status-value inactive" id="musicStatus">Stopped</div>
                            </div>
                            <div class="status-item">
                                <div class="status-label">Navigation</div>
                                <div class="status-value inactive" id="navStatus">Inactive</div>
                            </div>
                            <div class="status-item">
                                <div class="status-label">Windows</div>
                                <div class="status-value inactive" id="windowStatus">Closed</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="widget">
                    <div class="widget-header">
                        <div class="widget-icon">🔋</div>
                        <h3>Energy Status</h3>
                    </div>
                    <div class="widget-body">
                        <div class="battery-gauge">
                            <div class="gauge-visual">
                                <div class="gauge-circle" id="batteryGauge" style="--percent: 78">
                                    <div class="gauge-inner" id="batteryPercent">78%</div>
                                </div>
                            </div>
                            <div class="gauge-info">
                                <div class="label">Battery Level</div>
                                <div class="value" id="batteryKwh">58.5 kWh</div>
                                <div class="sub">Est. range <span id="rangeKm">320</span> km</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="widget">
                    <div class="widget-header">
                        <div class="widget-icon">🎤</div>
                        <h3>Voice Settings</h3>
                    </div>
                    <div class="widget-body">
                        <div class="voice-settings">
                            <div class="voice-toggle">
                                <span class="voice-toggle-label">Wake Word</span>
                                <label class="toggle-switch">
                                    <input type="checkbox" id="wakeWordToggle" onchange="toggleWakeWord(this.checked)">
                                    <span class="toggle-slider"></span>
                                </label>
                            </div>
                            <div class="voice-toggle">
                                <span class="voice-toggle-label">Voice Response</span>
                                <label class="toggle-switch">
                                    <input type="checkbox" id="ttsToggle" checked onchange="toggleTTS(this.checked)">
                                    <span class="toggle-slider"></span>
                                </label>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Center console -->
            <section class="center-console">
                <div class="console-header">
                    <div class="assistant-orb">
                        <img src="/static/assistant_avatar.png" alt="Friday">
                    </div>
                    <div class="assistant-info">
                        <h2>Friday Assistant</h2>
                        <p>Online · Ready to serve</p>
                    </div>
                </div>

                <div class="chat-messages" id="chatMessages">
                    <div class="message assistant">
                        <div class="message-meta">Friday</div>
                        <div class="message-bubble">Hello! I'm Friday, your intelligent cockpit assistant. You can communicate with me via text or voice, or say "Hey Friday" to wake me up. Would you like me to adjust the cabin environment, plan a route, or play some music?</div>
                    </div>
                </div>

                <div class="typing-indicator" id="typingIndicator">
                    <span></span><span></span><span></span>
                </div>

                <div class="console-input">
                    <div class="input-row">
                        <input type="text" class="input-field" id="userInput" placeholder="Enter command or question..." autocomplete="off">
                        <div class="btn-group">
                            <button class="voice-btn" id="voiceBtn" onclick="toggleVoice()" title="Voice input">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                                    <line x1="12" y1="19" x2="12" y2="23"></line>
                                    <line x1="8" y1="23" x2="16" y2="23"></line>
                                </svg>
                            </button>
                            <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                                Send
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                    <line x1="22" y1="2" x2="11" y2="13"></line>
                                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>
            </section>

            <!-- Right panel -->
            <div class="panel-right">
                <div class="widget">
                    <div class="widget-header">
                        <div class="widget-icon">⚡</div>
                        <h3>Quick Actions</h3>
                    </div>
                    <div class="widget-body">
                        <div class="quick-actions">
                            <button class="quick-btn" onclick="quickSend('Turn on the AC')">
                                <span class="icon">❄️</span>
                                AC
                            </button>
                            <button class="quick-btn" onclick="quickSend('Play music')">
                                <span class="icon">🎵</span>
                                Music
                            </button>
                            <button class="quick-btn" onclick="quickSend('Open all windows')">
                                <span class="icon">🪟</span>
                                Windows
                            </button>
                            <button class="quick-btn" onclick="quickSend('Navigate to nearest charging station')">
                                <span class="icon">🔋</span>
                                Charging
                            </button>
                            <button class="quick-btn" onclick="quickSend('Check vehicle status')">
                                <span class="icon">📊</span>
                                Status
                            </button>
                            <button class="quick-btn" onclick="quickSend('Set temperature to 22 degrees')">
                                <span class="icon">🌡️</span>
                                Temp
                            </button>
                        </div>
                    </div>
                </div>

                <div class="widget" style="flex: 1;">
                    <div class="widget-header">
                        <div class="widget-icon">📍</div>
                        <h3>Navigation</h3>
                    </div>
                    <div class="widget-body">
                        <div style="text-align: center; padding: 20px 0; color: var(--text-muted); font-size: 0.85rem;">
                            <div style="font-size: 2rem; margin-bottom: 8px;">🗺️</div>
                            No active navigation<br>
                            <span style="font-size: 0.75rem;">Say "Navigate to..." to start</span>
                        </div>
                    </div>
                </div>
            </div>
        </main>

        <!-- Bottom control bar -->
        <footer class="control-bar">
            <button class="control-btn" onclick="quickSend('Turn on AC')">
                <span class="icon">❄️</span>
                <span class="label">AC</span>
            </button>
            <button class="control-btn" onclick="quickSend('Play music')">
                <span class="icon">🎵</span>
                <span class="label">Media</span>
            </button>
            <button class="control-btn" onclick="quickSend('Start navigation')">
                <span class="icon">🧭</span>
                <span class="label">Nav</span>
            </button>
            <button class="control-btn" onclick="quickSend('Check vehicle status')">
                <span class="icon">🚗</span>
                <span class="label">Vehicle</span>
            </button>
            <button class="control-btn" onclick="quickSend('Open windows')">
                <span class="icon">🪟</span>
                <span class="label">Windows</span>
            </button>
        </footer>
    </div>

    <!-- Voice assistant overlay -->
    <div class="voice-overlay" id="voiceOverlay">
        <div class="voice-assistant">
            <div class="voice-avatar">
                <img src="/static/assistant_avatar.png" alt="Friday">
            </div>
            <div class="voice-status" id="voiceStatus">Listening...</div>
            <div class="voice-hint" id="voiceHint">Please speak your command</div>
            <div class="voice-waveform" id="voiceWaveform">
                <span></span><span></span><span></span><span></span><span></span>
            </div>
            <div class="voice-transcript" id="voiceTranscript"></div>
            <button class="voice-cancel" onclick="cancelVoice()">Cancel</button>
        </div>
    </div>

    <!-- Wake word indicator -->
    <div class="wake-indicator" id="wakeIndicator" style="display: none;">
        <div class="mic-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="23"></line>
                <line x1="8" y1="23" x2="16" y2="23"></line>
            </svg>
        </div>
        <span>Say "Hey Friday" to wake</span>
    </div>

    <script>
        // ===== Global state =====
        let ws = null;
        let isGenerating = false;
        let currentAssistantMessage = null;
        
        // Voice-related state
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let wakeWordEnabled = false;
        let ttsEnabled = true;
        let recognition = null;
        
        // ★★★ New: Recording time tracking, fixes first recording WebM header incomplete issue ★★★
        let recordingStartTime = 0;
        const MIN_RECORDING_TIME = 800;  // Minimum recording time 800ms, ensures WebM header is complete
        
        // ===== WebSocket connection =====
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                const badge = document.getElementById('connectionStatus');
                badge.classList.add('connected');    
                badge.querySelector('span').textContent = 'Connected';
            };

            ws.onclose = () => {
                const badge = document.getElementById('connectionStatus');
                badge.classList.remove('connected');
                badge.querySelector('span').textContent = 'Disconnected';
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
            messageDiv.innerHTML = `<div class="message-meta">${role === 'user' ? 'You' : 'Friday'}</div><div class="message-bubble">${content}</div>`;
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
                acEl.textContent = status.ac.on ? 'Running' : 'Off';
                acEl.className = 'status-value ' + (status.ac.on ? 'active' : 'inactive');
                // Update HUD temperature display
                const tempEl = document.getElementById('acTemp');
                if (tempEl) tempEl.textContent = status.ac.temperature + '°';
            }
            if (status.navigation) {
                const navEl = document.getElementById('navStatus');
                navEl.textContent = status.navigation.active ? (status.navigation.destination || 'Navigating') : 'Inactive';
                navEl.className = 'status-value ' + (status.navigation.active ? 'active' : 'inactive');
            }
            if (status.music) {
                const musicEl = document.getElementById('musicStatus');
                musicEl.textContent = status.music.playing ? 'Playing' : 'Stopped';
                musicEl.className = 'status-value ' + (status.music.playing ? 'active' : 'inactive');
            }
            if (status.battery !== undefined) {
                // HUD battery display
                const batteryStatus = document.getElementById('batteryStatus');
                if (batteryStatus) batteryStatus.textContent = status.battery + '%';
                
                // Battery gauge
                const batteryGauge = document.getElementById('batteryGauge');
                if (batteryGauge) batteryGauge.style.setProperty('--percent', status.battery);
                
                const batteryPercent = document.getElementById('batteryPercent');
                if (batteryPercent) batteryPercent.textContent = status.battery + '%';
                
                // Battery kWh (assuming 75kWh total capacity)
                const batteryKwh = document.getElementById('batteryKwh');
                if (batteryKwh) batteryKwh.textContent = (status.battery * 0.75).toFixed(1) + ' kWh';
            }
            if (status.range !== undefined) {
                // HUD range display
                const rangeStatus = document.getElementById('rangeStatus');
                if (rangeStatus) rangeStatus.textContent = status.range;
                
                // Detailed range display
                const rangeKm = document.getElementById('rangeKm');
                if (rangeKm) rangeKm.textContent = status.range;
            }
        }

        // ===== Voice functions (fixed version - solves first recording WebM header incomplete issue) =====
        
        async function toggleVoice() {
            if (isRecording) {
                stopRecording();
            } else {
                await startRecording();
            }
        }

        /**
         * Use OfflineAudioContext for high-quality resampling
         * Key: Browser automatically performs anti-aliasing filter, avoids frequency aliasing
         */
        async function resampleWithOfflineContext(audioBuffer, targetSampleRate) {
            const sourceSampleRate = audioBuffer.sampleRate;
            
            if (sourceSampleRate === targetSampleRate) {
                return audioBuffer.getChannelData(0);
            }
            
            const duration = audioBuffer.duration;
            const targetLength = Math.round(duration * targetSampleRate);
            
            // Create offline context for resampling (automatic low-pass filter)
            const offlineCtx = new OfflineAudioContext(1, targetLength, targetSampleRate);
            
            const source = offlineCtx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(offlineCtx.destination);
            source.start(0);
            
            const renderedBuffer = await offlineCtx.startRendering();
            
            console.log(`OfflineAudioContext resampling: ${sourceSampleRate}Hz -> ${targetSampleRate}Hz, ${renderedBuffer.length} samples`);
            
            return renderedBuffer.getChannelData(0);
        }

        /**
         * ★★★ Fixed start recording function ★★★
         * Key fix: Don't use timeslice parameter, let browser manage data chunks
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
                
                // Show voice overlay
                const overlay = document.getElementById('voiceOverlay');
                overlay.classList.add('active', 'listening');
                document.getElementById('voiceStatus').textContent = 'Listening...';
                document.getElementById('voiceHint').textContent = 'Please speak your command';
                document.getElementById('voiceTranscript').textContent = '';
                
                const voiceBtn = document.getElementById('voiceBtn');
                voiceBtn.classList.add('listening');
                
                // Use MediaRecorder for recording (no frame drops)
                audioChunks = [];
                
                // Select supported format
                let mimeType = 'audio/webm;codecs=opus';
                if (!MediaRecorder.isTypeSupported(mimeType)) {
                    mimeType = 'audio/webm';
                }
                if (!MediaRecorder.isTypeSupported(mimeType)) {
                    mimeType = 'audio/mp4';
                }
                console.log('MediaRecorder format:', mimeType);
                
                mediaRecorder = new MediaRecorder(stream, { 
                    mimeType: mimeType,
                    audioBitsPerSecond: 128000
                });
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                        console.log(`Received audio chunk: ${event.data.size} bytes, total ${audioChunks.length} chunks`);
                    }
                };
                
                mediaRecorder.onstop = async () => {
                    console.log(`Recording stopped, total ${audioChunks.length} audio chunks`);
                    stream.getTracks().forEach(track => track.stop());
                    await processRecordedAudio(mimeType);
                };
                
                // ★★★ Key fix: Don't use timeslice parameter, let browser manage data chunks ★★★
                // Previously was mediaRecorder.start(250); which caused data fragmentation, first recording header incomplete
                mediaRecorder.start();
                
                // ★★★ Record start time ★★★
                recordingStartTime = Date.now();
                console.log('Recording start time:', recordingStartTime);
                
                window.currentStream = stream;
                isRecording = true;
                
                // Auto-stop after 8 seconds
                window.recordingTimeout = setTimeout(() => {
                    if (isRecording) {
                        stopRecording();
                    }
                }, 8000);
                
            } catch (error) {
                console.error('Cannot access microphone:', error);
                alert('Cannot access microphone, please check permissions');
            }
        }

        /**
         * ★★★ Fixed stop recording function ★★★
         * Key fix: Add minimum recording time protection, ensure WebM header is complete
         */
        function stopRecording() {
            clearTimeout(window.recordingTimeout);
            
            // ★★★ Calculate recording duration ★★★
            const recordingDuration = Date.now() - recordingStartTime;
            console.log(`Recording duration: ${recordingDuration}ms`);
            
            // ★★★ Key fix: If recording time too short, wait before stopping ★★★
            if (recordingDuration < MIN_RECORDING_TIME) {
                const waitTime = MIN_RECORDING_TIME - recordingDuration;
                console.log(`Recording time insufficient, waiting ${waitTime}ms`);
                
                document.getElementById('voiceStatus').textContent = 'Processing...';
                
                setTimeout(() => {
                    doStopRecording();
                }, waitTime);
            } else {
                doStopRecording();
            }
        }
        
        /**
         * ★★★ New: Actually execute stop recording ★★★
         */
        function doStopRecording() {
            isRecording = false;
            
            const voiceBtn = document.getElementById('voiceBtn');
            voiceBtn.classList.remove('listening');
            voiceBtn.classList.add('processing');
            
            document.getElementById('voiceStatus').textContent = 'Processing...';
            document.getElementById('voiceOverlay').classList.remove('listening');
            
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                // ★★★ Key fix: Request all pending data before stopping ★★★
                try {
                    mediaRecorder.requestData();
                } catch (e) {
                    console.log('requestData not supported or no data');
                }
                
                // Delay stop slightly to ensure requestData completes
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
         * ★★★ Fixed process recorded audio function ★★★
         * Key fix: Add data size check
         */
        async function processRecordedAudio(mimeType) {
            if (audioChunks.length === 0) {
                console.warn('No audio data recorded');
                handleASRResult('');
                return;
            }
            
            try {
                // Merge audio chunks
                const audioBlob = new Blob(audioChunks, { type: mimeType });
                console.log('Recording complete, size:', audioBlob.size, 'bytes, chunks:', audioChunks.length);
                
                // ★★★ Key fix: Check data size (WebM header usually needs at least a few hundred bytes) ★★★
                if (audioBlob.size < 500) {
                    console.warn('Audio data too small, may be incomplete, trying to send raw data');
                    await sendRawAudio(audioBlob, mimeType);
                    return;
                }
                
                // Decode to AudioBuffer
                const arrayBuffer = await audioBlob.arrayBuffer();
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                
                let audioBuffer;
                try {
                    audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                } catch (decodeError) {
                    console.error('Browser decode failed:', decodeError);
                    console.log('Trying to send raw data for backend processing...');
                    // If decode fails, send raw data for backend processing
                    await sendRawAudio(audioBlob, mimeType);
                    audioContext.close();
                    return;
                }
                
                console.log('Decode successful:', audioBuffer.sampleRate, 'Hz,', audioBuffer.duration.toFixed(2), 'seconds');
                
                // Use OfflineAudioContext to resample to 16kHz (key: automatic low-pass filter)
                const targetSampleRate = 16000;
                const resampledData = await resampleWithOfflineContext(audioBuffer, targetSampleRate);
                
                // Encode to WAV
                const wavBlob = encodeWAV(resampledData, targetSampleRate);
                console.log('WAV encoding complete:', wavBlob.size, 'bytes');
                
                // Send to server
                await sendAudioForRecognition(wavBlob);
                
                audioContext.close();
                
            } catch (error) {
                console.error('Audio processing error:', error);
                handleASRResult('');
            }
        }

        /**
         * Send raw audio (when decodeAudioData fails)
         */
        async function sendRawAudio(audioBlob, mimeType) {
            const reader = new FileReader();
            reader.onloadend = () => {
                const base64Audio = reader.result.split(',')[1];
                if (ws && ws.readyState === WebSocket.OPEN) {
                    const format = mimeType.includes('webm') ? 'webm' : 'mp4';
                    console.log('Sending raw audio, format:', format, ', size:', audioBlob.size);
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
         * Encode to WAV format
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
            
            // Write sample data
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
                    console.log('Sending WAV to server, base64 length:', base64Audio.length);
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
                document.getElementById('voiceStatus').textContent = 'Recognized';
                
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
                document.getElementById('voiceStatus').textContent = 'No speech detected';
                setTimeout(() => {
                    document.getElementById('voiceOverlay').classList.remove('active');
                }, 1500);
            }
        }

        /**
         * ★★★ Fixed cancel voice function ★★★
         * Key fix: Don't trigger onstop processing when cancelling
         */
        function cancelVoice() {
            clearTimeout(window.recordingTimeout);
            
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                // ★★★ Don't trigger onstop processing when cancelling ★★★
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
                console.error('TTS audio playback failed:', error);
            }
        }

        // ===== Wake word detection =====
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
                console.warn('Browser does not support speech recognition');
                alert('Your browser does not support speech recognition. Please use Chrome browser');
                document.getElementById('wakeWordToggle').checked = false;
                document.getElementById('wakeIndicator').style.display = 'none';
                return;
            }
            
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';
            
            recognition.onresult = (event) => {
                const indicator = document.getElementById('wakeIndicator');
                
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript.toLowerCase();
                    console.log('Recognized:', transcript);
                    
                    if (transcript.includes('friday') || 
                        transcript.includes('hey friday') || 
                        transcript.includes('hi friday') ||
                        transcript.includes('hello friday')) {
                        
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
                console.error('Wake word detection error:', event.error);
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
                                console.log('Restarting wake word detection');
                            }
                        }
                    }, 500);
                }
            };
            
            try {
                recognition.start();
                console.log('Wake word detection started');
            } catch (e) {
                console.error('Failed to start wake word detection:', e);
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


# ===== ASR Engine =====
class ASREngine:
    """Speech recognition engine"""
    
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
            return "Speech recognition unavailable"
        
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
            language="en",
            beam_size=5
        )
        
        text = "".join([segment.text for segment in segments])
        return text.strip()


# ===== TTS Engine =====
class TTSEngine:
    """Text-to-speech engine"""
    
    def __init__(self, voice: str = "en-US-AvaNeural"):
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


# ===== Audio processing (fixed version) =====

def resample_audio_scipy(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """High-quality resampling using scipy (with anti-aliasing filter)"""
    if from_rate == to_rate:
        return audio
    
    # Calculate resampling parameters
    gcd = np.gcd(from_rate, to_rate)
    up = to_rate // gcd
    down = from_rate // gcd
    
    # scipy.signal.resample_poly automatically performs anti-aliasing filter
    resampled = signal.resample_poly(audio, up, down)
    
    return resampled.astype(np.float32)


def resample_audio_linear(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Simple linear interpolation resampling (fallback, lower quality)"""
    if from_rate == to_rate:
        return audio
    
    ratio = from_rate / to_rate
    new_length = int(len(audio) / ratio)
    
    old_indices = np.arange(len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)
    
    resampled = np.interp(new_indices, old_indices, audio)
    return resampled.astype(np.float32)


def convert_wav_to_numpy(wav_data: bytes) -> np.ndarray:
    """Convert WAV audio data to numpy array (fixed version: add sample rate validation and resampling)"""
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
        
        logger.info(f"WAV params: {sample_rate}Hz, {n_channels}ch, {sample_width*8}bit, {n_frames} frames")
        
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
        
        # Key fix: If sample rate is not 16kHz, perform high-quality resampling
        if sample_rate != 16000:
            logger.info(f"Backend resampling: {sample_rate}Hz -> 16000Hz")
            if HAS_SCIPY:
                audio_float = resample_audio_scipy(audio_float, sample_rate, 16000)
                logger.info("Using scipy high-quality resampling")
            else:
                audio_float = resample_audio_linear(audio_float, sample_rate, 16000)
                logger.warning("Using linear interpolation resampling (recommend installing scipy: pip install scipy)")
        
        logger.info(f"Conversion complete: {len(audio_float)} samples ({len(audio_float)/16000:.2f} seconds)")
        return audio_float
        
    except Exception as e:
        logger.error(f"WAV conversion error: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])


def convert_webm_to_numpy(webm_data: bytes) -> np.ndarray:
    """Convert WebM audio to numpy array (requires pydub + ffmpeg)"""
    try:
        from pydub import AudioSegment
        
        audio_io = io.BytesIO(webm_data)
        audio = AudioSegment.from_file(audio_io, format='webm')
        
        # Convert to mono 16kHz
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Convert to numpy
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0
        
        logger.info(f"WebM conversion complete: {len(samples)} samples ({len(samples)/16000:.2f} seconds)")
        return samples
        
    except ImportError:
        logger.error("Need pydub to process WebM: pip install pydub")
        logger.error("Also need to install ffmpeg")
        return np.array([])
    except Exception as e:
        logger.error(f"WebM conversion error: {e}")
        return np.array([])


def clean_text_for_tts(text: str) -> str:
    """Clean text for TTS"""
    cleaned = re.sub(r'\{[^}]+\}', '', text)
    cleaned = re.sub(r'[✅❌🔧📊🔋🛞🛢️📍🌡️🎵🪟❄️⚡]', '', cleaned)
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()


# ===== FastAPI application =====
if HAS_FASTAPI:
    app = FastAPI(title="Intelligent Cockpit Assistant - Voice Enhanced (Fixed)")
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
                            await websocket.send_json({"type": "token", "content": f"Error: {str(e)}"})
                            await websocket.send_json({"type": "end"})
                    else:
                        await websocket.send_json({"type": "token", "content": "Assistant not initialized"})
                        await websocket.send_json({"type": "end"})

                elif data.get("type") == "audio":
                    audio_base64 = data.get("audio", "")
                    audio_format = data.get("format", "wav")
                    
                    if audio_base64 and asr_engine:
                        try:
                            audio_bytes = base64.b64decode(audio_base64)
                            logger.info(f"Received audio: {len(audio_bytes)} bytes, format: {audio_format}")
                            
                            # Select conversion method based on format
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
                            "text": "Speech recognition unavailable"
                        })

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")


def main(model_path: str, host: str = "0.0.0.0", port: int = 8000,
         n_ctx: int = 4096, n_gpu_layers: int = 0,
         asr_model: str = "small", tts_voice: str = "en-US-AvaNeural"):
    global assistant, asr_engine, tts_engine

    if not HAS_FASTAPI:
        print("Please install dependencies: pip install fastapi uvicorn")
        return

    print("=" * 60)
    print("  🚗 Friday Intelligent Cockpit Assistant")
    print("=" * 60)
    print(f"\nLoading model: {model_path}")

    try:
        assistant = CockpitAssistant(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
    except Exception as e:
        print(f"⚠ Warning: Model loading failed ({e}), using mock mode")
        assistant = CockpitAssistant("mock_model.gguf")

    if HAS_WHISPER:
        asr_engine = ASREngine(model_size=asr_model, device="cuda" if n_gpu_layers > 0 else "cpu")
        try:
            asr_engine.initialize()
        except Exception as e:
            print(f"⚠ ASR model loading failed: {e}")
    else:
        print("\n⚠ ASR unavailable (please install faster-whisper)")
    
    if HAS_EDGE_TTS:
        tts_engine = TTSEngine(voice=tts_voice)
    else:
        print("⚠ TTS unavailable (please install edge-tts)")

    print(f"\nStarting web server: http://{host}:{port}")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligent Cockpit Assistant - Friday")
    parser.add_argument("model_path", nargs="?", default="models/qwen2.5-7b-instruct-q4_k_m.gguf", help="Model file path")
    parser.add_argument("--host", default="0.0.0.0", help="Server address")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("-c", "--ctx", type=int, default=4096, help="Context length")
    parser.add_argument("-g", "--gpu-layers", type=int, default=0, help="Number of GPU layers")
    parser.add_argument("--asr-model", default="small", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper ASR model size")
    parser.add_argument("--tts-voice", default="en-US-AvaNeural")
    args = parser.parse_args()
    main(args.model_path, args.host, args.port, args.ctx, args.gpu_layers, 
         args.asr_model, args.tts_voice)