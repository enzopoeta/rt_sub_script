# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, colorchooser, font, messagebox
import threading
import queue
import json
import os
import tempfile
import time
import io
from collections import deque


# --- Dependências de Áudio e VAD ---
import numpy as np
import soundcard as sc
from scipy.io.wavfile import write as write_wav
from scipy.signal import butter, lfilter
import webrtcvad

# --- Dependência de Rede ---
import requests

# --- Dependências Opcionais para o Logo SVG ---
try:
    from PIL import Image, ImageTk
    import cairosvg
    SVG_SUPPORT = True
except ImportError:
    SVG_SUPPORT = False
    print("AVISO: Para exibir o logo SVG, instale 'Pillow' e 'CairoSVG'.")
    print("Execute: pip install Pillow CairoSVG")

# --- Constantes e Funções Auxiliares ---
CONFIG_FILE = "config.json"

def find_default_loopback_device():
    """Encontra o dispositivo de loopback padrão de forma robusta."""
    try:
        default_speaker = sc.default_speaker()
        all_mics = sc.all_microphones(include_loopback=True)
        # Tenta encontrar por nome correspondente primeiro
        for mic in all_mics:
            if mic.is_loopback and default_speaker.name in mic.name:
                return mic
        # Se não encontrar, retorna o primeiro loopback que achar
        for mic in all_mics:
            if mic.is_loopback:
                return mic
    except Exception as e:
        print(f"Não foi possível encontrar o dispositivo de loopback padrão: {e}")
    return None

# --- Classes de Lógica de Negócio ---

class ConfigManager:
    """Gerencia o carregamento e salvamento das configurações em config.json."""
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(CONFIG_FILE):
            print(f"Arquivo '{CONFIG_FILE}' não encontrado. Criando com valores padrão.")
            default_config = {
                "webservice_settings": {"url": "http://localhost:8080/inference", "file_form_name": "file", "response_type": "json", "json_response_key": "text", "temperature": "0.0", "temperature_inc": "0.2", "response_format_param": "json"},
                "translation_settings": {"enabled": False, "ollama_host": "localhost", "ollama_port": "11434", "ollama_model": "llama3", "system_prompt": "You are a helpful and expert translator. Translate the following user-provided text to Brazilian Portuguese. Only provide the translated text, without any additional comments, preambles, or explanations."},
                "audio_settings": {"device_index": "default_loopback", "vad_aggressiveness": 3, "vad_trigger_ratio": 0.75, "vad_silence_ms": 700},
                "audio_processing_settings": {"enable_high_pass": True, "high_pass_cutoff_hz": 100.0, "enable_normalization": True, "min_segment_duration_ms": 250},
                "overlay_settings": {"font_family": "Arial", "font_size": 24, "font_color": "white", "bg_color": "black", "alpha": 0.75, "position_x": 100, "position_y": 800, "min_display_ms": 4000, "merge_consecutive_ms": 1500}
            }
            self.save_config(default_config)
            return default_config
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
            # Adiciona chaves faltantes para compatibilidade com versões antigas
            config.setdefault("audio_processing_settings", { "enable_high_pass": True, "high_pass_cutoff_hz": 100.0, "enable_normalization": True, "min_segment_duration_ms": 250 })
            config.setdefault("overlay_settings", {}).setdefault("min_display_ms", 4000)
            config["overlay_settings"].setdefault("merge_consecutive_ms", 1500)
            config.setdefault("audio_settings", {}).setdefault("device_index", "default_loopback")
            config.setdefault("translation_settings", {"enabled": False, "ollama_host": "localhost", "ollama_port": "11434", "ollama_model": "llama3", "system_prompt": "You are a helpful and expert translator. Translate the following user-provided text to Brazilian Portuguese. Only provide the translated text, without any additional comments, preambles, or explanations."})
            return config

    def save_config(self, new_config):
        self.config = new_config
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def get(self, key, default=None):
        return self.config.get(key, default)

class AudioRecorder(threading.Thread):
    def __init__(self, audio_queue, config_manager, status_callback):
        super().__init__(daemon=True)
        self.audio_queue = audio_queue; self.config_manager = config_manager; self.status_callback = status_callback; self._stop_event = threading.Event(); self.device = None
        proc_settings = self.config_manager.get("audio_processing_settings", {})
        self.enable_filter = proc_settings.get("enable_high_pass", True); self.filter_cutoff = proc_settings.get("high_pass_cutoff_hz", 100.0); self.enable_normalize = proc_settings.get("enable_normalization", True); self.min_duration_ms = proc_settings.get("min_segment_duration_ms", 250)
    def high_pass_filter(self, data, cutoff, fs, order=5): nyq = 0.5 * fs; normal_cutoff = cutoff / nyq; b, a = butter(order, normal_cutoff, btype='high', analog=False); y = lfilter(b, a, data); return y
    def normalize_audio(self, data): peak = np.abs(data).max(); return data / peak * 0.95 if peak > 1e-5 else data
    def preprocess_audio(self, audio_data, samplerate):
        min_samples = int((self.min_duration_ms / 1000.0) * samplerate)
        if len(audio_data) < min_samples: self.status_callback(f"Segmento muito curto ({len(audio_data)/samplerate:.2f}s). Descartando.", "blue"); return None
        processed_data = audio_data.copy()
        if self.enable_filter: processed_data = self.high_pass_filter(processed_data, self.filter_cutoff, samplerate)
        if self.enable_normalize: processed_data = self.normalize_audio(processed_data)
        return processed_data.astype(np.float32)
    def run(self):
        while not self._stop_event.is_set():
            try: self.start_recording()
            except Exception as e: import traceback; traceback.print_exc(); self.status_callback(f"Erro no áudio: {e}", "red"); time.sleep(5)
    def start_recording(self):
        settings = self.config_manager.get("audio_settings", {}); device_index = settings.get("device_index", "default_loopback"); aggressiveness = int(settings.get("vad_aggressiveness", 3)); silence_duration_ms = int(settings.get("vad_silence_ms", 700)); trigger_ratio = float(settings.get("vad_trigger_ratio", 0.75)); samplerate = 16000; frame_duration_ms = 10; frame_size = int(samplerate * frame_duration_ms / 1000)
        if not webrtcvad.valid_rate_and_frame_length(samplerate, frame_size): raise RuntimeError(f"Taxa de amostragem/frame inválida para webrtcvad: {samplerate}/{frame_size}")
        try:
            if device_index == "default_mic": self.device = sc.default_microphone()
            elif device_index == "default_loopback":
                self.device = find_default_loopback_device()
                if not self.device: raise RuntimeError("Não foi possível encontrar um dispositivo de loopback padrão.")
            else: self.device = sc.get_microphone(id=device_index, include_loopback=True)
            self.status_callback(f"Gravando de: {self.device.name}", "blue"); vad = webrtcvad.Vad(aggressiveness)
            with self.device.recorder(samplerate=samplerate, blocksize=frame_size) as mic:
                self.status_callback("Ouvindo com VAD...", "green")
                self.record_with_vad(mic, vad, samplerate, frame_size, silence_duration_ms, trigger_ratio)
        except Exception as e: raise RuntimeError(f"Não foi possível iniciar a gravação VAD. Verifique o dispositivo. Detalhes: {e}")
    def record_with_vad(self, mic, vad, samplerate, frame_size, silence_ms, trigger_ratio):
        ms_per_frame = (frame_size * 1000) // samplerate; padding_duration_ms = 300; num_padding_frames = padding_duration_ms // ms_per_frame
        ring_buffer = deque(maxlen=num_padding_frames); trigger_window_size = 50; trigger_buffer = deque(maxlen=trigger_window_size); silence_frames_needed = silence_ms // ms_per_frame
        is_speaking = False; voiced_frames = []; silent_frames_count = 0
        while not self._stop_event.is_set():
            frame_data_float = mic.record(numframes=frame_size)
            if frame_data_float.ndim > 1: frame_data_float = frame_data_float.mean(axis=1)
            frame_data_int16 = (frame_data_float * 32767).astype(np.int16); frame_bytes = frame_data_int16.tobytes()
            try: is_speech = vad.is_speech(frame_bytes, samplerate)
            except Exception: continue
            trigger_buffer.append(is_speech)
            if not is_speaking:
                ring_buffer.append(frame_data_float); num_voiced_in_trigger = sum(trigger_buffer)
                if num_voiced_in_trigger > trigger_ratio * trigger_buffer.maxlen: self.status_callback("Detectando fala...", "orange"); is_speaking = True; voiced_frames.extend(ring_buffer); ring_buffer.clear(); silent_frames_count = 0
            else:
                voiced_frames.append(frame_data_float)
                if not is_speech: silent_frames_count += 1
                else: silent_frames_count = 0
                if silent_frames_count >= silence_frames_needed:
                    self.status_callback("Segmento detectado. Pré-processando...", "blue"); audio_segment = np.concatenate(voiced_frames); processed_segment = self.preprocess_audio(audio_segment, samplerate)
                    if processed_segment is not None: self.status_callback("Enviando segmento processado...", "blue"); self.audio_queue.put((processed_segment, samplerate))
                    is_speaking = False; voiced_frames.clear(); ring_buffer.clear(); trigger_buffer.clear(); self.status_callback("Ouvindo com VAD...", "green")
    def stop(self): self._stop_event.set()

class TranscriptionProcessor(threading.Thread):
    def __init__(self, audio_queue, transcribed_text_queue, config_manager, status_callback):
        super().__init__(daemon=True); self.audio_queue = audio_queue; self.transcribed_text_queue = transcribed_text_queue; self.config_manager = config_manager; self.status_callback = status_callback; self._stop_event = threading.Event()
    def run(self):
        while not self._stop_event.is_set():
            try:
                settings = self.config_manager.get("webservice_settings", {}); url = settings.get("url")
                if not url: self.status_callback("URL do Web Service não configurada!", "red"); time.sleep(5); continue
                audio_data, samplerate = self.audio_queue.get(timeout=1); self.status_callback("Enviando áudio para transcrição...", "orange")
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile: write_wav(tmpfile.name, samplerate, audio_data); tmpfile_path = tmpfile.name
                try:
                    payload = {"temperature": settings.get("temperature", "0.0"), "temperature_inc": settings.get("temperature_inc", "0.2"), "response_format": settings.get("response_format_param", "json")}
                    with open(tmpfile_path, "rb") as audio_file: files = {settings.get("file_form_name", "file"): audio_file}; response = requests.post(url, files=files, data=payload, timeout=30); response.raise_for_status()
                    transcribed_text = ""
                    if settings.get("response_type") == "json":
                        json_key = settings.get("json_response_key", "text"); json_response = response.json(); keys = json_key.split('.'); value = json_response
                        for key in keys:
                            if isinstance(value, dict): value = value.get(key)
                            else: value = None; break
                        transcribed_text = value
                    else: transcribed_text = response.text
                    if transcribed_text and transcribed_text.strip(): self.status_callback("Transcrição recebida. Aguardando processamento...", "blue"); self.transcribed_text_queue.put(transcribed_text.strip())
                    else: self.status_callback("Transcrição vazia recebida.", "blue"); self.status_callback("Ouvindo com VAD...", "green")
                except requests.exceptions.RequestException as e: self.status_callback(f"Erro de rede (Whisper): {e}", "red"); time.sleep(5)
                finally:
                    if os.path.exists(tmpfile_path): os.remove(tmpfile_path)
            except queue.Empty: continue
            except Exception as e: self.status_callback(f"Erro no processamento (Whisper): {e}", "red"); time.sleep(5)
    def stop(self): self._stop_event.set()

class TranslationProcessor(threading.Thread):
    def __init__(self, transcribed_text_queue, final_text_queue, config_manager, status_callback):
        super().__init__(daemon=True); self.input_queue = transcribed_text_queue; self.output_queue = final_text_queue; self.config_manager = config_manager; self.status_callback = status_callback; self._stop_event = threading.Event()
    def run(self):
        while not self._stop_event.is_set():
            try:
                text_to_process = self.input_queue.get(timeout=1)
                settings = self.config_manager.get("translation_settings", {}); is_enabled = settings.get("enabled", False)
                if not is_enabled: self.output_queue.put(text_to_process); self.status_callback("Ouvindo com VAD...", "green"); continue
                self.status_callback("Traduzindo com Ollama...", "purple")
                host = settings.get("ollama_host", "localhost"); port = settings.get("ollama_port", "11434"); model = settings.get("ollama_model", "llama3"); system_prompt = settings.get("system_prompt", "")
                if not all([host, port, model]): self.status_callback("Config. Ollama incompleta!", "red"); self.output_queue.put(f"[ERRO: Tradução não configurada] {text_to_process}"); time.sleep(5); continue
                api_url = f"http://{host}:{port}/api/generate"; payload = {"model": model, "system": system_prompt, "prompt": text_to_process, "stream": False}
                try:
                    response = requests.post(api_url, json=payload, timeout=45); response.raise_for_status()
                    response_json = response.json(); translated_text = response_json.get("response", "").strip()
                    if translated_text: self.output_queue.put(translated_text); self.status_callback("Ouvindo com VAD...", "green")
                    else: self.status_callback("Tradução vazia recebida.", "blue"); self.output_queue.put(f"[Trad. vazia] {text_to_process}")
                except requests.exceptions.RequestException as e: self.status_callback(f"Erro de rede (Ollama): {e}", "red"); self.output_queue.put(f"[ERRO: Ollama] {text_to_process}"); time.sleep(5)
            except queue.Empty: continue
            except Exception as e: self.status_callback(f"Erro na tradução: {e}", "red"); time.sleep(5)
    def stop(self): self._stop_event.set()

# --- Classes de Interface Gráfica (GUI) ---

class OverlayWindow(tk.Toplevel):
    def __init__(self, parent, final_text_queue, config_manager):
        super().__init__(parent); self.text_queue = final_text_queue; self.config_manager = config_manager
        self.min_display_ms = 4000; self.merge_consecutive_ms = 1500; self.last_text_update_time = 0; self.clear_timer_id = None; self.listening_indicator = "..."
        self.overrideredirect(True); self.attributes('-topmost', True)
        self.text_label = tk.Label(self, text=self.listening_indicator, wraplength=self.winfo_screenwidth() - 100); self.text_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.apply_settings(); self.text_label.bind("<ButtonPress-1>", self.start_move); self.text_label.bind("<ButtonRelease-1>", self.stop_move); self.text_label.bind("<B1-Motion>", self.do_move); self.check_text_queue()
    def apply_settings(self):
        settings = self.config_manager.get("overlay_settings", {}); pos_x = settings.get("position_x", 100); pos_y = settings.get("position_y", 800)
        self.geometry(f"+{pos_x}+{pos_y}"); self.attributes("-alpha", settings.get("alpha", 0.75)); bg_color = settings.get("bg_color", "black"); font_color = settings.get("font_color", "white"); font_size = settings.get("font_size", 24); font_family = settings.get("font_family", "Arial")
        min_display = int(settings.get("min_display_ms", 4000)); self.min_display_ms = max(min_display, 200)
        merge_time = int(settings.get("merge_consecutive_ms", 1500)); self.merge_consecutive_ms = max(merge_time, 100)
        self.configure(bg=bg_color); self.text_label.configure(bg=bg_color, fg=font_color, font=(font_family, font_size))
    def clear_text(self): self.text_label.config(text=self.listening_indicator); self.clear_timer_id = None
    def _update_display_text(self, text_to_add):
        now = time.time()
        if self.clear_timer_id: self.after_cancel(self.clear_timer_id)
        time_since_last_update = (now - self.last_text_update_time) * 1000; current_text = self.text_label.cget("text")
        if time_since_last_update < self.merge_consecutive_ms and current_text != self.listening_indicator: new_text = f"{current_text} {text_to_add}"
        else: new_text = text_to_add
        self.text_label.config(text=new_text); self.last_text_update_time = now; self.clear_timer_id = self.after(self.min_display_ms, self.clear_text)
    def check_text_queue(self):
        try:
            while True: text = self.text_queue.get_nowait(); self._update_display_text(text)
        except queue.Empty: pass
        finally: self.after(100, self.check_text_queue)
    def start_move(self, event): self.x, self.y = event.x, event.y
    def stop_move(self, event): self.x, self.y = None, None
    def do_move(self, event): self.geometry(f"+{self.winfo_x() + event.x - self.x}+{self.winfo_y() + event.y - self.y}")

class SettingsWindow(tk.Toplevel):
    """Janela de Configurações, agora sem a lógica do logo."""
    def __init__(self, master, config_manager, restart_callback):
        super().__init__(master)
        self.title("Configurações do Transcritor")
        self.config_manager = config_manager
        self.restart_callback = restart_callback
        self.config_vars = {}
        self.device_map = {}
        self.create_widgets()
        self.load_settings()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def create_widgets(self):
        frame = ttk.Frame(self, padding="10")
        frame.pack(expand=True, fill="both")
        notebook = ttk.Notebook(frame)
        notebook.pack(expand=True, fill="both", pady=5)
        ws_tab = ttk.Frame(notebook, padding="10"); notebook.add(ws_tab, text="Web Service"); self.create_ws_widgets(ws_tab)
        translation_tab = ttk.Frame(notebook, padding="10"); notebook.add(translation_tab, text="Tradução (Ollama)"); self.create_translation_widgets(translation_tab)
        audio_tab = ttk.Frame(notebook, padding="10"); notebook.add(audio_tab, text="Áudio e VAD"); self.create_audio_widgets(audio_tab)
        proc_tab = ttk.Frame(notebook, padding="10"); notebook.add(proc_tab, text="Processamento de Áudio"); self.create_proc_widgets(proc_tab)
        overlay_tab = ttk.Frame(notebook, padding="10"); notebook.add(overlay_tab, text="Overlay"); self.create_overlay_widgets(overlay_tab)
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill="x", pady=(10,0))
        ttk.Button(button_frame, text="Salvar e Reiniciar", command=self.save_and_restart).pack(side="right", padx=5)
        ttk.Button(button_frame, text="Cancelar", command=self.destroy).pack(side="right")

    def create_ws_widgets(self, parent):
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="URL do Serviço:").grid(row=0, column=0, sticky="w", pady=2, padx=5); self.config_vars["ws_url"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["ws_url"]).grid(row=0, column=1, sticky="ew")
        ttk.Label(parent, text="Nome do campo (arquivo):").grid(row=1, column=0, sticky="w", pady=2, padx=5); self.config_vars["ws_form_name"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["ws_form_name"]).grid(row=1, column=1, sticky="ew")
        ttk.Label(parent, text="Tipo de Resposta (leitura):").grid(row=2, column=0, sticky="w", pady=2, padx=5); self.config_vars["ws_resp_type"] = tk.StringVar(); ttk.Combobox(parent, textvariable=self.config_vars["ws_resp_type"], values=["json", "text"], state="readonly").grid(row=2, column=1, sticky="ew")
        ttk.Label(parent, text="Chave do JSON (ex: text):").grid(row=3, column=0, sticky="w", pady=2, padx=5); self.config_vars["ws_json_key"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["ws_json_key"]).grid(row=3, column=1, sticky="ew")
        ttk.Label(parent, text="Temperature:").grid(row=4, column=0, sticky="w", pady=2, padx=5); self.config_vars["ws_temperature"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["ws_temperature"]).grid(row=4, column=1, sticky="ew")
        ttk.Label(parent, text="Temperature Inc:").grid(row=5, column=0, sticky="w", pady=2, padx=5); self.config_vars["ws_temperature_inc"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["ws_temperature_inc"]).grid(row=5, column=1, sticky="ew")
        ttk.Label(parent, text="Response Format (parâmetro):").grid(row=6, column=0, sticky="w", pady=2, padx=5); self.config_vars["ws_response_format_param"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["ws_response_format_param"]).grid(row=6, column=1, sticky="ew")

    def create_translation_widgets(self, parent):
        parent.columnconfigure(1, weight=1)
        self.config_vars["trans_enabled"] = tk.BooleanVar(); ttk.Checkbutton(parent, text="Habilitar Tradução via Ollama", variable=self.config_vars["trans_enabled"]).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        ttk.Label(parent, text="Host do Ollama:").grid(row=1, column=0, sticky="w", padx=5, pady=2); self.config_vars["trans_host"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["trans_host"]).grid(row=1, column=1, sticky="ew")
        ttk.Label(parent, text="Porta do Ollama:").grid(row=2, column=0, sticky="w", padx=5, pady=2); self.config_vars["trans_port"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["trans_port"]).grid(row=2, column=1, sticky="ew")
        ttk.Label(parent, text="Modelo (Ollama):").grid(row=3, column=0, sticky="w", padx=5, pady=2); self.config_vars["trans_model"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["trans_model"]).grid(row=3, column=1, sticky="ew")
        ttk.Label(parent, text="Prompt do Sistema (Instrução):").grid(row=4, column=0, sticky="nw", padx=5, pady=5)
        self.config_vars["trans_system_prompt"] = tk.Text(parent, height=6, wrap="word"); self.config_vars["trans_system_prompt"].grid(row=4, column=1, sticky="ew", pady=5); parent.rowconfigure(4, weight=1)

    def create_audio_widgets(self, parent):
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Dispositivo de Áudio:").grid(row=0, column=0, sticky="w", padx=5, pady=5); self.config_vars["device_index"] = tk.StringVar()
        devices = self.get_audio_devices(); device_menu = ttk.Combobox(parent, textvariable=self.config_vars["device_index"], values=list(devices.keys()), state="readonly"); device_menu.grid(row=0, column=1, sticky="ew"); self.device_map = devices
        ttk.Label(parent, text="Agressividade VAD (0-3):").grid(row=1, column=0, sticky="w", padx=5, pady=5); self.config_vars["vad_aggressiveness"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["vad_aggressiveness"]).grid(row=1, column=1, sticky="ew")
        ttk.Label(parent, text="Gatilho de Voz (ex: 0.75):").grid(row=2, column=0, sticky="w", padx=5, pady=5); self.config_vars["vad_trigger_ratio"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["vad_trigger_ratio"]).grid(row=2, column=1, sticky="ew")
        ttk.Label(parent, text="Pausa de Silêncio (ms):").grid(row=3, column=0, sticky="w", padx=5, pady=5); self.config_vars["vad_silence_ms"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["vad_silence_ms"]).grid(row=3, column=1, sticky="ew")

    def create_proc_widgets(self, parent):
        parent.columnconfigure(1, weight=1)
        self.config_vars["proc_enable_filter"] = tk.BooleanVar(); ttk.Checkbutton(parent, text="Habilitar Filtro Passa-Alta", variable=self.config_vars["proc_enable_filter"]).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        ttk.Label(parent, text="Frequência de Corte do Filtro (Hz):").grid(row=1, column=0, sticky="w", padx=5, pady=2); self.config_vars["proc_filter_cutoff"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["proc_filter_cutoff"]).grid(row=1, column=1, sticky="ew")
        self.config_vars["proc_enable_normalize"] = tk.BooleanVar(); ttk.Checkbutton(parent, text="Habilitar Normalização de Volume", variable=self.config_vars["proc_enable_normalize"]).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        ttk.Label(parent, text="Duração Mínima do Segmento (ms):").grid(row=3, column=0, sticky="w", padx=5, pady=2); self.config_vars["proc_min_duration"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["proc_min_duration"]).grid(row=3, column=1, sticky="ew")

    def create_overlay_widgets(self, parent):
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Fonte:").grid(row=0, column=0, sticky="w", padx=5, pady=5); self.config_vars["font_family"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["font_family"]).grid(row=0, column=1, sticky="ew")
        ttk.Label(parent, text="Tamanho da Fonte:").grid(row=1, column=0, sticky="w", padx=5, pady=5); self.config_vars["font_size"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["font_size"]).grid(row=1, column=1, sticky="ew")
        ttk.Label(parent, text="Cor da Fonte:").grid(row=2, column=0, sticky="w", padx=5, pady=5); self.config_vars["font_color"] = tk.StringVar(); font_color_entry = ttk.Entry(parent, textvariable=self.config_vars["font_color"]); font_color_entry.grid(row=2, column=1, sticky="ew"); ttk.Button(parent, text="Escolher...", command=lambda: self.choose_color("font_color")).grid(row=2, column=2, padx=5)
        ttk.Label(parent, text="Cor de Fundo:").grid(row=3, column=0, sticky="w", padx=5, pady=5); self.config_vars["bg_color"] = tk.StringVar(); bg_color_entry = ttk.Entry(parent, textvariable=self.config_vars["bg_color"]); bg_color_entry.grid(row=3, column=1, sticky="ew"); ttk.Button(parent, text="Escolher...", command=lambda: self.choose_color("bg_color")).grid(row=3, column=2, padx=5)
        ttk.Label(parent, text="Transparência (0.1-1.0):").grid(row=4, column=0, sticky="w", padx=5, pady=5); self.config_vars["alpha"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["alpha"]).grid(row=4, column=1, sticky="ew")
        ttk.Label(parent, text="Tempo Mín. de Exibição (ms):").grid(row=5, column=0, sticky="w", padx=5, pady=5); self.config_vars["overlay_min_display"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["overlay_min_display"]).grid(row=5, column=1, sticky="ew")
        ttk.Label(parent, text="Agrupar Legendas (ms):").grid(row=6, column=0, sticky="w", padx=5, pady=5); self.config_vars["overlay_merge_time"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["overlay_merge_time"]).grid(row=6, column=1, sticky="ew")

    def get_audio_devices(self):
        devices = {}
        try:
            default_mic, default_loopback = None, None; default_mic_name, default_loopback_name = "", ""
            try:
                default_mic = sc.default_microphone()
                if default_mic: devices["Padrão: Microfone (Entrada)"] = "default_mic"; default_mic_name = default_mic.name
            except Exception as e: print(f"Aviso: não foi possível encontrar o microfone padrão. {e}")
            try:
                default_loopback = find_default_loopback_device()
                if default_loopback: devices["Padrão: Saída de Som (Loopback)"] = "default_loopback"; default_loopback_name = default_loopback.name
            except Exception as e: print(f"Aviso: não foi possível encontrar o loopback padrão. {e}")
            all_recordable_devices = sc.all_microphones(include_loopback=True)
            for device in all_recordable_devices:
                if device.name == default_mic_name or device.name == default_loopback_name: continue
                is_loopback = hasattr(device, 'is_loopback') and device.is_loopback
                label = f"[SAÍDA] {device.name}" if is_loopback else f"[ENTRADA] {device.name}"
                devices[label] = device.id
        except Exception as e:
            print(f"ERRO CRÍTICO ao listar dispositivos de áudio: {e}")
            messagebox.showerror("Erro de Áudio", f"Não foi possível listar os dispositivos.\nDetalhes: {e}")
        if not devices: devices["Nenhum dispositivo encontrado"] = ""
        return devices

    def choose_color(self, var_name):
        color_code = colorchooser.askcolor(title="Escolha uma cor");
        if color_code and color_code[1]: self.config_vars[var_name].set(color_code[1])

    def load_settings(self):
        config = self.config_manager.config; ws = config.get("webservice_settings", {}); trans = config.get("translation_settings", {}); audio = config.get("audio_settings", {}); proc = config.get("audio_processing_settings", {}); overlay = config.get("overlay_settings", {})
        self.config_vars["ws_url"].set(ws.get("url", "")); self.config_vars["ws_form_name"].set(ws.get("file_form_name", "file")); self.config_vars["ws_resp_type"].set(ws.get("response_type", "json")); self.config_vars["ws_json_key"].set(ws.get("json_response_key", "text")); self.config_vars["ws_temperature"].set(ws.get("temperature", "0.0")); self.config_vars["ws_temperature_inc"].set(ws.get("temperature_inc", "0.2")); self.config_vars["ws_response_format_param"].set(ws.get("response_format_param", "json"))
        self.config_vars["trans_enabled"].set(trans.get("enabled", False)); self.config_vars["trans_host"].set(trans.get("ollama_host", "localhost")); self.config_vars["trans_port"].set(trans.get("ollama_port", "11434")); self.config_vars["trans_model"].set(trans.get("ollama_model", "llama3")); self.config_vars["trans_system_prompt"].delete("1.0", tk.END); self.config_vars["trans_system_prompt"].insert("1.0", trans.get("system_prompt", ""))
        device_id = audio.get("device_index", "default_loopback"); reversed_device_map = {v: k for k, v in self.device_map.items()}; device_name = reversed_device_map.get(device_id, "Padrão: Saída de Som (Loopback)"); self.config_vars["device_index"].set(device_name); self.config_vars["vad_aggressiveness"].set(audio.get("vad_aggressiveness", 3)); self.config_vars["vad_trigger_ratio"].set(audio.get("vad_trigger_ratio", 0.75)); self.config_vars["vad_silence_ms"].set(audio.get("vad_silence_ms", 700))
        self.config_vars["proc_enable_filter"].set(proc.get("enable_high_pass", True)); self.config_vars["proc_filter_cutoff"].set(proc.get("high_pass_cutoff_hz", 100.0)); self.config_vars["proc_enable_normalize"].set(proc.get("enable_normalization", True)); self.config_vars["proc_min_duration"].set(proc.get("min_segment_duration_ms", 250))
        self.config_vars["font_family"].set(overlay.get("font_family", "Arial")); self.config_vars["font_size"].set(overlay.get("font_size", 24)); self.config_vars["font_color"].set(overlay.get("font_color", "white")); self.config_vars["bg_color"].set(overlay.get("bg_color", "black")); self.config_vars["alpha"].set(overlay.get("alpha", 0.75))
        self.config_vars["overlay_min_display"].set(overlay.get("min_display_ms", 4000)); self.config_vars["overlay_merge_time"].set(overlay.get("merge_consecutive_ms", 1500))

    def save_and_restart(self):
        try:
            selected_device_name = self.config_vars["device_index"].get(); device_id_to_save = self.device_map.get(selected_device_name)
            overlay_x, overlay_y = 0, 0
            if hasattr(self.master, 'overlay') and self.master.overlay.winfo_exists():
                overlay_x = self.master.overlay.winfo_x()
                overlay_y = self.master.overlay.winfo_y()

            new_config = {
                "webservice_settings": {"url": self.config_vars["ws_url"].get(), "file_form_name": self.config_vars["ws_form_name"].get(), "response_type": self.config_vars["ws_resp_type"].get(), "json_response_key": self.config_vars["ws_json_key"].get(), "temperature": self.config_vars["ws_temperature"].get(), "temperature_inc": self.config_vars["ws_temperature_inc"].get(), "response_format_param": self.config_vars["ws_response_format_param"].get()},
                "translation_settings": {"enabled": self.config_vars["trans_enabled"].get(), "ollama_host": self.config_vars["trans_host"].get(), "ollama_port": self.config_vars["trans_port"].get(), "ollama_model": self.config_vars["trans_model"].get(), "system_prompt": self.config_vars["trans_system_prompt"].get("1.0", tk.END).strip()},
                "audio_settings": {"device_index": device_id_to_save, "vad_aggressiveness": int(self.config_vars["vad_aggressiveness"].get()), "vad_trigger_ratio": float(self.config_vars["vad_trigger_ratio"].get()), "vad_silence_ms": int(self.config_vars["vad_silence_ms"].get())},
                "audio_processing_settings": {"enable_high_pass": self.config_vars["proc_enable_filter"].get(), "high_pass_cutoff_hz": float(self.config_vars["proc_filter_cutoff"].get()), "enable_normalization": self.config_vars["proc_enable_normalize"].get(), "min_segment_duration_ms": int(self.config_vars["proc_min_duration"].get())},
                "overlay_settings": {"font_family": self.config_vars["font_family"].get(), "font_size": int(self.config_vars["font_size"].get()), "font_color": self.config_vars["font_color"].get(), "bg_color": self.config_vars["bg_color"].get(), "alpha": float(self.config_vars["alpha"].get()), "position_x": overlay_x, "position_y": overlay_y, "min_display_ms": int(self.config_vars["overlay_min_display"].get()), "merge_consecutive_ms": int(self.config_vars["overlay_merge_time"].get())}
            }
            self.config_manager.save_config(new_config)
            self.destroy()
            self.restart_callback()
        except Exception as e:
            messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar as configurações. Verifique se os valores estão corretos.\n\nDetalhe: {e}")

class MainApplication(tk.Tk):
    """Classe principal da aplicação, agora responsável por exibir o logo."""
    def __init__(self):
        super().__init__()
        self.withdraw()
        self.title("Controlador do Transcritor")
        self.logo_image = None # Atributo para manter a referência da imagem
        
        try: self.config_manager = ConfigManager()
        except Exception as e: root = tk.Tk(); root.withdraw(); messagebox.showerror("Erro Crítico", f"Não foi possível carregar a configuração: {e}\nO programa será fechado."); return
        
        self.audio_queue = queue.Queue(); self.transcribed_text_queue = queue.Queue(); self.final_text_queue = queue.Queue()
        self.audio_thread = None; self.transcription_thread = None; self.translation_thread = None
        self.overlay = None; self.settings_window = None
        self.create_control_window(); self.start_services()

    def create_logo_widget(self, parent_frame):
        """Carrega, redimensiona e exibe o logo na janela de controle."""
        if not SVG_SUPPORT:
            return

        logo_path = "rt-subs.svg"
        if not os.path.exists(logo_path):
            print(f"Aviso: Arquivo do logo '{logo_path}' não encontrado.")
            return

        try:
            max_dimension = 128
            png_data = cairosvg.svg2png(url=logo_path, output_width=max_dimension, output_height=max_dimension)
            pil_image = Image.open(io.BytesIO(png_data))
            self.logo_image = ImageTk.PhotoImage(pil_image)
            
            logo_label = tk.Label(parent_frame, image=self.logo_image, borderwidth=0)
            logo_label.pack(pady=(10, 5))
        except Exception as e:
            print(f"ERRO ao carregar o logo SVG: {e}")

    def create_control_window(self):
        self.control_window = tk.Toplevel(self)
        self.control_window.title("Controle")
        self.control_window.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.control_window.geometry("+100+100")

        # Chama a função para criar o logo antes dos outros widgets
        self.create_logo_widget(self.control_window)
        
        ttk.Button(self.control_window, text="Abrir Configurações", command=self.open_settings).pack(pady=10, padx=10, fill='x')
        self.status_label = ttk.Label(self.control_window, text="Inicializando...", foreground="blue", wraplength=280)
        self.status_label.pack(pady=5, padx=10)
        ttk.Button(self.control_window, text="Sair", command=self.on_exit).pack(pady=10, padx=10)

    def update_status(self, message, color="black"):
        if hasattr(self, 'status_label') and self.status_label.winfo_exists():
            if color == "purple": color = "#8A2BE2"
            self.status_label.config(text=message, foreground=color)

    def start_services(self):
        self.stop_services()
        if self.overlay: self.overlay.destroy()
        for q in [self.audio_queue, self.transcribed_text_queue, self.final_text_queue]:
            while not q.empty(): q.get()
        self.overlay = OverlayWindow(self, self.final_text_queue, self.config_manager)
        self.audio_thread = AudioRecorder(self.audio_queue, self.config_manager, self.update_status)
        self.transcription_thread = TranscriptionProcessor(self.audio_queue, self.transcribed_text_queue, self.config_manager, self.update_status)
        self.translation_thread = TranslationProcessor(self.transcribed_text_queue, self.final_text_queue, self.config_manager, self.update_status)
        self.audio_thread.start(); self.transcription_thread.start(); self.translation_thread.start()

    def stop_services(self):
        if self.audio_thread and self.audio_thread.is_alive(): self.audio_thread.stop(); self.audio_thread.join(timeout=1)
        if self.transcription_thread and self.transcription_thread.is_alive(): self.transcription_thread.stop(); self.transcription_thread.join(timeout=1)
        if self.translation_thread and self.translation_thread.is_alive(): self.translation_thread.stop(); self.translation_thread.join(timeout=1)

    def restart_services(self):
        self.update_status("Reiniciando...", "orange")
        self.config_manager.load_config()
        self.start_services()

    def open_settings(self):
        if self.settings_window is None or not self.settings_window.winfo_exists():
            self.settings_window = SettingsWindow(self, self.config_manager, self.restart_services)
            self.settings_window.transient(self.control_window)
        else:
            self.settings_window.lift()

    def on_exit(self):
        self.update_status("Fechando...", "gray")
        if self.overlay and self.overlay.winfo_exists():
            config = self.config_manager.config
            config["overlay_settings"]["position_x"] = self.overlay.winfo_x()
            config["overlay_settings"]["position_y"] = self.overlay.winfo_y()
            self.config_manager.save_config(config)
        self.stop_services()
        self.destroy()

# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    # Verificações de dependências críticas antes de iniciar o app
    try: import webrtcvad
    except ImportError: root = tk.Tk(); root.withdraw(); messagebox.showerror("Dependência Faltando", "A biblioteca 'webrtcvad' não foi encontrada.\n\nPor favor, instale-a com o comando:\npip install webrtcvad")
    else:
        try: import scipy
        except ImportError: root = tk.Tk(); root.withdraw(); messagebox.showerror("Dependência Faltando", "A biblioteca 'scipy' não foi encontrada.\n\nPor favor, instale-a com o comando:\npip install scipy")
        else:
            app = MainApplication()
            # Garante que o loop principal só inicie se a janela de controle foi criada com sucesso
            if hasattr(app, 'control_window') and app.control_window.winfo_exists():
                app.mainloop()