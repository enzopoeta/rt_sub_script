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
import numpy as np

# --- Dependências de Áudio e VAD ---
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
        for mic in all_mics:
            if mic.is_loopback and default_speaker.name in mic.name:
                return mic
        for mic in all_mics:
            if mic.is_loopback:
                return mic
    except Exception as e:
        print(f"Não foi possível encontrar o dispositivo de loopback padrão: {e}")
    return None

# --- Classes de Lógica de Negócio ---

class VADAutoTuner(threading.Thread):
    """Gerencia o ajuste automático dos parâmetros do VAD."""
    def __init__(self, config_manager, status_callback):
        super().__init__(daemon=True)
        self.config_manager = config_manager
        self.status_callback = status_callback
        self._stop_event = threading.Event()
        self.lock = threading.Lock()
        self.false_positives = deque(maxlen=20)
        self.ambient_noise_levels = deque(maxlen=100)
        self.load_settings()

    def load_settings(self):
        with self.lock:
            settings = self.config_manager.get("autotune_settings", {})
            self.is_enabled = settings.get("enabled", False)
            self.tune_interval_sec = settings.get("interval_sec", 30)
            self.fp_threshold_ratio = settings.get("fp_threshold_ratio", 0.3)
            self.fp_min_char_count = settings.get("fp_min_char_count", 5)
            self.noise_threshold_db = settings.get("noise_threshold_db", -40.0)

    def report_transcription_result(self, text, audio_duration_s):
        if not self.is_enabled: return
        is_fp = (text is None) or (len(text.strip()) < self.fp_min_char_count)
        with self.lock:
            self.false_positives.append(1 if is_fp else 0)

    def report_silent_frame(self, frame_data):
        if not self.is_enabled: return
        rms = np.sqrt(np.mean(frame_data**2))
        if rms > 1e-9:
            dbfs = 20 * np.log10(rms)
            with self.lock:
                self.ambient_noise_levels.append(dbfs)

    def run(self):
        while not self._stop_event.is_set():
            time.sleep(self.tune_interval_sec)
            if not self.is_enabled:
                continue

            with self.lock:
                if not len(self.false_positives): continue
                
                fp_rate = sum(self.false_positives) / len(self.false_positives)
                avg_noise = np.mean(self.ambient_noise_levels) if len(self.ambient_noise_levels) > 0 else -90.0
                
                audio_cfg = self.config_manager.config["audio_settings"]
                current_aggressiveness = audio_cfg["vad_aggressiveness"]
                current_trigger_ratio = audio_cfg["vad_trigger_ratio"]

                adjusted = False
                if fp_rate > self.fp_threshold_ratio or avg_noise > self.noise_threshold_db:
                    if current_aggressiveness < 3:
                        audio_cfg["vad_aggressiveness"] += 1
                        adjusted = True
                    if current_trigger_ratio < 0.95:
                         audio_cfg["vad_trigger_ratio"] = min(0.95, current_trigger_ratio + 0.05)
                         adjusted = True

                if adjusted:
                    self.status_callback(f"Auto-Tune: Agressividade->{audio_cfg['vad_aggressiveness']}", "purple")
                    self.config_manager.save_config(self.config_manager.config)
                    self.false_positives.clear()
                    self.ambient_noise_levels.clear()

    def stop(self):
        self._stop_event.set()


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
                "overlay_settings": {"mode": "dinamico", "static_width": 400, "static_height": 150, "font_family": "Arial", "font_size": 24, "font_color": "white", "bg_color": "black", "alpha": 0.75, "position_x": 100, "position_y": 800, "min_display_ms": 4000, "merge_consecutive_ms": 1500},
                "autotune_settings": {"enabled": False, "interval_sec": 30, "fp_threshold_ratio": 0.3, "fp_min_char_count": 5, "noise_threshold_db": -40.0}
            }
            self.save_config(default_config)
            return default_config
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
            config.setdefault("audio_processing_settings", { "enable_high_pass": True, "high_pass_cutoff_hz": 100.0, "enable_normalization": True, "min_segment_duration_ms": 250 })
            overlay_defaults = config.setdefault("overlay_settings", {})
            overlay_defaults.setdefault("mode", "dinamico"); overlay_defaults.setdefault("static_width", 400); overlay_defaults.setdefault("static_height", 150)
            config.setdefault("audio_settings", {}).setdefault("device_index", "default_loopback")
            config.setdefault("translation_settings", {"enabled": False, "ollama_host": "localhost", "ollama_port": "11434", "ollama_model": "llama3", "system_prompt": "You are a helpful and expert translator."})
            config.setdefault("autotune_settings", {"enabled": False, "interval_sec": 30, "fp_threshold_ratio": 0.3, "fp_min_char_count": 5, "noise_threshold_db": -40.0})
            return config

    def save_config(self, new_config):
        self.config = new_config
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def get(self, key, default=None):
        return self.config.get(key, default)

class AudioRecorder(threading.Thread):
    def __init__(self, audio_queue, config_manager, status_callback, vad_tuner):
        super().__init__(daemon=True)
        self.audio_queue = audio_queue; self.config_manager = config_manager; self.status_callback = status_callback; self.vad_tuner = vad_tuner
        self._stop_event = threading.Event(); self.device = None; self.lock = threading.Lock()
        self.load_proc_settings()
    
    def load_proc_settings(self):
        with self.lock:
            proc_settings = self.config_manager.get("audio_processing_settings", {})
            self.enable_filter = proc_settings.get("enable_high_pass", True); self.filter_cutoff = proc_settings.get("high_pass_cutoff_hz", 100.0)
            self.enable_normalize = proc_settings.get("enable_normalization", True); self.min_duration_ms = proc_settings.get("min_segment_duration_ms", 250)
    
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
            except Exception as e: self.status_callback(f"Erro no áudio: {e}", "red"); time.sleep(5)

    def start_recording(self):
        with self.lock:
            settings = self.config_manager.get("audio_settings", {}); device_index = settings.get("device_index", "default_loopback"); samplerate = 16000; frame_duration_ms = 10; frame_size = int(samplerate * frame_duration_ms / 1000)
            aggressiveness = int(settings.get("vad_aggressiveness", 3)); silence_duration_ms = int(settings.get("vad_silence_ms", 700)); trigger_ratio = float(settings.get("vad_trigger_ratio", 0.75))
        
        if not webrtcvad.valid_rate_and_frame_length(samplerate, frame_size): raise RuntimeError(f"Taxa de amostragem/frame inválida: {samplerate}/{frame_size}")
        
        try:
            if device_index == "default_mic": self.device = sc.default_microphone()
            elif device_index == "default_loopback":
                self.device = find_default_loopback_device()
                if not self.device: raise RuntimeError("Dispositivo de loopback padrão não encontrado.")
            else: self.device = sc.get_microphone(id=device_index, include_loopback=True)
            
            self.status_callback(f"Gravando de: {self.device.name}", "blue")
            vad = webrtcvad.Vad(aggressiveness)
            
            with self.device.recorder(samplerate=samplerate, blocksize=frame_size) as mic:
                self.status_callback("Ouvindo com VAD...", "green")
                self.record_with_vad(mic, vad, samplerate, frame_size, silence_duration_ms, trigger_ratio)

        except Exception as e: raise RuntimeError(f"Não foi possível iniciar a gravação VAD. Verifique o dispositivo. {e}")

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
            
            if not is_speech: self.vad_tuner.report_silent_frame(frame_data_float)

            trigger_buffer.append(is_speech)
            if not is_speaking:
                ring_buffer.append(frame_data_float)
                if sum(trigger_buffer) > trigger_ratio * trigger_buffer.maxlen:
                    self.status_callback("Detectando fala...", "orange"); is_speaking = True; voiced_frames.extend(ring_buffer); ring_buffer.clear(); silent_frames_count = 0
            else:
                voiced_frames.append(frame_data_float)
                if not is_speech: silent_frames_count += 1
                else: silent_frames_count = 0
                if silent_frames_count >= silence_frames_needed:
                    audio_segment = np.concatenate(voiced_frames)
                    processed_segment = self.preprocess_audio(audio_segment, samplerate)
                    if processed_segment is not None:
                        self.audio_queue.put((processed_segment, samplerate))
                    is_speaking = False; voiced_frames.clear(); ring_buffer.clear(); trigger_buffer.clear()
                    self.status_callback("Ouvindo com VAD...", "green")

    def stop(self): self._stop_event.set()


class TranscriptionProcessor(threading.Thread):
    def __init__(self, audio_queue, transcribed_text_queue, config_manager, status_callback, vad_tuner):
        super().__init__(daemon=True); self.audio_queue = audio_queue; self.transcribed_text_queue = transcribed_text_queue; self.config_manager = config_manager; self.status_callback = status_callback; self.vad_tuner = vad_tuner; self._stop_event = threading.Event()
    def run(self):
        while not self._stop_event.is_set():
            try:
                settings = self.config_manager.get("webservice_settings", {}); url = settings.get("url")
                if not url: self.status_callback("URL do Web Service não configurada!", "red"); time.sleep(5); continue
                
                audio_data, samplerate = self.audio_queue.get(timeout=1)
                self.status_callback("Enviando áudio para transcrição...", "orange")
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    write_wav(tmpfile.name, samplerate, audio_data); tmpfile_path = tmpfile.name
                
                try:
                    payload = {"temperature": settings.get("temperature", "0.0"), "temperature_inc": settings.get("temperature_inc", "0.2"), "response_format": settings.get("response_format_param", "json")}
                    with open(tmpfile_path, "rb") as audio_file:
                        files = {settings.get("file_form_name", "file"): audio_file}
                        response = requests.post(url, files=files, data=payload, timeout=30); response.raise_for_status()
                    
                    transcribed_text = None
                    if settings.get("response_type") == "json":
                        json_key = settings.get("json_response_key", "text"); json_response = response.json(); keys = json_key.split('.'); value = json_response
                        for key in keys: value = value.get(key) if isinstance(value, dict) else None
                        transcribed_text = value
                    else: transcribed_text = response.text

                    self.vad_tuner.report_transcription_result(transcribed_text, len(audio_data) / samplerate)

                    if transcribed_text and transcribed_text.strip():
                        self.transcribed_text_queue.put(transcribed_text.strip())
                    else:
                        self.status_callback("Ouvindo com VAD...", "green")

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
                    else: self.output_queue.put(f"[Trad. vazia] {text_to_process}")
                except requests.exceptions.RequestException as e: self.status_callback(f"Erro de rede (Ollama): {e}", "red"); self.output_queue.put(f"[ERRO: Ollama] {text_to_process}"); time.sleep(5)
            except queue.Empty: continue
            except Exception as e: self.status_callback(f"Erro na tradução: {e}", "red"); time.sleep(5)
    def stop(self): self._stop_event.set()

class OverlayWindow(tk.Toplevel):
    def __init__(self, parent, final_text_queue, config_manager):
        super().__init__(parent); self.text_queue = final_text_queue; self.config_manager = config_manager
        self.min_display_ms = 4000; self.merge_consecutive_ms = 1500; self.last_text_update_time = 0; self.clear_timer_id = None; self.listening_indicator = "..."
        self.font_family = "Arial"; self.original_font_size = 24; self.mode = "dinamico"
        self.overrideredirect(True); self.attributes('-topmost', True)
        self.text_label = tk.Label(self, text="", justify="center", anchor="center"); self.text_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.apply_settings(); self.text_label.bind("<ButtonPress-1>", self.start_move); self.text_label.bind("<ButtonRelease-1>", self.stop_move); self.text_label.bind("<B1-Motion>", self.do_move); self.check_text_queue()
    def apply_settings(self):
        settings = self.config_manager.get("overlay_settings", {}); pos_x = settings.get("position_x", 100); pos_y = settings.get("position_y", 800)
        self.attributes("-alpha", settings.get("alpha", 0.75)); bg_color = settings.get("bg_color", "black"); font_color = settings.get("font_color", "white")
        self.font_family = settings.get("font_family", "Arial"); self.original_font_size = int(settings.get("font_size", 24)); self.mode = settings.get("mode", "dinamico")
        min_display = int(settings.get("min_display_ms", 4000)); self.min_display_ms = max(min_display, 200)
        merge_time = int(settings.get("merge_consecutive_ms", 1500)); self.merge_consecutive_ms = max(merge_time, 100)
        self.configure(bg=bg_color); self.text_label.configure(bg=bg_color, fg=font_color, font=(self.font_family, self.original_font_size))
        self.clear_text(cancel_timer=False)
        if self.mode == "estatico":
            width = int(settings.get("static_width", 400)); height = int(settings.get("static_height", 150))
            self.geometry(f"{width}x{height}+{pos_x}+{pos_y}"); self.text_label.config(wraplength=width - 20)
        else:
            self.geometry(f"+{pos_x}+{pos_y}"); self.text_label.config(wraplength=self.winfo_screenwidth() - 100)
    def _adjust_font_to_fit(self):
        self.update_idletasks()
        container_width = self.winfo_width() - 20; container_height = self.winfo_height() - 10
        text_width = self.text_label.winfo_reqwidth(); text_height = self.text_label.winfo_reqheight()
        current_size = self.original_font_size
        while (text_height > container_height or text_width > container_width) and current_size > 4:
            current_size -= 1; new_font = font.Font(family=self.font_family, size=current_size)
            self.text_label.config(font=new_font); self.update_idletasks()
            text_height = self.text_label.winfo_reqheight(); text_width = self.text_label.winfo_reqwidth()
    def is_cleared(self):
        current_text = self.text_label.cget("text")
        return current_text == "" if self.mode == 'estatico' else current_text == self.listening_indicator
    def clear_text(self, cancel_timer=True):
        if cancel_timer and self.clear_timer_id: self.after_cancel(self.clear_timer_id)
        text_to_show = "" if self.mode == "estatico" else self.listening_indicator
        self.text_label.config(text=text_to_show, font=(self.font_family, self.original_font_size))
        if cancel_timer: self.clear_timer_id = None
    def _update_display_text(self, text_to_add):
        now = time.time()
        if self.clear_timer_id: self.after_cancel(self.clear_timer_id)
        new_text = f"{self.text_label.cget('text')} {text_to_add}" if (now - self.last_text_update_time) * 1000 < self.merge_consecutive_ms and not self.is_cleared() else text_to_add
        self.text_label.config(font=(self.font_family, self.original_font_size), text=new_text)
        if self.mode == "estatico": self._adjust_font_to_fit()
        self.last_text_update_time = now; self.clear_timer_id = self.after(self.min_display_ms, self.clear_text)
    def check_text_queue(self):
        try:
            while True: text = self.text_queue.get_nowait(); self._update_display_text(text)
        except queue.Empty: pass
        finally: self.after(100, self.check_text_queue)
    def start_move(self, event): self.x, self.y = event.x, event.y
    def stop_move(self, event): self.x, self.y = None, None
    def do_move(self, event): self.geometry(f"+{self.winfo_x() + event.x - self.x}+{self.winfo_y() + event.y - self.y}")

class SettingsWindow(tk.Toplevel):
    def __init__(self, master, config_manager, restart_callback):
        super().__init__(master)
        self.title("Configurações"); self.config_manager = config_manager; self.restart_callback = restart_callback; self.config_vars = {}; self.device_map = {}
        self.create_widgets(); self.load_settings(); self.protocol("WM_DELETE_WINDOW", self.destroy)
    def create_widgets(self):
        frame = ttk.Frame(self, padding="10"); frame.pack(expand=True, fill="both")
        notebook = ttk.Notebook(frame); notebook.pack(expand=True, fill="both", pady=5)
        ws_tab = ttk.Frame(notebook, padding="10"); notebook.add(ws_tab, text="Web Service"); self.create_ws_widgets(ws_tab)
        translation_tab = ttk.Frame(notebook, padding="10"); notebook.add(translation_tab, text="Tradução"); self.create_translation_widgets(translation_tab)
        audio_tab = ttk.Frame(notebook, padding="10"); notebook.add(audio_tab, text="Áudio e VAD"); self.create_audio_widgets(audio_tab)
        autotune_tab = ttk.Frame(notebook, padding="10"); notebook.add(autotune_tab, text="VAD Auto-Tune"); self.create_autotune_widgets(autotune_tab)
        proc_tab = ttk.Frame(notebook, padding="10"); notebook.add(proc_tab, text="Processamento"); self.create_proc_widgets(proc_tab)
        overlay_tab = ttk.Frame(notebook, padding="10"); notebook.add(overlay_tab, text="Overlay"); self.create_overlay_widgets(overlay_tab)
        button_frame = ttk.Frame(frame); button_frame.pack(fill="x", pady=(10,0))
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
        ttk.Label(parent, text="Dispositivo de Áudio:").grid(row=0, column=0, sticky="w", padx=5, pady=5); self.config_vars["device_index"] = tk.StringVar(); self.device_map = self.get_audio_devices(); ttk.Combobox(parent, textvariable=self.config_vars["device_index"], values=list(self.device_map.keys()), state="readonly").grid(row=0, column=1, sticky="ew")
        ttk.Label(parent, text="Agressividade VAD (0-3):").grid(row=1, column=0, sticky="w", padx=5, pady=5); self.config_vars["vad_aggressiveness"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["vad_aggressiveness"]).grid(row=1, column=1, sticky="ew")
        ttk.Label(parent, text="Gatilho de Voz (0.1-0.95):").grid(row=2, column=0, sticky="w", padx=5, pady=5); self.config_vars["vad_trigger_ratio"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["vad_trigger_ratio"]).grid(row=2, column=1, sticky="ew")
        ttk.Label(parent, text="Pausa de Silêncio (ms):").grid(row=3, column=0, sticky="w", padx=5, pady=5); self.config_vars["vad_silence_ms"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["vad_silence_ms"]).grid(row=3, column=1, sticky="ew")
    
    def create_autotune_widgets(self, parent):
        parent.columnconfigure(1, weight=1)
        self.config_vars["autotune_enabled"] = tk.BooleanVar()
        ttk.Checkbutton(parent, text="Habilitar Auto-Tune do VAD", variable=self.config_vars["autotune_enabled"], command=self._toggle_autotune_fields).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        self.config_vars["autotune_interval"] = tk.StringVar()
        self.config_vars["autotune_fp_ratio"] = tk.StringVar()
        self.config_vars["autotune_fp_chars"] = tk.StringVar()
        self.config_vars["autotune_noise_db"] = tk.StringVar()
        
        self.autotune_widgets = [
            (ttk.Label(parent, text="Intervalo de Ajuste (s):"), ttk.Entry(parent, textvariable=self.config_vars["autotune_interval"])),
            (ttk.Label(parent, text="Limiar Falso Positivo (%):"), ttk.Entry(parent, textvariable=self.config_vars["autotune_fp_ratio"])),
            (ttk.Label(parent, text="Tam. Mín. Texto (chars):"), ttk.Entry(parent, textvariable=self.config_vars["autotune_fp_chars"])),
            (ttk.Label(parent, text="Limiar Ruído (dBFS):"), ttk.Entry(parent, textvariable=self.config_vars["autotune_noise_db"]))
        ]
        for i, (label, entry) in enumerate(self.autotune_widgets):
            label.grid(row=i+1, column=0, sticky="w", padx=5, pady=2)
            entry.grid(row=i+1, column=1, sticky="ew", padx=5, pady=2)
            
    def _toggle_autotune_fields(self):
        state = "normal" if self.config_vars["autotune_enabled"].get() else "disabled"
        for label, entry in self.autotune_widgets:
            label.config(state=state); entry.config(state=state)
    
    def create_proc_widgets(self, parent):
        parent.columnconfigure(1, weight=1)
        self.config_vars["proc_enable_filter"] = tk.BooleanVar(); ttk.Checkbutton(parent, text="Habilitar Filtro Passa-Alta", variable=self.config_vars["proc_enable_filter"]).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        ttk.Label(parent, text="Frequência de Corte do Filtro (Hz):").grid(row=1, column=0, sticky="w", padx=5, pady=2); self.config_vars["proc_filter_cutoff"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["proc_filter_cutoff"]).grid(row=1, column=1, sticky="ew")
        self.config_vars["proc_enable_normalize"] = tk.BooleanVar(); ttk.Checkbutton(parent, text="Habilitar Normalização de Volume", variable=self.config_vars["proc_enable_normalize"]).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        ttk.Label(parent, text="Duração Mínima do Segmento (ms):").grid(row=3, column=0, sticky="w", padx=5, pady=2); self.config_vars["proc_min_duration"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["proc_min_duration"]).grid(row=3, column=1, sticky="ew")
    
    def create_overlay_widgets(self, parent):
        parent.columnconfigure(1, weight=1)
        mode_map = {"Overlay Dinâmico": "dinamico", "Overlay Estático": "estatico"}
        ttk.Label(parent, text="Modo do Overlay:").grid(row=0, column=0, sticky="w", padx=5, pady=5); self.config_vars["overlay_mode"] = tk.StringVar()
        mode_combo = ttk.Combobox(parent, textvariable=self.config_vars["overlay_mode"], values=list(mode_map.keys()), state="readonly"); mode_combo.grid(row=0, column=1, columnspan=2, sticky="ew"); mode_combo.bind("<<ComboboxSelected>>", self._toggle_static_fields)
        
        self.static_width_label = ttk.Label(parent, text="Largura Fixa (px):"); self.static_width_label.grid(row=1, column=0, sticky="w", padx=5, pady=2); self.config_vars["overlay_static_width"] = tk.StringVar(); self.static_width_entry = ttk.Entry(parent, textvariable=self.config_vars["overlay_static_width"]); self.static_width_entry.grid(row=1, column=1, columnspan=2, sticky="ew")
        self.static_height_label = ttk.Label(parent, text="Altura Fixa (px):"); self.static_height_label.grid(row=2, column=0, sticky="w", padx=5, pady=2); self.config_vars["overlay_static_height"] = tk.StringVar(); self.static_height_entry = ttk.Entry(parent, textvariable=self.config_vars["overlay_static_height"]); self.static_height_entry.grid(row=2, column=1, columnspan=2, sticky="ew")
        
        ttk.Separator(parent, orient='horizontal').grid(row=3, column=0, columnspan=3, sticky='ew', pady=10)
        
        ttk.Label(parent, text="Fonte:").grid(row=4, column=0, sticky="w", padx=5, pady=5); self.config_vars["font_family"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["font_family"]).grid(row=4, column=1, columnspan=2, sticky="ew")
        ttk.Label(parent, text="Tam. Fonte (Máx.):").grid(row=5, column=0, sticky="w", padx=5, pady=5); self.config_vars["font_size"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["font_size"]).grid(row=5, column=1, columnspan=2, sticky="ew")
        ttk.Label(parent, text="Cor da Fonte:").grid(row=6, column=0, sticky="w", padx=5, pady=5); self.config_vars["font_color"] = tk.StringVar(); font_color_entry = ttk.Entry(parent, textvariable=self.config_vars["font_color"]); font_color_entry.grid(row=6, column=1, sticky="ew"); ttk.Button(parent, text="...", width=4, command=lambda: self.choose_color("font_color")).grid(row=6, column=2, padx=5)
        ttk.Label(parent, text="Cor de Fundo:").grid(row=7, column=0, sticky="w", padx=5, pady=5); self.config_vars["bg_color"] = tk.StringVar(); bg_color_entry = ttk.Entry(parent, textvariable=self.config_vars["bg_color"]); bg_color_entry.grid(row=7, column=1, sticky="ew"); ttk.Button(parent, text="...", width=4, command=lambda: self.choose_color("bg_color")).grid(row=7, column=2, padx=5)
        ttk.Label(parent, text="Transparência (0.1-1.0):").grid(row=8, column=0, sticky="w", padx=5, pady=5); self.config_vars["alpha"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["alpha"]).grid(row=8, column=1, columnspan=2, sticky="ew")
        ttk.Label(parent, text="Tempo Mín. de Exibição (ms):").grid(row=9, column=0, sticky="w", padx=5, pady=5); self.config_vars["overlay_min_display"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["overlay_min_display"]).grid(row=9, column=1, columnspan=2, sticky="ew")
        ttk.Label(parent, text="Agrupar Legendas (ms):").grid(row=10, column=0, sticky="w", padx=5, pady=5); self.config_vars["overlay_merge_time"] = tk.StringVar(); ttk.Entry(parent, textvariable=self.config_vars["overlay_merge_time"]).grid(row=10, column=1, columnspan=2, sticky="ew")

    def _toggle_static_fields(self, event=None):
        is_static = self.config_vars["overlay_mode"].get() == "Overlay Estático"
        new_state = "normal" if is_static else "disabled"
        self.static_width_label.config(state=new_state); self.static_width_entry.config(state=new_state)
        self.static_height_label.config(state=new_state); self.static_height_entry.config(state=new_state)

    def choose_color(self, var_name):
        color_code = colorchooser.askcolor(title="Escolha uma cor");
        if color_code and color_code[1]: self.config_vars[var_name].set(color_code[1])

    def get_audio_devices(self):
        try: return {mic.name: mic.id for mic in sc.all_microphones(include_loopback=True)}
        except Exception as e: messagebox.showerror("Erro de Áudio", f"Não foi possível listar dispositivos.\n{e}"); return {"Nenhum dispositivo encontrado": ""}
    
    def load_settings(self):
        config = self.config_manager.config; ws = config.get("webservice_settings", {}); trans = config.get("translation_settings", {}); audio = config.get("audio_settings", {}); proc = config.get("audio_processing_settings", {}); overlay = config.get("overlay_settings", {}); autotune = config.get("autotune_settings", {})
        
        self.config_vars["ws_url"].set(ws.get("url", "")); self.config_vars["ws_form_name"].set(ws.get("file_form_name", "file")); self.config_vars["ws_resp_type"].set(ws.get("response_type", "json")); self.config_vars["ws_json_key"].set(ws.get("json_response_key", "text")); self.config_vars["ws_temperature"].set(ws.get("temperature", "0.0")); self.config_vars["ws_temperature_inc"].set(ws.get("temperature_inc", "0.2")); self.config_vars["ws_response_format_param"].set(ws.get("response_format_param", "json"))
        
        self.config_vars["trans_enabled"].set(trans.get("enabled", False)); self.config_vars["trans_host"].set(trans.get("ollama_host", "localhost")); self.config_vars["trans_port"].set(trans.get("ollama_port", "11434")); self.config_vars["trans_model"].set(trans.get("ollama_model", "llama3")); self.config_vars["trans_system_prompt"].delete("1.0", tk.END); self.config_vars["trans_system_prompt"].insert("1.0", trans.get("system_prompt", ""))
        
        self.config_vars["device_index"].set(next((name for name, id_ in self.device_map.items() if id_ == audio.get("device_index")), "Nenhum"))
        self.config_vars["vad_aggressiveness"].set(audio.get("vad_aggressiveness", 3)); self.config_vars["vad_trigger_ratio"].set(audio.get("vad_trigger_ratio", 0.75)); self.config_vars["vad_silence_ms"].set(audio.get("vad_silence_ms", 700))
        
        self.config_vars["autotune_enabled"].set(autotune.get("enabled", False)); self.config_vars["autotune_interval"].set(autotune.get("interval_sec", 30)); self.config_vars["autotune_fp_ratio"].set(int(autotune.get("fp_threshold_ratio", 0.3)*100)); self.config_vars["autotune_fp_chars"].set(autotune.get("fp_min_char_count", 5)); self.config_vars["autotune_noise_db"].set(autotune.get("noise_threshold_db", -40.0))
        
        self.config_vars["proc_enable_filter"].set(proc.get("enable_high_pass", True)); self.config_vars["proc_filter_cutoff"].set(proc.get("high_pass_cutoff_hz", 100.0)); self.config_vars["proc_enable_normalize"].set(proc.get("enable_normalization", True)); self.config_vars["proc_min_duration"].set(proc.get("min_segment_duration_ms", 250))
        
        mode_map_reversed = {"dinamico": "Overlay Dinâmico", "estatico": "Overlay Estático"}; self.config_vars["overlay_mode"].set(mode_map_reversed.get(overlay.get("mode", "dinamico"))); self.config_vars["overlay_static_width"].set(overlay.get("static_width", 400)); self.config_vars["overlay_static_height"].set(overlay.get("static_height", 150))
        self.config_vars["font_family"].set(overlay.get("font_family", "Arial")); self.config_vars["font_size"].set(overlay.get("font_size", 24)); self.config_vars["font_color"].set(overlay.get("font_color", "white")); self.config_vars["bg_color"].set(overlay.get("bg_color", "black")); self.config_vars["alpha"].set(overlay.get("alpha", 0.75))
        self.config_vars["overlay_min_display"].set(overlay.get("min_display_ms", 4000)); self.config_vars["overlay_merge_time"].set(overlay.get("merge_consecutive_ms", 1500))

        self._toggle_autotune_fields()
        self._toggle_static_fields()

    def save_and_restart(self):
        try:
            device_id = self.device_map.get(self.config_vars["device_index"].get())
            overlay_pos = (self.master.overlay.winfo_x(), self.master.overlay.winfo_y()) if hasattr(self.master, 'overlay') and self.master.overlay.winfo_exists() else (100, 800)
            cfg = self.config_manager.config
            cfg["webservice_settings"]["url"] = self.config_vars["ws_url"].get()
            cfg["translation_settings"] = {"enabled": self.config_vars["trans_enabled"].get(), "ollama_host": self.config_vars["trans_host"].get(), "ollama_port": self.config_vars["trans_port"].get(), "ollama_model": self.config_vars["trans_model"].get(), "system_prompt": self.config_vars["trans_system_prompt"].get("1.0", tk.END).strip()}
            cfg["audio_settings"] = {"device_index": device_id, "vad_aggressiveness": int(self.config_vars["vad_aggressiveness"].get()), "vad_trigger_ratio": float(self.config_vars["vad_trigger_ratio"].get()), "vad_silence_ms": int(self.config_vars["vad_silence_ms"].get())}
            cfg["autotune_settings"] = {"enabled": self.config_vars["autotune_enabled"].get(), "interval_sec": int(self.config_vars["autotune_interval"].get()), "fp_threshold_ratio": float(self.config_vars["autotune_fp_ratio"].get())/100, "fp_min_char_count": int(self.config_vars["autotune_fp_chars"].get()), "noise_threshold_db": float(self.config_vars["autotune_noise_db"].get())}
            cfg["audio_processing_settings"] = {"enable_high_pass": self.config_vars["proc_enable_filter"].get(), "high_pass_cutoff_hz": float(self.config_vars["proc_filter_cutoff"].get()), "enable_normalization": self.config_vars["proc_enable_normalize"].get(), "min_segment_duration_ms": int(self.config_vars["proc_min_duration"].get())}
            cfg["overlay_settings"] = {"mode": "dinamico" if self.config_vars["overlay_mode"].get() == "Overlay Dinâmico" else "estatico", "static_width": int(self.config_vars["overlay_static_width"].get()), "static_height": int(self.config_vars["overlay_static_height"].get()), "font_family": self.config_vars["font_family"].get(), "font_size": int(self.config_vars["font_size"].get()), "font_color": self.config_vars["font_color"].get(), "bg_color": self.config_vars["bg_color"].get(), "alpha": float(self.config_vars["alpha"].get()), "position_x": overlay_pos[0], "position_y": overlay_pos[1], "min_display_ms": int(self.config_vars["overlay_min_display"].get()), "merge_consecutive_ms": int(self.config_vars["overlay_merge_time"].get())}
            
            self.config_manager.save_config(cfg)
            self.destroy(); self.restart_callback()
        except Exception as e: messagebox.showerror("Erro ao Salvar", f"Verifique os valores inseridos.\n\nDetalhe: {e}")

class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__(); self.withdraw(); self.title("Controlador"); self.logo_image = None
        try: self.config_manager = ConfigManager()
        except Exception as e: root = tk.Tk(); root.withdraw(); messagebox.showerror("Erro Crítico", f"Erro ao carregar config: {e}\nFechando."); return
        self.audio_queue = queue.Queue(); self.transcribed_text_queue = queue.Queue(); self.final_text_queue = queue.Queue()
        self.threads = {}; self.overlay = None; self.settings_window = None
        self.create_control_window(); self.start_services()
    def create_control_window(self):
        self.control_window = tk.Toplevel(self); self.control_window.title("Controle"); self.control_window.protocol("WM_DELETE_WINDOW", self.on_exit); self.control_window.geometry("320x260"); self.control_window.resizable(False, False); self.control_window.pack_propagate(False)
        self.create_logo_widget(self.control_window)
        ttk.Button(self.control_window, text="Abrir Configurações", command=self.open_settings).pack(pady=10, padx=20, fill='x')
        self.status_label = ttk.Label(self.control_window, text="Inicializando...", foreground="blue", wraplength=280, justify="center"); self.status_label.pack(pady=10, padx=10, fill='x')
        ttk.Button(self.control_window, text="Sair", command=self.on_exit).pack(pady=10, padx=20)
    def create_logo_widget(self, parent):
        if not SVG_SUPPORT or not os.path.exists("rt-subs.svg"): return
        try:
            png_data = cairosvg.svg2png(url="rt-subs.svg", output_width=128, output_height=128)
            self.logo_image = ImageTk.PhotoImage(Image.open(io.BytesIO(png_data)))
            tk.Label(parent, image=self.logo_image, borderwidth=0).pack(pady=(10, 5))
        except Exception as e: print(f"ERRO ao carregar logo SVG: {e}")
    def update_status(self, message, color="black"):
        if hasattr(self, 'status_label') and self.status_label.winfo_exists():
            self.status_label.config(text=message, foreground=color if color != "purple" else "#8A2BE2")
    def start_services(self):
        self.stop_services()
        if self.overlay: self.overlay.destroy()
        for q in [self.audio_queue, self.transcribed_text_queue, self.final_text_queue]:
            while not q.empty(): q.get()
        
        vad_tuner = VADAutoTuner(self.config_manager, self.update_status)
        self.threads = {
            'tuner': vad_tuner,
            'audio': AudioRecorder(self.audio_queue, self.config_manager, self.update_status, vad_tuner),
            'transcription': TranscriptionProcessor(self.audio_queue, self.transcribed_text_queue, self.config_manager, self.update_status, vad_tuner),
            'translation': TranslationProcessor(self.transcribed_text_queue, self.final_text_queue, self.config_manager, self.update_status)
        }
        for thread in self.threads.values(): thread.start()
        self.overlay = OverlayWindow(self, self.final_text_queue, self.config_manager)
    def stop_services(self):
        for thread in self.threads.values():
            if thread and thread.is_alive(): thread.stop(); thread.join(timeout=1)
        self.threads.clear()
    def restart_services(self):
        self.update_status("Reiniciando...", "orange"); self.config_manager.load_config(); self.start_services()
    def open_settings(self):
        if not self.settings_window or not self.settings_window.winfo_exists():
            self.settings_window = SettingsWindow(self, self.config_manager, self.restart_services)
            self.settings_window.transient(self.control_window)
        else: self.settings_window.lift()
    def on_exit(self):
        self.update_status("Fechando...", "gray")
        if self.overlay and self.overlay.winfo_exists():
            cfg = self.config_manager.config; cfg["overlay_settings"]["position_x"], cfg["overlay_settings"]["position_y"] = self.overlay.winfo_x(), self.overlay.winfo_y()
            self.config_manager.save_config(cfg)
        self.stop_services(); self.destroy()

if __name__ == "__main__":
    try: import webrtcvad
    except ImportError: root = tk.Tk(); root.withdraw(); messagebox.showerror("Dependência Faltando", "Instale 'webrtcvad': pip install webrtcvad"); exit()
    try: import scipy
    except ImportError: root = tk.Tk(); root.withdraw(); messagebox.showerror("Dependência Faltando", "Instale 'scipy': pip install scipy"); exit()
    
    app = MainApplication()
    if hasattr(app, 'control_window') and app.control_window.winfo_exists():
        app.mainloop()