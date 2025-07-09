# RT-Subs: Real-Time Transcriber and Translator

# RT-Subs: Real-Time Transcriber and Translator

**RT-Subs** is a desktop application that captures your system (or microphone) audio, transcribes it to text using a Whisper.cpp service, and optionally translates the text into other languages. The result is displayed in a customizable on-screen overlay.

The application is designed to be modular and responsive, ensuring the GUI never freezes, even during heavy audio and AI processing.

## Application Architecture

The architecture of RT-Subs is based on **multithreading** and the use of **Queues** for inter-thread communication. This design decouples the main tasks (audio capture, transcription, translation, and display), ensuring the user interface remains fluid and responsive.

The data flow can be visualized as follows:

```
[Audio Device]
       |
       v
[ Thread 1: AudioRecorder ] --(Audio Segments)--> [ Audio Queue ]
                                                       |
                                                       v
                         [ Thread 2: TranscriptionProcessor ] --(Transcribed Text)--> [ Transcription Queue ]
                                                                                            |
                                                                                            v
                                      [ Thread 3: TranslationProcessor ] --(Translated Text)--> [ Final Queue ]
                                                                                                      |
                                                                                                      v
                                                                                            [ GUI: OverlayWindow ]
```

### Multithreading and Queues

The application is divided into a main thread (GUI) and three worker threads:

1. **Main Thread (MainApplication)**: Manages the control window and user interface, starting and stopping the worker threads.
2. **`AudioRecorder` (Thread 1)**: Captures audio using VAD (Voice Activity Detection) and sends speech segments to the `Audio Queue`.
3. **`TranscriptionProcessor` (Thread 2)**: Takes audio segments, sends them to the **Whisper.cpp** server, and places the resulting text into the `Transcription Queue`.
4. **`TranslationProcessor` (Thread 3)**: Takes the transcribed text and, if LLM translation is enabled, sends it to the **Ollama** server. Otherwise, it simply passes the text through.
5. **`OverlayWindow` (GUI)**: Takes the text from the `Final Queue` and displays it on the screen in a non-blocking way.

This "producer-consumer" model ensures that slow tasks, like network I/O, do not freeze the audio capture or the interface.

## Installation Procedures

### 1. Prerequisites

* Python 3.8 or higher.
* Git.
* An NVIDIA GPU with **at least 17 GB of VRAM** is recommended for the full setup (Whisper + LLM Translation). The VRAM is distributed as follows:
  * **~6 GB** for Whisper.cpp (with the `large-v3` model).
  * **~11 GB** for the Phi-4 translation model via Ollama.

### 2. Installing the RT-Subs Application

```bash
# 1. Clone the repository
git clone https://github.com/your-username/your-repository.git
cd your-repository

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
# On Windows:
# .\venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

### 3. Setting Up External Dependencies

RT-Subs communicates with local servers that you need to set up: **Whisper.cpp** and (optionally) **Ollama**.

#### **Whisper.cpp (for Transcription and Translation to English)**

1. **Download and compile Whisper.cpp:** Follow the instructions on the [official repository](https://github.com/ggerganov/whisper.cpp). Compilation with CUDA/CUBLAS support is essential for performance on NVIDIA GPUs.

2. **Download a model:** The `large-v3` model is recommended for high quality.
   
   ```bash
   # Inside the whisper.cpp folder
   bash ./models/download-ggml-model.sh large-v3
   ```

3. **Start the server:**
   
   * **For simple transcription:**
     
     ```bash
     ./server -m models/ggml-large-v3.bin --host 0.0.0.0 --port 8080
     ```
   
   * **For transcription AND translation to English:** Whisper itself can translate audio from any language into English. Use the `--translate` parameter.
     
     ```bash
     ./server -m models/ggml-large-v3.bin --host 0.0.0.0 --port 8080 --translate
     ```
     
     The server will consume about **6 GB of VRAM**.

#### **Ollama (for Advanced Translation)**

Use Ollama if you need to translate into languages **other than English** (such as Brazilian Portuguese) or if you want finer control over the process.

1. **Install Ollama:** Download and install it from the official website: [ollama.com](https://ollama.com/).

2. **Download the translation model:** The `phi4:14b` model with `q4_K_M` quantization is an excellent choice.
   
   ```bash
   ollama pull phi4:14b
   ```

3. **Ensure Ollama is running:** Ollama typically runs as a background service. It will load the model into memory (consuming about **11 GB of VRAM**) on its first use.

### 4. Running the Application

After installing the Python dependencies and ensuring the necessary servers are running, start the application:

```bash
python app.py
```

The "Control" window will appear, and the system will start listening for audio.

---

## Configuration Parameters (`config.json`)

### `webservice_settings`

Settings for communication with the **Whisper.cpp** server.

* `"url"`: The server address. E.g., `"http://localhost:8080/inference"`.
* `"file_form_name"`: The form field name. Default: `"file"`.
* `"response_type"`: The response format. Options: `"json"` or `"text"`.
* `"json_response_key"`: The key to extract text from the JSON response. Default: `"text"`.

### `translation_settings`

Settings for translation with **Ollama**.

* `"enabled"`: Enables (`true`) or disables (`false`) translation via Ollama. **Keep this `false` if you are using Whisper's native translation-to-English feature.**
* `"ollama_host"`: Ollama server address. Default: `"localhost"`.
* `"ollama_port"`: Ollama server port. Default: `"11434"`.
* `"ollama_model"`: The model name in Ollama. E.g., `"phi4:14b"`.
* `"system_prompt"`: The main instruction given to the language model to guide the translation.

### `audio_settings`

Audio capture and VAD settings.

* `"device_index"`: Audio device ID. `"default_mic"` for microphone, `"default_loopback"` for system audio.
* `"vad_aggressiveness"`: Aggressiveness of the voice detector (0-3). `3` is stricter.
* `"vad_trigger_ratio"`: The ratio of voice frames to start recording (0.0 to 1.0).
* `"vad_silence_ms"`: Milliseconds of silence to end a segment.

### `audio_processing_settings`

Audio pre-processing settings.

* `"enable_high_pass"`: Enables (`true`) a filter to remove low-frequency noise.
* `"high_pass_cutoff_hz"`: The filter's cutoff frequency.
* `"enable_normalization"`: Enables (`true`) audio volume normalization.
* `"min_segment_duration_ms"`: Discards segments shorter than this value to avoid noise.

### `overlay_settings`

Visual settings for the overlay.

* `"font_family"`, `"font_size"`, `"font_color"`: Font styles.
* `"bg_color"`: Background color.
* `"alpha"`: Transparency level (0.1 to 1.0).
* `"position_x"`, `"position_y"`: Initial overlay position (saved automatically on exit).
* `"min_display_ms"`: Minimum time a text remains on the screen.
* `"merge_consecutive_ms"`: If a new transcription arrives within this time, it is concatenated with the previous one.