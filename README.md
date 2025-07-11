# RT-Subs: Real-Time Transcriber and Translator

**RT-Subs** is a desktop application that captures your system (or microphone) audio, transcribes it to text using a Whisper.cpp service, and optionally translates the text into other languages. The result is displayed in a customizable on-screen overlay.

The application is designed to be modular, responsive, and adaptive, ensuring the GUI never freezes, even during heavy audio and AI processing, and that its audio detection can adjust to changing environments.

[![Watch the video](https://img.youtube.com/vi/gTduokGR_II/maxresdefault.jpg)](https://www.youtube.com/watch?v=gTduokGR_II)

## Application Architecture

The architecture of RT-Subs is based on **multithreading** and the use of **Queues** for inter-thread communication. This design decouples the main tasks (audio capture, transcription, translation, and display), ensuring the user interface remains fluid and responsive.

The data flow can be visualized as follows:

```
                  <---------------------------------- [ Config Feedback ] <------------------------------------
                 |                                                                                             |
[Audio Device]   |                                                                                             |
       |         |                                                                                             |
       v         |                                                                                             |
[ Thread 1: AudioRecorder ] --(Audio Segments)--> [ Audio Queue ] --> [ Thread 2: TranscriptionProcessor ] --(Text)--> [ Transcription Queue ]
       |         ^                                                       |                                                                   |
(Noise Level)    |                                                  (Text Quality)                                                             v
       |         |                                                       |                                        [ Thread 3: TranslationProcessor ] --(Final Text)--> [ Final Queue ]
       |         |                                                       v                                                                                                  |
       |---------|----------------> [ Thread 4: VADAutoTuner ] <---------|                                                                                                  v
                                                                                                                                                                  [ GUI: OverlayWindow ]
```

### Multithreading and Queues

The application is divided into a main thread (GUI) and four worker threads:

1.  **Main Thread (MainApplication)**: Manages the control window and user interface, starting and stopping the worker threads.
2.  **`AudioRecorder` (Thread 1)**: Captures audio using VAD (Voice Activity Detection) and sends speech segments to the `Audio Queue`. It also reports ambient noise levels to the `VADAutoTuner`.
3.  **`TranscriptionProcessor` (Thread 2)**: Takes audio segments, sends them to the **Whisper.cpp** server, and places the resulting text into the `Transcription Queue`. It reports the quality of the transcription (e.g., if it was empty or too short) to the `VADAutoTuner`.
4.  **`TranslationProcessor` (Thread 3)**: Takes the transcribed text and, if LLM translation is enabled, sends it to the **Ollama** server. Otherwise, it simply passes the text through.
5.  **`VADAutoTuner` (Thread 4)**: This thread creates a **feedback loop**. It analyzes the noise levels and transcription quality metrics over time. If it detects too much noise or too many "false positives" (transcriptions of silence), it automatically adjusts the VAD parameters to make it stricter, improving accuracy in changing environments.
6.  **`OverlayWindow` (GUI)**: Takes the text from the `Final Queue` and displays it on the screen in a non-blocking way.

This "producer-consumer-analyzer" model ensures that slow tasks, like network I/O, do not freeze the application, while also allowing it to adapt intelligently.

## Installation Procedures

### 1. Prerequisites

*   Python 3.8 or higher.
*   Git.
*   An NVIDIA GPU with **at least 17 GB of VRAM** is recommended for the full setup (Whisper + LLM Translation). The VRAM is distributed as follows:
    *   **~6 GB** for Whisper.cpp (with the `large-v3` model).
    *   **~11 GB** for the Phi-4 translation model via Ollama.

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

1.  **Download and compile Whisper.cpp:** Follow the instructions on the [official repository](https://github.com/ggerganov/whisper.cpp). Compilation with CUDA/CUBLAS support is essential for performance on NVIDIA GPUs.

2.  **Download a model:** The `large-v3` model is recommended for high quality.
    ```bash
    # Inside the whisper.cpp folder
    bash ./models/download-ggml-model.sh large-v3
    ```

3.  **Start the server:**
    *   **For simple transcription:**
        ```bash
        ./server -m models/ggml-large-v3.bin --host 0.0.0.0 --port 8080
        ```
    *   **For transcription AND translation to English:** Whisper itself can translate audio from any language into English. Use the `--translate` parameter.
        ```bash
        ./server -m models/ggml-large-v3.bin --host 0.0.0.0 --port 8080 --translate
        ```
    The server will consume about **6 GB of VRAM**.

#### **Ollama (for Advanced Translation)**

Use Ollama if you need to translate into languages **other than English** (such as Brazilian Portuguese) or if you want finer control over the process.

1.  **Install Ollama:** Download and install it from the official website: [ollama.com](https://ollama.com/).

2.  **Download the translation model:** The `phi4:14b` model with `q4_K_M` quantization is an excellent choice.
    ```bash
    ollama pull phi4:14b
    ```

3.  **Ensure Ollama is running:** Ollama typically runs as a background service. It will load the model into memory (consuming about **11 GB of VRAM**) on its first use.

### 4. Running the Application

After installing the Python dependencies and ensuring the necessary servers are running, start the application:

```bash
python main.py
```

The "Control" window will appear, and the system will start listening for audio.

---

## Configuration Parameters (`config.json`)

### `webservice_settings`

Settings for communication with the **Whisper.cpp** server.

*   `"url"`: The server address. E.g., `"http://localhost:8080/inference"`.
*   `"file_form_name"`: The form field name. Default: `"file"`.
*   `"response_type"`: The response format. Options: `"json"` or `"text"`.
*   `"json_response_key"`: The key to extract text from the JSON response. Default: `"text"`.

### `translation_settings`

Settings for translation with **Ollama**.

*   `"enabled"`: Enables (`true`) or disables (`false`) translation via Ollama. **Keep this `false` if you are using Whisper's native translation-to-English feature.**
*   `"ollama_host"`: Ollama server address. Default: `"localhost"`.
*   `"ollama_port"`: Ollama server port. Default: `"11434"`.
*   `"ollama_model"`: The model name in Ollama. E.g., `"phi4:14b"`.
*   `"system_prompt"`: The main instruction given to the language model to guide the translation.

### `audio_settings`

Audio capture and VAD settings.

*   `"device_index"`: Audio device ID. `"default_mic"` for microphone, `"default_loopback"` for system audio.
*   `"vad_aggressiveness"`: Aggressiveness of the voice detector (0-3). `3` is stricter and less sensitive to noise.
*   `"vad_trigger_ratio"`: The ratio of voice frames needed to start recording (0.0 to 1.0). Higher values require a clearer voice signal.
*   `"vad_silence_ms"`: Milliseconds of silence to end a speech segment.

### `autotune_settings` (New Feature)

Settings for the VAD Auto-Tune feature.

*   `"enabled"`: Enables (`true`) or disables (`false`) the automatic adjustment of VAD parameters.
*   `"interval_sec"`: How often (in seconds) the tuner analyzes metrics to decide if an adjustment is needed.
*   `"fp_threshold_ratio"`: The threshold of "bad" transcriptions (as a percentage, e.g., 0.3 for 30%) that triggers an adjustment. A bad transcription is one that is too short.
*   `"fp_min_char_count"`: The minimum number of characters a transcription must have to be considered valid. Anything below this is counted as a false positive.
*   `"noise_threshold_db"`: The ambient noise level (in dBFS) above which the VAD will be made stricter.

### `audio_processing_settings`

Audio pre-processing settings.

*   `"enable_high_pass"`: Enables (`true`) a filter to remove low-frequency noise.
*   `"enable_normalization"`: Enables (`true`) audio volume normalization.
*   `"min_segment_duration_ms"`: Discards audio segments shorter than this value to avoid processing noise.

### `overlay_settings`

Visual settings for the overlay.

*   `"mode"`: Sets the overlay behavior.
    *   `"dinamico"`: The overlay window resizes automatically to fit the transcribed text.
    *   `"estatico"`: The overlay has a fixed size. If the text is too long, the **font size will shrink** to make it fit.
*   `"static_width"`, `"static_height"`: The width and height (in pixels) for the overlay when in `"estatico"` mode.
*   `"font_family"`, `"font_color"`: Font styles.
*   `"font_size"`: The base (or maximum) font size.
*   `"bg_color"`: Background color.
*   `"alpha"`: Transparency level (0.1 for mostly transparent, 1.0 for opaque).
*   `"position_x"`, `"position_y"`: Initial overlay position (saved automatically on exit).
*   `"min_display_ms"`: Minimum time a text remains on the screen before clearing.
*   `"merge_consecutive_ms"`: If a new transcription arrives within this time, it is appended to the previous one instead of replacing it.