import sys
import os
import wave
import struct
import configparser
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
import logging

import pyaudio
import webrtcvad
import pyautogui
import pyperclip
import pystray
from PIL import Image, ImageDraw
from pynput import mouse, keyboard
from pynput.mouse import Button
from pynput.keyboard import Key, KeyCode, Listener as KBListener
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Для работы с сервером
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

# Попытка импорта keyboard для вставки
try:
    import keyboard as kb
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    kb = None

# ----------------------------------------------------------------------
# Динамическое определение путей
# RESOURCE_DIR — чтение ресурсов (whisper-cli, модели) — рядом с EXE
# DATA_DIR    — запись данных (конфиг, логи, записи) — %APPDATA%
# ----------------------------------------------------------------------
def get_resource_dir():
    """Путь к ресурсам программы — работает и в разработке, и в собранном EXE."""
    try:
        # PyInstaller 6.0+ — файлы в _internal, путь в _MEIPASS
        return Path(sys._MEIPASS)
    except AttributeError:
        # Запущено как скрипт
        return Path(__file__).parent

RESOURCE_DIR = get_resource_dir()

# DATA_DIR — пользовательские данные в %APPDATA%\WhisperDictation
APPDATA_DIR = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
DATA_DIR = APPDATA_DIR / "WhisperDictation"

# ----------------------------------------------------------------------
# Пути к данным (запись) — в DATA_DIR
# ----------------------------------------------------------------------
CONFIG_PATH = DATA_DIR / "config.ini"
RECORDINGS_DIR = DATA_DIR / "recordings"
TEMP_WAV = RECORDINGS_DIR / "temp_record.wav"
CHUNK_DIR = RECORDINGS_DIR / "chunks"
LOG_PATH = DATA_DIR / "error.log"

# Создаём папки ДО настройки логирования
DATA_DIR.mkdir(parents=True, exist_ok=True)
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Логирование (ПОСЛЕ создания папок)
# ----------------------------------------------------------------------
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

# ----------------------------------------------------------------------
# Утилита для путей
# ----------------------------------------------------------------------
def resolve_path(path_str):
    """Если путь относительный — добавить RESOURCE_DIR (для чтения ресурсов)."""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(RESOURCE_DIR / p)

# ----------------------------------------------------------------------
# Конфигурация (относительные пути)
# ----------------------------------------------------------------------
DEFAULT_CONFIG = {
    "Paths": {
        "whisper_gpu": "whisper-vulkan\\whisper-cli.exe",
        "whisper_cpu": "whisper-CPU\\whisper-cli.exe",
        "model": "models\\ggml-large-v3-turbo-q8_0.bin",
        "gpu_device_index": "1",
        "server_path": "whisper-vulkan-Server\\whisper-server.exe",
    },
    "Recognition": {
        "device": "gpu",
        "mode": "record_then_analyze",
        "language": "ru",
        "method": "whisper-server",   # "whisper-cli" или "whisper-server"
        "server_url": "http://127.0.0.1:8080/inference",
        "use_vad": "false",
    },
    "Audio": {
        "gain": "2.0",
        "sample_rate": "16000",
    },
    "Hotkeys": {
        "hotkey": "middle_mouse",
    },
}

def load_config():
    config = configparser.ConfigParser()
    if CONFIG_PATH.exists():
        config.read(CONFIG_PATH, encoding="utf-8")
        for section, values in DEFAULT_CONFIG.items():
            if not config.has_section(section):
                config.add_section(section)
            for key, val in values.items():
                if not config.has_option(section, key):
                    config.set(section, key, val)
        save_config(config)
    else:
        config.read_dict(DEFAULT_CONFIG)
        save_config(config)
    return config

def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        config.write(f)
    logging.info("Конфиг сохранён")

# ----------------------------------------------------------------------
# Аудио захват
# ----------------------------------------------------------------------
class AudioRecorder:
    def __init__(self, config):
        self.config = config
        self.p = pyaudio.PyAudio()
        self.device_index = self.p.get_default_input_device_info()["index"]
        dev_info = self.p.get_device_info_by_index(self.device_index)
        target_rate = int(config["Audio"].get("sample_rate", "16000"))
        supported_rates = self._get_supported_sample_rates(dev_info)
        if target_rate in supported_rates:
            self.rate = target_rate
        else:
            self.rate = min(supported_rates, key=lambda x: abs(x - target_rate))
            logging.warning(f"Устройство не поддерживает {target_rate} Гц, используется {self.rate} Гц")
        self.channels = 1  # Запись всегда в моно — достаточно для речи, работает с VAD
        self.format = pyaudio.paInt16
        self.chunk = 1024
        self.gain = 2.0

    def _get_supported_sample_rates(self, dev_info):
        common_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000]
        supported = []
        for rate in common_rates:
            try:
                self.p.is_format_supported(
                    rate, input_device=self.device_index,
                    input_channels=self.channels,
                    input_format=self.format
                )
                supported.append(rate)
            except:
                pass
        if not supported:
            supported = [16000]
        return supported

    def set_gain(self, gain):
        self.gain = max(1.0, min(10.0, float(gain)))

    def record_to_file(self, filepath, stop_event):
        stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk,
        )
        frames = []
        try:
            while not stop_event.is_set():
                data = stream.read(self.chunk, exception_on_overflow=False)
                if self.gain != 1.0:
                    samples = struct.unpack(f"<{len(data)//2}h", data)
                    amplified = []
                    for s in samples:
                        ns = int(s * self.gain)
                        if ns > 32767:
                            ns = 32767
                        elif ns < -32768:
                            ns = -32768
                        amplified.append(ns)
                    data = struct.pack(f"<{len(amplified)}h", *amplified)
                frames.append(data)
        except Exception as e:
            logging.error(f"Ошибка записи: {e}")
        finally:
            stream.stop_stream()
            stream.close()

        if frames:
            wf = wave.open(str(filepath), "wb")
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(frames))
            wf.close()
            logging.info(f"Записано {len(frames)} фреймов, файл {filepath}, частота {self.rate} Гц")
        else:
            logging.warning("Нет записанных данных")

    def close(self):
        self.p.terminate()

# ----------------------------------------------------------------------
# Whisper Runner (поддерживает whisper-cli и whisper-server с автозапуском)
# ----------------------------------------------------------------------
class WhisperRunner:
    def __init__(self, config):
        self.config = config
        self.method = config["Recognition"].get("method", "whisper-cli")
        self.server_url = config["Recognition"].get("server_url", "http://127.0.0.1:8080/inference")
        self.use_vad = config.getboolean("Recognition", "use_vad", fallback=False)
        self.server_process = None
        self.update_paths()
        self._auto_start_server_if_needed()

    def _auto_start_server_if_needed(self):
        """Если выбран метод whisper-server, запускает сервер и ждёт его готовности."""
        if self.method != "whisper-server" or not REQUESTS_AVAILABLE:
            return

        # Проверяем, не запущен ли уже сервер
        try:
            r = requests.get("http://127.0.0.1:8080/health", timeout=1)
            if r.status_code == 200:
                logging.info("Сервер уже запущен, подключаемся")
                return
        except:
            pass

        # Запускаем сервер
        server_exe = resolve_path(self.config["Paths"].get("server_path", "whisper-vulkan-Server\\whisper-server.exe"))
        if not os.path.exists(server_exe):
            logging.error(f"whisper-server.exe не найден: {server_exe}")
            # Переключаемся на whisper-cli как fallback
            self.method = "whisper-cli"
            return

        model_path = resolve_path(self.config["Paths"]["model"])
        device = self.config["Paths"].get("gpu_device_index", "1")
        language = self.config["Recognition"]["language"]
        vad_model_path = str(RESOURCE_DIR / "models" / "for-tests-silero-v6.2.0-ggml.bin")

        cmd = [
            server_exe,
            "-m", model_path,
            "--device", device,
            "--host", "127.0.0.1",
            "--port", "8080",
            "-t", "6",
            "-l", language,
        ]

        # VAD-флаги для сервера
        if self.use_vad:
            cmd += ["--vad", "--vad-model", vad_model_path]
        logging.info(f"Запуск сервера: {' '.join(cmd)}")

        try:
            # Запускаем сервер в скрытом режиме
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except Exception as e:
            logging.error(f"Не удалось запустить сервер: {e}")
            self.method = "whisper-cli"
            return

        # Ждём, пока сервер загрузится (проверяем /health)
        logging.info("Ожидание загрузки сервера...")
        for i in range(30):  # максимум 30 попыток (~30 секунд)
            time.sleep(1)
            try:
                r = requests.get("http://127.0.0.1:8080/health", timeout=1)
                if r.status_code == 200:
                    logging.info("Сервер готов к работе")
                    return
            except:
                pass
            if i % 5 == 0:
                logging.info(f"Ожидание сервера... {i+1} сек")
        # Если сервер не запустился, переключаемся на cli
        logging.error("Сервер не запустился вовремя, переключение на whisper-cli")
        self.method = "whisper-cli"
        self._stop_server()

    def _stop_server(self):
        """Завершает процесс сервера."""
        if self.server_process and self.server_process.poll() is None:
            self.server_process.terminate()
            self.server_process.wait(timeout=3)
            logging.info("Сервер остановлен")
            self.server_process = None

    def update_paths(self):
        device = self.config["Recognition"]["device"]
        if device == "gpu":
            self.whisper_exe = resolve_path(self.config["Paths"]["whisper_gpu"])
            gpu_idx = self.config["Paths"].get("gpu_device_index", "0")
            self.device_flag = ["--device", gpu_idx]
        else:
            self.whisper_exe = resolve_path(self.config["Paths"]["whisper_cpu"])
            self.device_flag = []  # CPU-версия не поддерживает --device
        self.model_path = resolve_path(self.config["Paths"]["model"])
        self.language = self.config["Recognition"]["language"]
        self.method = self.config["Recognition"].get("method", "whisper-cli")
        self.server_url = self.config["Recognition"].get("server_url", "http://127.0.0.1:8080/inference")
        self.use_vad = self.config.getboolean("Recognition", "use_vad", fallback=False)
        logging.info(f"Whisper обновлён: method={self.method}, model={self.model_path}, device_flag={self.device_flag}, use_vad={self.use_vad}")

        # Если изменился метод, перезапускаем сервер при необходимости
        if self.method == "whisper-server":
            self._auto_start_server_if_needed()

    def transcribe(self, wav_path):
        if self.method == "whisper-server" and REQUESTS_AVAILABLE:
            try:
                with open(wav_path, 'rb') as f:
                    files = {'file': (os.path.basename(wav_path), f, 'audio/wav')}
                    response = requests.post(self.server_url, files=files, timeout=120)
                if response.status_code == 200:
                    result = response.json()
                    text = result.get('text', '').strip()
                    if not text:
                        text = result.get('result', '').strip()
                    return text
                else:
                    logging.error(f"Ошибка сервера: {response.status_code} - {response.text}")
                    return self._transcribe_with_cli(wav_path)
            except Exception as e:
                logging.error(f"Ошибка при запросе к серверу: {e}")
                return self._transcribe_with_cli(wav_path)
        else:
            return self._transcribe_with_cli(wav_path)

    def _transcribe_with_cli(self, wav_path):
        if not os.path.exists(self.whisper_exe):
            raise FileNotFoundError(f"whisper-cli не найден: {self.whisper_exe}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")

        # Путь к файлу .txt, который создаст whisper-cli
        base_name = str(wav_path).rsplit(".", 1)[0]
        output_txt = base_name + ".txt"

        cmd = [
            self.whisper_exe,
            "-m", self.model_path,
            "-f", str(wav_path),
            "-l", self.language,
            "--no-timestamps",
            "-otxt",
        ] + self.device_flag

        # VAD-флаги для whisper-cli
        if self.use_vad:
            vad_model_path = str(RESOURCE_DIR / "models" / "for-tests-silero-v6.2.0-ggml.bin")
            cmd += ["--vad", "--vad-model", vad_model_path]

        logging.info(f"Запуск команды: {' '.join(cmd)}")
        try:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
                timeout=120,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            # Логирование stderr для диагностики
            if result.stderr:
                logging.warning(f"whisper-cli stderr: {result.stderr.strip()[:500]}")
            if result.stdout:
                logging.debug(f"whisper-cli stdout: {result.stdout.strip()[:500]}")
            
            # Читаем результат из файла
            if os.path.exists(output_txt):
                with open(output_txt, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                try:
                    os.remove(output_txt)
                    logging.debug(f"Файл .txt удалён: {output_txt}")
                except Exception as e:
                    logging.warning(f"Не удалось удалить {output_txt}: {e}")
                return text
            else:
                logging.warning(f"Файл {output_txt} не создан")
                # Попробовать вернуть текст из stdout как fallback
                stdout_text = result.stdout.strip() if result.stdout else ""
                if stdout_text:
                    logging.info("Текст получен из stdout")
                    return stdout_text
                return ""
        except subprocess.TimeoutExpired:
            logging.error("Превышено время ожидания")
            return "[Ошибка: превышено время ожидания]"
        except Exception as e:
            logging.error(f"Ошибка whisper: {e}")
            return f"[Ошибка: {e}]"

    def shutdown(self):
        """Останавливает сервер при завершении программы."""
        self._stop_server()

# ----------------------------------------------------------------------
# Ядро диктовки
# ----------------------------------------------------------------------
class DictationCore:
    def __init__(self, config):
        self.config = config
        self.audio = AudioRecorder(config)
        self.runner = WhisperRunner(config)
        self.recording_thread = None
        self.stop_event = threading.Event()
        self.mode = config["Recognition"]["mode"]
        self.full_buffer = []
        self.is_recording = False
        self.is_running = True

    def update_settings(self, config):
        self.config = config
        self.audio.set_gain(float(config["Audio"]["gain"]))
        self.runner.update_paths()
        self.mode = config["Recognition"]["mode"]
        logging.info("Настройки обновлены")

    def start_recording(self):
        if not self.is_running:
            return
        if self.recording_thread and self.recording_thread.is_alive():
            return
        self.stop_event.clear()
        if self.mode == "record_then_analyze":
            self.recording_thread = threading.Thread(
                target=self._record_and_transcribe, daemon=True
            )
            self.recording_thread.start()
        else:
            self.full_buffer = []
            self.is_recording = True
            self.recording_thread = threading.Thread(
                target=self._streaming_record, daemon=True
            )
            self.recording_thread.start()
        logging.info("Запись начата")

    def stop_recording(self):
        if self.recording_thread is None:
            return
        self.stop_event.set()
        if self.mode == "streaming" and self.is_recording:
            self._process_remaining_buffer()
        self.recording_thread.join(timeout=2.0)
        self.recording_thread = None
        self.is_recording = False
        # Очистка оставшихся чанков после вставки всего текста
        self._cleanup_chunks()
        logging.info("Запись остановлена")

    def _cleanup_chunks(self):
        """Удаляет все временные чанки из папки (.txt уже удалены в _transcribe_with_cli)."""
        cleaned = 0
        for f in CHUNK_DIR.glob("chunk_*.wav"):
            try:
                f.unlink()
                cleaned += 1
            except Exception as e:
                logging.warning(f"Не удалось удалить {f}: {e}")
        if cleaned:
            logging.info(f"Временные чанки очищены: {cleaned} файлов")

    def _process_remaining_buffer(self):
        if not self.full_buffer:
            return
        logging.info(f"Обработка остатка буфера: {len(self.full_buffer)} фреймов")
        self._process_chunk(self.full_buffer)
        self.full_buffer = []

    def _process_chunk(self, frames):
        if not frames:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        chunk_path = CHUNK_DIR / f"chunk_{timestamp}.wav"
        amplified_frames = []
        for data in frames:
            if self.audio.gain != 1.0:
                samples = struct.unpack(f"<{len(data)//2}h", data)
                amp = []
                for s in samples:
                    ns = int(s * self.audio.gain)
                    if ns > 32767:
                        ns = 32767
                    elif ns < -32768:
                        ns = -32768
                    amp.append(ns)
                data = struct.pack(f"<{len(amp)}h", *amp)
            amplified_frames.append(data)

        wf = wave.open(str(chunk_path), "wb")
        wf.setnchannels(self.audio.channels)
        wf.setsampwidth(self.audio.p.get_sample_size(self.audio.format))
        wf.setframerate(self.audio.rate)
        wf.writeframes(b"".join(amplified_frames))
        wf.close()

        text = self.runner.transcribe(chunk_path)
        if text and not text.startswith("[Ошибка"):
            self._paste_text(text + " ")
        # Удаление чанка (.txt уже удалён в _transcribe_with_cli)
        try:
            os.remove(chunk_path)
            logging.debug(f"Чанк удалён: {chunk_path}")
        except Exception as e:
            logging.warning(f"Не удалось удалить чанк {chunk_path}: {e}")

    def _record_and_transcribe(self):
        wav_path = TEMP_WAV
        try:
            self.audio.record_to_file(wav_path, self.stop_event)
        except Exception as e:
            logging.error(f"Ошибка записи: {e}")
            self._paste_text(f"[Ошибка записи: {e}]")
            return
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            text = self.runner.transcribe(wav_path)
            self._paste_text(text)
            # Удаление временного WAV (.txt уже удалён в _transcribe_with_cli)
            try:
                os.remove(wav_path)
                logging.debug(f"Временный WAV удалён: {wav_path}")
            except Exception as e:
                logging.warning(f"Не удалось удалить {wav_path}: {e}")
        else:
            logging.warning("Файл пуст или не создан")
            self._paste_text("[Нет записанных данных]")

    def _streaming_record(self):
        """Стриминговая запись с VAD (webrtcvad) — разделение по паузам речи."""
        vad = webrtcvad.Vad(1)  # агрессивность 0-3, 1 = баланс
        FRAME_DURATION_MS = 20
        frame_size = int(self.audio.rate * FRAME_DURATION_MS / 1000) * 2  # 2 байта на сэмпл (int16)
        # Для частоты 16000: frame_size = 640 байт (320 сэмплов * 2 байта)
        
        audio_buffer = bytearray()
        is_speaking = False
        silence_counter = 0
        # Порог тишины: 0.8 сек / 0.020 сек = 40 кадров
        SILENCE_THRESHOLD = int(0.8 * 1000 / FRAME_DURATION_MS)
        # Таймер принудительной отправки: 10 сек
        last_send_time = time.time()
        FORCE_SEND_INTERVAL = 10.0
        
        vad_buffer = bytearray()
        stream = self.audio.p.open(
            format=self.audio.format,
            channels=1,  # webrtcvad требует моно
            rate=self.audio.rate,
            input=True,
            input_device_index=self.audio.device_index,
            frames_per_buffer=self.audio.chunk,
        )

        try:
            while not self.stop_event.is_set() and self.is_running:
                data = stream.read(self.audio.chunk, exception_on_overflow=False)
                audio_buffer += data
                vad_buffer += data
                
                # Обработка VAD-кадров
                while len(vad_buffer) >= frame_size:
                    frame = bytes(vad_buffer[:frame_size])
                    vad_buffer = vad_buffer[frame_size:]
                    
                    is_speech = vad.is_speech(frame, self.audio.rate)
                    
                    if is_speech:
                        is_speaking = True
                        silence_counter = 0
                    else:
                        silence_counter += 1
                    
                    # Пауза после речи — отправляем чанк
                    if silence_counter >= SILENCE_THRESHOLD and is_speaking:
                        self._process_chunk(audio_buffer)
                        audio_buffer = bytearray()
                        is_speaking = False
                        silence_counter = 0
                        vad_buffer = bytearray()
                        last_send_time = time.time()
                
                # Принудительная отправка через 10 сек (защита от зависания)
                if len(audio_buffer) > 0 and (time.time() - last_send_time) >= FORCE_SEND_INTERVAL:
                    self._process_chunk(audio_buffer)
                    audio_buffer = bytearray()
                    is_speaking = False
                    silence_counter = 0
                    vad_buffer = bytearray()
                    last_send_time = time.time()
        except Exception as e:
            logging.error(f"Ошибка в потоковой записи: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            # Отправка остатка буфера
            if len(audio_buffer) > 0:
                self._process_chunk(audio_buffer)

    def _paste_text(self, text):
        if not text or text.startswith("[Ошибка]"):
            logging.warning(f"Не вставляем текст: {text}")
            return
        pyperclip.copy(text)
        time.sleep(0.15)
        if KEYBOARD_AVAILABLE:
            try:
                kb.send("ctrl+v")
                logging.info("Вставка через keyboard.send")
            except Exception as e:
                logging.warning(f"keyboard.send не сработал: {e}")
                pyautogui.hotkey("ctrl", "v", interval=0.1)
                logging.info("Вставка через pyautogui.hotkey")
        else:
            pyautogui.hotkey("ctrl", "v", interval=0.1)
            logging.info("Вставка через pyautogui.hotkey")
        # Очистка буфера обмена после вставки
        time.sleep(0.1)
        pyperclip.copy("")
        logging.debug("Буфер обмена очищен")

    def shutdown(self):
        self.is_running = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.stop_event.set()
            self.recording_thread.join(timeout=2.0)
        self.audio.close()
        self.runner.shutdown()  # завершаем сервер

# ----------------------------------------------------------------------
# Горячие клавиши + пауза
# ----------------------------------------------------------------------
class HotkeyManager:
    def __init__(self, core, config):
        self.core = core
        self.config = config
        self.hotkey_str = config["Hotkeys"]["hotkey"]
        self.listener_mouse = None
        self.listener_keyboard = None
        self.is_recording = False
        self.pressed_keys = set()
        self.target_combo = None
        self.use_mouse = False
        self.paused = False
        self.parse_hotkey()
        self.running = False

    def parse_hotkey(self):
        self.hotkey_str = self.config["Hotkeys"]["hotkey"].strip().lower()
        if self.hotkey_str == "middle_mouse":
            self.use_mouse = True
            self.target_combo = None
        else:
            self.use_mouse = False
            parts = self.hotkey_str.split("+")
            combo = set()
            for part in parts:
                part = part.strip()
                if part == "ctrl":
                    combo.add(Key.ctrl)
                elif part == "shift":
                    combo.add(Key.shift)
                elif part == "alt":
                    combo.add(Key.alt)
                else:
                    combo.add(KeyCode.from_char(part))
            self.target_combo = combo
        logging.info(f"Горячая клавиша: {self.hotkey_str}")

    def toggle_pause(self):
        self.paused = not self.paused
        logging.info(f"Пауза {'включена' if self.paused else 'выключена'}")
        return self.paused

    def start_listeners(self):
        if self.running:
            return
        if self.use_mouse:
            self.listener_mouse = mouse.Listener(on_click=self._on_mouse_click)
            self.listener_mouse.start()
        else:
            self.listener_keyboard = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            self.listener_keyboard.start()
        self.running = True
        logging.info("Слушатели запущены")

    def stop_listeners(self):
        if not self.running:
            return
        if self.listener_mouse:
            self.listener_mouse.stop()
            self.listener_mouse = None
        if self.listener_keyboard:
            self.listener_keyboard.stop()
            self.listener_keyboard = None
        self.running = False
        logging.info("Слушатели остановлены")

    def restart_listeners(self):
        self.stop_listeners()
        self.parse_hotkey()
        self.start_listeners()

    def _on_mouse_click(self, x, y, button, pressed):
        if self.paused or not self.core.is_running:
            return
        if button == Button.middle:
            if pressed and not self.is_recording:
                self.is_recording = True
                self.core.start_recording()
            elif not pressed and self.is_recording:
                self.is_recording = False
                self.core.stop_recording()

    def _on_key_press(self, key):
        if self.paused or not self.core.is_running:
            return
        self.pressed_keys.add(key)
        if self.target_combo and self.target_combo.issubset(self.pressed_keys):
            if not self.is_recording:
                self.is_recording = True
                self.core.start_recording()

    def _on_key_release(self, key):
        if self.paused or not self.core.is_running:
            return
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)
        if self.is_recording and self.target_combo and not self.target_combo.issubset(self.pressed_keys):
            self.is_recording = False
            self.core.stop_recording()

# ----------------------------------------------------------------------
# Окно настроек
# ----------------------------------------------------------------------
class SettingsWindow:
    def __init__(self, root, config, core, hotkey_manager):
        self.root = root
        self.config = config
        self.core = core
        self.hotkey_manager = hotkey_manager
        self.root.title("Whisper Dictation - Настройки")
        self.root.resizable(False, False)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_engine = ttk.Frame(notebook)
        self.tab_audio = ttk.Frame(notebook)
        self.tab_hotkeys = ttk.Frame(notebook)
        notebook.add(self.tab_engine, text="Движок")
        notebook.add(self.tab_audio, text="Звук")
        notebook.add(self.tab_hotkeys, text="Горячие клавиши")

        self._build_engine_tab()
        self._build_audio_tab()
        self._build_hotkeys_tab()

    def _build_engine_tab(self):
        row = 0
        ttk.Label(self.tab_engine, text="Путь к whisper-cli (GPU):").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.gpu_path_var = tk.StringVar(value=self.config["Paths"]["whisper_gpu"])
        ttk.Entry(self.tab_engine, textvariable=self.gpu_path_var, width=60).grid(row=row, column=1, padx=5)
        ttk.Button(self.tab_engine, text="Обзор", command=lambda: self._browse_file(self.gpu_path_var)).grid(row=row, column=2)
        row += 1

        ttk.Label(self.tab_engine, text="Путь к whisper-cli (CPU):").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.cpu_path_var = tk.StringVar(value=self.config["Paths"]["whisper_cpu"])
        ttk.Entry(self.tab_engine, textvariable=self.cpu_path_var, width=60).grid(row=row, column=1, padx=5)
        ttk.Button(self.tab_engine, text="Обзор", command=lambda: self._browse_file(self.cpu_path_var)).grid(row=row, column=2)
        row += 1

        ttk.Label(self.tab_engine, text="Путь к модели:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.model_path_var = tk.StringVar(value=self.config["Paths"]["model"])
        ttk.Entry(self.tab_engine, textvariable=self.model_path_var, width=60).grid(row=row, column=1, padx=5)
        ttk.Button(self.tab_engine, text="Обзор", command=lambda: self._browse_file(self.model_path_var)).grid(row=row, column=2)
        row += 1

        ttk.Label(self.tab_engine, text="Индекс GPU (--device):").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        gpu_idx_value = self._detect_gpu_index()
        self.gpu_idx_var = tk.StringVar(value=gpu_idx_value)
        gpu_combo = ttk.Combobox(self.tab_engine, textvariable=self.gpu_idx_var,
                                 values=["0", "1"], state="readonly", width=5)
        gpu_combo.grid(row=row, column=1, sticky="w", padx=5)
        ttk.Label(self.tab_engine, text="(0=первая GPU)").grid(row=row, column=2, sticky="w", padx=5)
        row += 1

        ttk.Label(self.tab_engine, text="Устройство:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.device_var = tk.StringVar(value=self.config["Recognition"]["device"])
        combo = ttk.Combobox(self.tab_engine, textvariable=self.device_var,
                             values=["gpu", "cpu"], state="readonly", width=10)
        combo.grid(row=row, column=1, sticky="w", padx=5)
        row += 1

        ttk.Label(self.tab_engine, text="Метод транскрипции:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.method_var = tk.StringVar(value=self.config["Recognition"].get("method", "whisper-cli"))
        method_combo = ttk.Combobox(self.tab_engine, textvariable=self.method_var,
                                    values=["whisper-cli", "whisper-server"], state="readonly", width=15)
        method_combo.grid(row=row, column=1, sticky="w", padx=5)
        row += 1

        ttk.Label(self.tab_engine, text="Путь к whisper-server.exe:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.server_path_var = tk.StringVar(value=self.config["Paths"].get("server_path", "whisper-vulkan-Server\\whisper-server.exe"))
        ttk.Entry(self.tab_engine, textvariable=self.server_path_var, width=60).grid(row=row, column=1, padx=5)
        ttk.Button(self.tab_engine, text="Обзор", command=lambda: self._browse_file(self.server_path_var)).grid(row=row, column=2)
        row += 1

        ttk.Label(self.tab_engine, text="URL сервера:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.server_url_var = tk.StringVar(value=self.config["Recognition"].get("server_url", "http://127.0.0.1:8080/inference"))
        ttk.Entry(self.tab_engine, textvariable=self.server_url_var, width=50).grid(row=row, column=1, sticky="w", padx=5)

        for var in (self.gpu_path_var, self.cpu_path_var, self.model_path_var, self.gpu_idx_var, self.server_path_var):
            var.trace_add("write", lambda *a: self._save_config())
        self.device_var.trace_add("write", lambda *a: self._save_config())
        self.method_var.trace_add("write", lambda *a: self._save_config())
        self.server_url_var.trace_add("write", lambda *a: self._save_config())

    def _build_audio_tab(self):
        ttk.Label(self.tab_audio, text="Режим работы:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.mode_var = tk.StringVar(value=self.config["Recognition"]["mode"])
        combo = ttk.Combobox(self.tab_audio, textvariable=self.mode_var,
                             values=["record_then_analyze", "streaming"], state="readonly", width=20)
        combo.grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(self.tab_audio, text="Частота дискретизации:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.sample_rate_var = tk.StringVar(value=self.config["Audio"].get("sample_rate", "16000"))
        sr_combo = ttk.Combobox(self.tab_audio, textvariable=self.sample_rate_var,
                                values=["16000", "48000"], state="readonly", width=10)
        sr_combo.grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(self.tab_audio, text="(16000 рекомендуется)").grid(row=1, column=2, sticky="w", padx=5)

        ttk.Label(self.tab_audio, text="Усиление (Gain):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.gain_var = tk.DoubleVar(value=float(self.config["Audio"]["gain"]))
        scale = ttk.Scale(self.tab_audio, from_=1.0, to=10.0, variable=self.gain_var,
                          orient="horizontal", command=lambda v: self._save_config())
        scale.grid(row=2, column=1, sticky="we", padx=5)
        ttk.Label(self.tab_audio, textvariable=self.gain_var).grid(row=2, column=2, padx=5)

        self.mic_info = tk.StringVar(value=self._get_mic_info())
        ttk.Label(self.tab_audio, text="Микрофон:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        ttk.Label(self.tab_audio, textvariable=self.mic_info).grid(row=3, column=1, sticky="w", padx=5)

        # Чекбокс VAD (Silero)
        self.vad_var = tk.BooleanVar(value=self.config.getboolean("Recognition", "use_vad", fallback=False))
        ttk.Checkbutton(self.tab_audio, text="VAD (улучшает качество при паузах)", variable=self.vad_var).grid(row=4, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(self.tab_audio, text="VAD:").grid(row=4, column=0, sticky="w", padx=5, pady=5)

        self.mode_var.trace_add("write", lambda *a: self._on_mode_change())
        self.sample_rate_var.trace_add("write", lambda *a: self._save_config())
        self.gain_var.trace_add("write", lambda *a: self._save_config())
        self.vad_var.trace_add("write", lambda *a: self._save_config())

        # Авто-включение VAD при стриминге
        if self.mode_var.get() == "streaming":
            self.vad_var.set(True)

        # Привязка изменения метода транскрипции — whisper-server = только GPU
        self.method_var.trace_add("write", lambda *a: self._on_method_change())

    def _get_mic_info(self):
        try:
            p = pyaudio.PyAudio()
            info = p.get_default_input_device_info()
            txt = f"{info['name']} ({int(info['defaultSampleRate'])} Гц, {info['maxInputChannels']} кан.)"
            p.terminate()
            return txt
        except:
            return "Не удалось получить"

    def _build_hotkeys_tab(self):
        ttk.Label(self.tab_hotkeys, text="Горячая клавиша:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.hotkey_var = tk.StringVar(value=self.config["Hotkeys"]["hotkey"])
        self.hotkey_entry = ttk.Entry(self.tab_hotkeys, textvariable=self.hotkey_var, width=20)
        self.hotkey_entry.grid(row=0, column=1, sticky="w", padx=5)
        ttk.Button(self.tab_hotkeys, text="Запомнить", command=self._capture_hotkey).grid(row=0, column=2, padx=5)
        ttk.Button(self.tab_hotkeys, text="Сбросить на среднюю кнопку мыши",
                   command=lambda: self.hotkey_var.set("middle_mouse")).grid(row=1, column=1, sticky="w", padx=5, pady=5)

        self.hotkey_var.trace_add("write", lambda *a: self._save_config())

    def _capture_hotkey(self):
        top = tk.Toplevel(self.root)
        top.title("Нажмите комбинацию...")
        top.grab_set()
        label = ttk.Label(top, text="Нажмите нужную комбинацию клавиш")
        label.pack(padx=20, pady=20)
        captured = []

        def on_press(key):
            if key == Key.esc:
                captured.clear()
                top.destroy()
                return
            try:
                k = key.char
            except AttributeError:
                k = key.name
            if k not in captured:
                captured.append(k)

        def on_release(key):
            if captured:
                combo = "+".join(captured)
                self.hotkey_var.set(combo)
                top.destroy()

        listener = KBListener(on_press=on_press, on_release=on_release)
        listener.start()
        top.protocol("WM_DELETE_WINDOW", lambda: (listener.stop(), top.destroy()))

    @staticmethod
    def _detect_gpu_index():
        """Определяет индекс GPU через whisper-cli --help (Vulkan инициализация выводит устройства в stderr).
        
        Важно: whisper-cli.exe блокируется при capture_output=True (pipe deadlock),
        поэтому вывод перенаправляется во временный файл.
        """
        try:
            whisper_exe = resolve_path("whisper-vulkan\\whisper-cli.exe")
            if not os.path.exists(whisper_exe):
                logging.warning(f"whisper-cli.exe не найден: {whisper_exe}")
                return "0"
            
            whisper_dir = os.path.dirname(whisper_exe)
            
            # Создаём временный файл для вывода
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False, 
                                             dir=str(DATA_DIR), encoding='utf-8') as tmp:
                tmp_path = tmp.name
            
            try:
                # Запускаем без capture_output, вывод в файл
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                proc = subprocess.Popen(
                    [whisper_exe, "--help"],
                    stdout=open(tmp_path, 'w', encoding='utf-8', errors='replace'),
                    stderr=subprocess.STDOUT,
                    cwd=whisper_dir,
                    startupinfo=startupinfo,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                proc.wait(timeout=10)
                
                # Читаем вывод
                with open(tmp_path, 'r', encoding='utf-8', errors='replace') as f:
                    output = f.read()
                
                logging.info(f"GPU detect: returncode={proc.returncode}, output_len={len(output)}")
                if len(output) < 500:
                    logging.debug(f"GPU detect: output = {output}")
                
                import re
                devices = re.findall(r'(\d+)\s*=\s*(.+?)\s*\|\s*uma:\s*(\d+)', output)
                logging.info(f"GPU detect: найдено устройств = {len(devices)}")
                # uma:0 = дискретная GPU (приоритет), uma:1 = встроенная
                for idx, name, uma in devices:
                    logging.info(f"GPU detect: устройство {idx} = {name}, uma={uma}")
                    if uma == "0":
                        return idx
                # Если только встроенная — вернём 0
                return "0"
            finally:
                # Удаляем временный файл
                try:
                    os.remove(tmp_path)
                except:
                    pass
        except subprocess.TimeoutExpired:
            logging.warning("whisper-cli --help завис, определяем GPU как 0")
            return "0"
        except Exception as e:
            logging.warning(f"Не удалось определить GPU: {e}")
            return "0"

    def _on_mode_change(self):
        """Авто-включение VAD при стриминге."""
        if self.mode_var.get() == "streaming":
            self.vad_var.set(True)
        self._save_config()

    def _on_method_change(self):
        """При выборе whisper-server — только GPU."""
        if self.method_var.get() == "whisper-server":
            self.device_var.set("gpu")
        self._save_config()

    def _show_temp_notification(self, text):
        """Показывает временное уведомление, которое исчезает через 0.5 сек."""
        label = tk.Label(
            self.root,
            text=text,
            bg="#28a745",
            fg="white",
            font=("Segoe UI", 9),
            padx=10,
            pady=5,
        )
        # Центрируем на окне
        self.root.update_idletasks()
        x = (self.root.winfo_width() - 200) // 2
        y = (self.root.winfo_height() - 30) // 2
        label.place(relx=0.5, rely=0.5, anchor="center")
        self.root.after(500, label.destroy)

    def _browse_file(self, var):
        filename = filedialog.askopenfilename(title="Выберите исполняемый файл")
        if filename:
            var.set(filename)

    def _save_config(self):
        try:
            self.config["Paths"]["whisper_gpu"] = self.gpu_path_var.get()
            self.config["Paths"]["whisper_cpu"] = self.cpu_path_var.get()
            self.config["Paths"]["model"] = self.model_path_var.get()
            self.config["Paths"]["gpu_device_index"] = self.gpu_idx_var.get().strip() or "1"
            self.config["Paths"]["server_path"] = self.server_path_var.get().strip()
            self.config["Recognition"]["device"] = self.device_var.get()
            self.config["Recognition"]["mode"] = self.mode_var.get()
            self.config["Recognition"]["method"] = self.method_var.get()
            self.config["Recognition"]["server_url"] = self.server_url_var.get().strip()
            self.config["Audio"]["gain"] = str(self.gain_var.get())
            self.config["Audio"]["sample_rate"] = self.sample_rate_var.get()
            self.config["Recognition"]["use_vad"] = str(self.vad_var.get()).lower()
            self.config["Hotkeys"]["hotkey"] = self.hotkey_var.get().strip()
            save_config(self.config)
            self.core.update_settings(self.config)
            self.hotkey_manager.restart_listeners()
            logging.info("Настройки сохранены и применены")
            # Уведомление с автоскрытием через 0.5 сек
            self._show_temp_notification("Настройки сохранены и применены")
        except Exception as e:
            logging.error(f"Ошибка сохранения: {e}")
            messagebox.showerror("Ошибка", f"Не удалось сохранить настройки: {e}")

# ----------------------------------------------------------------------
# Иконка в трее
# ----------------------------------------------------------------------
class TrayIcon:
    def __init__(self, core, config, hotkey_manager, root):
        self.core = core
        self.config = config
        self.hotkey_manager = hotkey_manager
        self.root = root
        self.paused = False
        self.icon = pystray.Icon("whisper_dictation")
        self.icon.icon = self._create_image(active=True)
        self.icon.title = "Whisper Dictation"
        self._setup_menu()
        self.icon.on_click = self._on_click
        self.settings_window = None

    def _create_image(self, active=True):
        width, height = 64, 64
        image = Image.new("RGB", (width, height), "white")
        dc = ImageDraw.Draw(image)
        color = "blue" if active else "gray"
        dc.ellipse([8, 8, 56, 56], fill=color)
        return image

    def _setup_menu(self):
        menu = pystray.Menu(
            pystray.MenuItem("Пауза", self._toggle_pause),
            pystray.MenuItem("Настройки", self._on_settings),
            pystray.MenuItem("Выход", self._on_exit),
        )
        self.icon.menu = menu

    def _toggle_pause(self, icon, item):
        self.paused = not self.paused
        self.hotkey_manager.paused = self.paused
        new_icon = self._create_image(active=not self.paused)
        icon.icon = new_icon
        icon.title = "Whisper Dictation (Paused)" if self.paused else "Whisper Dictation"
        logging.info(f"Пауза {'включена' if self.paused else 'выключена'}")

    def _on_click(self, icon, pos, button, action):
        if button == pystray.Button.LEFT:
            self.paused = not self.paused
            self.hotkey_manager.paused = self.paused
            new_icon = self._create_image(active=not self.paused)
            icon.icon = new_icon
            icon.title = "Whisper Dictation (Paused)" if self.paused else "Whisper Dictation"
            logging.info(f"Пауза {'включена' if self.paused else 'выключена'}")

    def _on_settings(self, icon, item):
        if self.settings_window is not None:
            try:
                self.settings_window.deiconify()
                self.settings_window.lift()
                return
            except:
                self.settings_window = None
        settings_root = tk.Toplevel(self.root)
        self.settings_window = settings_root
        SettingsWindow(settings_root, self.config, self.core, self.hotkey_manager)
        settings_root.protocol("WM_DELETE_WINDOW", lambda: self._on_settings_close(settings_root))

    def _on_settings_close(self, window):
        window.destroy()
        self.settings_window = None

    def _on_exit(self, icon, item):
        logging.info("Выход из программы")
        self.hotkey_manager.stop_listeners()
        self.core.shutdown()
        self.root.quit()
        icon.stop()
        def force_exit():
            time.sleep(0.5)
            os._exit(0)
        threading.Thread(target=force_exit, daemon=True).start()

    def run(self):
        self.icon.run()

# ----------------------------------------------------------------------
# Очистка старых файлов при старте
# ----------------------------------------------------------------------
def _cleanup_old_chunks():
    """Удаляет временные чанки от предыдущей сессии (только из chunks/)."""
    cleaned = 0
    for f in CHUNK_DIR.glob("chunk_*.wav"):
        try:
            f.unlink()
            cleaned += 1
        except Exception as e:
            logging.warning(f"Не удалось удалить {f}: {e}")
    for f in CHUNK_DIR.glob("chunk_*.txt"):
        try:
            f.unlink()
            cleaned += 1
        except Exception as e:
            logging.warning(f"Не удалось удалить {f}: {e}")
    if cleaned:
        logging.info(f"Очищены старые чанки: {cleaned} файлов")

def clean_old_files():
    """Полная очистка временных файлов при запуске программы.
    
    Удаляет:
    - Все чанки из chunks/ (chunk_*.wav, chunk_*.txt)
    - Старые WAV файлы из recordings/ (кроме temp_record.wav)
    """
    cleaned = 0
    
    # Очистка чанков
    for f in CHUNK_DIR.glob("chunk_*.wav"):
        try:
            f.unlink()
            cleaned += 1
        except Exception as e:
            logging.warning(f"Не удалось удалить {f}: {e}")
    
    for f in CHUNK_DIR.glob("chunk_*.txt"):
        try:
            f.unlink()
            cleaned += 1
        except Exception as e:
            logging.warning(f"Не удалось удалить {f}: {e}")
    
    # Очистка старых WAV записей (кроме temp_record.wav)
    for f in RECORDINGS_DIR.glob("*.wav"):
        if f.name != "temp_record.wav":
            try:
                f.unlink()
                cleaned += 1
            except Exception as e:
                logging.warning(f"Не удалось удалить {f}: {e}")
    
    # Очистка старых TXT файлов в recordings/
    for f in RECORDINGS_DIR.glob("*.txt"):
        try:
            f.unlink()
            cleaned += 1
        except Exception as e:
            logging.warning(f"Не удалось удалить {f}: {e}")
    
    if cleaned:
        logging.info(f"Очистка при старте: удалено {cleaned} файлов из {DATA_DIR}")
    else:
        logging.info("Очистка при старте: временных файлов не найдено")

# ----------------------------------------------------------------------
# Защита от запуска второй копии
# ----------------------------------------------------------------------
def _ensure_single_instance():
    """Проверяет, не запущена ли уже копия программы (через named mutex)."""
    try:
        import win32event
        import win32api
        MUTEX_NAME = "Global\\WhisperDictation_Mutex"
        mutex = win32event.CreateMutex(None, False, MUTEX_NAME)
        if win32api.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            return False
        return True
    except ImportError:
        # Если pywin32 не установлен — пропускаем проверку
        logging.warning("pywin32 не установлен, проверка единственного экземпляра отключена")
        return True
    except Exception as e:
        logging.warning(f"Ошибка при проверке экземпляра: {e}")
        return True

# ----------------------------------------------------------------------
# Главная функция
# ----------------------------------------------------------------------
def main():
    # Проверка: не запущена ли уже копия
    if not _ensure_single_instance():
        try:
            import tkinter as _tk
            from tkinter import messagebox as _mb
            _root = _tk.Tk()
            _root.withdraw()
            _mb.showinfo("Внимание", "Whisper Dictation уже запущена")
            _root.destroy()
        except:
            pass
        sys.exit(0)
    
    logging.info("=== Программа запущена ===")
    # Полная очистка временных файлов при запуске
    clean_old_files()
    config = load_config()
    core = DictationCore(config)
    hotkey_manager = HotkeyManager(core, config)
    hotkey_manager.start_listeners()

    root = tk.Tk()
    root.withdraw()

    tray = TrayIcon(core, config, hotkey_manager, root)
    tray_thread = threading.Thread(target=tray.run, daemon=True)
    tray_thread.start()

    root.mainloop()

    hotkey_manager.stop_listeners()
    core.shutdown()
    logging.info("Программа завершена")
    os._exit(0)

if __name__ == "__main__":
    main()