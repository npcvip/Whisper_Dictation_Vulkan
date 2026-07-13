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
import numpy as np
import sherpa_onnx
import pyautogui
import pyperclip
import pystray

# ----------------------------------------------------------------------
# Добавляем путь к ffmpeg в PATH ДО импорта pydub
# ----------------------------------------------------------------------
_FFMPEG_DIR = str(Path(__file__).parent / "ffmpeg")
if os.path.isdir(_FFMPEG_DIR):
    os.environ["PATH"] += f";{_FFMPEG_DIR}"

import pydub
from pydub import AudioSegment
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
# Очистка логов при запуске
# ----------------------------------------------------------------------
if LOG_PATH.exists():
    try:
        LOG_PATH.unlink()
    except Exception:
        pass

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
# Инициализация pydub (путь к ffmpeg и ffprobe)
# ----------------------------------------------------------------------
FFMPEG_PATH = RESOURCE_DIR / "ffmpeg" / "ffmpeg.exe"
FFPROBE_PATH = RESOURCE_DIR / "ffmpeg" / "ffprobe.exe"
if FFMPEG_PATH.exists():
    pydub.AudioSegment.converter = str(FFMPEG_PATH)
    pydub.AudioSegment.ffmpeg = str(FFMPEG_PATH)
    logging.info(f"pydub: ffmpeg найден — {FFMPEG_PATH}")
else:
    logging.warning(f"pydub: ffmpeg не найден — {FFMPEG_PATH}. Конвертация аудио не будет работать.")

if FFPROBE_PATH.exists():
    pydub.AudioSegment.prober = str(FFPROBE_PATH)
    logging.info(f"pydub: ffprobe найден — {FFPROBE_PATH}")
else:
    logging.warning(f"pydub: ffprobe не найден — {FFPROBE_PATH}. Чтение метаданных аудио не будет работать.")

# ----------------------------------------------------------------------
# Конфигурация (относительные пути)
# ----------------------------------------------------------------------
DEFAULT_CONFIG = {
    "Paths": {
        "whisper_gpu": "whisper-vulkan\\whisper-cli.exe",
        "whisper_cpu": "whisper-CPU\\whisper-cli.exe",
        "model": "models\\ggml-large-v3-turbo-q8_0.bin",
        "gpu_device_index": "1",
        "server_gpu_path": "whisper-vulkan-Server\\whisper-server.exe",
        "server_cpu_path": "whisper-CPU\\whisper-server.exe",
    },
    "Recognition": {
        "device": "gpu",
        "mode": "record_then_analyze",
        "language": "ru",
        "method": "whisper-server",   # "whisper-cli" или "whisper-server"
        "server_url": "http://127.0.0.1:18877/inference",
        "use_vad": "false",
        "use_vad_streaming": "true",
        "vad_threshold": "0.5",
        "vad_min_silence_duration": "0.7",
        "vad_min_speech_duration": "0.25",
        "min_chunk_duration": "3.0",
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
        self.mic_available = True
        self.device_index = None
        self.rate = 16000
        self.channels = 1
        self.format = pyaudio.paInt16
        self.chunk = 1024
        self.gain = 2.0

        # Попытка получить микрофон
        try:
            dev_info = self.p.get_default_input_device_info()
            self.device_index = dev_info["index"]
            target_rate = int(config["Audio"].get("sample_rate", "16000"))
            supported_rates = self._get_supported_sample_rates(dev_info)
            if target_rate in supported_rates:
                self.rate = target_rate
            else:
                self.rate = min(supported_rates, key=lambda x: abs(x - target_rate))
                logging.warning(f"Устройство не поддерживает {target_rate} Гц, используется {self.rate} Гц")
            logging.info(f"Микрофон найден: {dev_info['name']} (индекс {self.device_index})")
        except Exception as e:
            self.mic_available = False
            logging.warning(f"Микрофон не найден: {e}")

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
# Ten VAD через sherpa-onnx
# ----------------------------------------------------------------------
class TenVAD:
    """Ten VAD через sherpa-onnx — потоковая детекция речи."""
    
    def __init__(self, model_path, threshold=0.5, sample_rate=16000,
                 min_silence_duration=0.7, min_speech_duration=0.25):
        vad_config = sherpa_onnx.TenVadModelConfig(
            model=model_path,
            threshold=threshold,
            min_silence_duration=min_silence_duration,
            min_speech_duration=min_speech_duration,
        )
        config = sherpa_onnx.VadModelConfig()
        config.ten_vad = vad_config
        config.sample_rate = sample_rate
        config.num_threads = 1
        self.vad = sherpa_onnx.VoiceActivityDetector(config)
        self.sample_rate = sample_rate
    
    def reset(self):
        """Сброс состояния VAD."""
        self.vad.reset()
    
    def accept_waveform(self, samples):
        """Подача сэмплов в VAD (list of float или numpy array)."""
        self.vad.accept_waveform(samples.tolist() if isinstance(samples, np.ndarray) else samples)
    
    def is_speech_detected(self):
        """Возвращает True если речь обнаружена."""
        return self.vad.is_speech_detected()


# ----------------------------------------------------------------------
# Whisper Runner (поддерживает whisper-cli и whisper-server с автозапуском)
# ----------------------------------------------------------------------
class WhisperRunner:
    def __init__(self, config):
        self.config = config
        self.method = config["Recognition"].get("method", "whisper-cli")
        self.server_url = config["Recognition"].get("server_url", "http://127.0.0.1:18877/inference")
        self.use_vad = config.getboolean("Recognition", "use_vad", fallback=False)
        self.server_process = None
        self._prev_method = self.method  # для отслеживания смены метода
        self._prev_device = self.config["Recognition"].get("device", "gpu")  # для отслеживания смены устройства
        self.update_paths()
        self._auto_start_server_if_needed()

    def _auto_start_server_if_needed(self):
        """Если выбран метод whisper-server, запускает сервер и ждёт его готовности."""
        if self.method != "whisper-server" or not REQUESTS_AVAILABLE:
            return

        # Проверяем, не запущен ли уже сервер
        try:
            r = requests.get("http://127.0.0.1:18877/health", timeout=1)
            if r.status_code == 200:
                # Если это наш процесс — всё ок
                if self.server_process and self.server_process.poll() is None:
                    logging.info("Сервер уже запущен (наш процесс), подключаемся")
                    return
                # Если это чужой процесс (от предыдущей сессии) — убиваем и запускаем свой
                logging.warning("Обнаружен чужой сервер на порту 18877, завершаем и запускаем новый")
                self._kill_server_by_port(18877)
                time.sleep(1)
        except:
            pass

        # Выбираем сервер в зависимости от устройства
        device = self.config["Recognition"].get("device", "gpu")
        if device == "cpu":
            server_exe = resolve_path(self.config["Paths"].get("server_cpu_path", "whisper-CPU\\whisper-server.exe"))
        else:
            server_exe = resolve_path(self.config["Paths"].get("server_gpu_path", "whisper-vulkan-Server\\whisper-server.exe"))
        if not os.path.exists(server_exe):
            logging.error(f"whisper-server.exe не найден: {server_exe}")
            # Переключаемся на whisper-cli как fallback
            self.method = "whisper-cli"
            return

        model_path = resolve_path(self.config["Paths"]["model"])
        device = self.config["Paths"].get("gpu_device_index", "1")
        language = self.config["Recognition"]["language"]
        vad_model_path = str(RESOURCE_DIR / "models" / "ggml-silero-v6.2.0.bin")

        cmd = [
            server_exe,
            "-m", model_path,
            "--device", device,
            "--host", "127.0.0.1",
            "--port", "18877",
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
                r = requests.get("http://127.0.0.1:18877/health", timeout=1)
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

    def _kill_server_by_port(self, port=18877):
        """Убивает процесс, занимающий указанный порт (Windows)."""
        try:
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True, timeout=5)
                    logging.info(f"Убит процесс PID={pid} на порту {port}")
                    return True
            logging.info(f"Нет процессов на порту {port}")
            return False
        except Exception as e:
            logging.warning(f"Не удалось убить сервер на порту {port}: {e}")
            return False

    def _stop_server(self):
        """Завершает процесс сервера."""
        if self.server_process and self.server_process.poll() is None:
            # Наш процесс — завершаем его
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
            logging.info("Сервер остановлен (собственный процесс)")
        elif REQUESTS_AVAILABLE:
            # Сервер не наш (от предыдущей сессии) — проверяем и убиваем по порту
            try:
                r = requests.get("http://127.0.0.1:18877/health", timeout=1)
                if r.status_code == 200:
                    logging.warning("Сервер не наш процесс, убиваем по порту 18877")
                    self._kill_server_by_port(18877)
                    time.sleep(0.5)
            except Exception as e:
                logging.warning(f"Сервер недоступен при остановке: {e}")
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
        old_method = self.method
        self.method = self.config["Recognition"].get("method", "whisper-cli")
        self.server_url = self.config["Recognition"].get("server_url", "http://127.0.0.1:18877/inference")
        self.use_vad = self.config.getboolean("Recognition", "use_vad", fallback=False)
        logging.info(f"Whisper обновлён: method={self.method}, model={self.model_path}, device_flag={self.device_flag}, use_vad={self.use_vad}")

        # Если переключились с whisper-server на whisper-cli — останавливаем сервер
        if old_method == "whisper-server" and self.method != "whisper-server":
            logging.info("Переключение с whisper-server на whisper-cli, остановка сервера")
            self._stop_server()

        # Если изменился метод, перезапускаем сервер при необходимости
        if self.method == "whisper-server" and old_method != "whisper-server":
            self._auto_start_server_if_needed()

        # Если изменилось устройство и метод = whisper-server — перезапуск сервера
        new_device = self.config["Recognition"].get("device", "gpu")
        if self.method == "whisper-server" and new_device != self._prev_device:
            logging.info(f"Смена устройства: {self._prev_device} → {new_device}, перезапуск сервера")
            self._stop_server()
            time.sleep(1)
            self._auto_start_server_if_needed()

        self._prev_method = self.method
        self._prev_device = new_device

    def transcribe(self, wav_path):
        if self.method == "whisper-server" and REQUESTS_AVAILABLE:
            # Быстрая проверка: сервер отвечает на /inference?
            try:
                r = requests.get("http://127.0.0.1:18877/health", timeout=3)
                if r.status_code != 200:
                    raise ConnectionError("Сервер не отвечает")
            except Exception as e:
                logging.warning(f"Сервер недоступен перед транскрипцией: {e}, переключение на cli")
                return self._transcribe_with_cli(wav_path)

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

        # whisper-cli с -otxt создаёт файл <output>.txt
        # -of принимает имя БЕЗ расширения (см. --help: "output file path (without file extension)")
        wav_basename = os.path.basename(str(wav_path)).rsplit(".", 1)[0]
        output_base = str(RECORDINGS_DIR / wav_basename)  # путь без расширения (для -of)
        output_txt = output_base + ".txt"                 # полный путь к .txt файлу

        # VAD-логирование
        wav_size = os.path.getsize(str(wav_path))
        # Примерная длительность: 16000 Гц * 2 байта * 1 канал = 32000 байт/сек
        wav_duration = wav_size / 32000.0
        if self.use_vad:
            vad_model_path = str(RESOURCE_DIR / "models" / "ggml-silero-v6.2.0.bin")
            logging.info(f"VAD [whisper-cli]: включён, модель={vad_model_path}, аудио={wav_size} байт (~{wav_duration:.1f} сек)")
        else:
            logging.info(f"VAD [whisper-cli]: выключен, аудио={wav_size} байт (~{wav_duration:.1f} сек)")

        cmd = [
            self.whisper_exe,
            "-m", self.model_path,
            "-f", str(wav_path),
            "-l", self.language,
            "--no-timestamps",
            "-otxt",
            "-of", output_base,  # путь БЕЗ расширения — whisper-cli добавит .txt
        ] + self.device_flag

        # VAD-флаги для whisper-cli
        if self.use_vad:
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
                cwd=str(RECORDINGS_DIR),  # .txt создастся в RECORDINGS_DIR
            )
            # Логирование stderr для диагностики
            if result.stderr:
                logging.warning(f"whisper-cli stderr: {result.stderr.strip()[:500]}")
            if result.stdout:
                logging.debug(f"whisper-cli stdout: {result.stdout.strip()[:500]}")
            
            # Читаем результат из .txt файла (основной способ)
            if os.path.exists(output_txt):
                with open(output_txt, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                # TXT НЕ удаляется — оставляем для отладки (удалится при следующей записи)
                logging.debug(f"Файл .txt сохранён для отладки: {output_txt}")
                logging.info(f"Текст получен из файла: {len(text)} символов")
                return text
            else:
                logging.warning(f"Файл {output_txt} не создан, fallback на stdout")
                # Fallback: текст из stdout (работает только в режиме разработки)
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
        
        # Предзагрузка модели Ten VAD (sherpa-onnx)
        self._vad = None
        try:
            model_path = str(RESOURCE_DIR / "models" / "ten-vad.onnx")
            if os.path.exists(model_path):
                self._vad = TenVAD(model_path, sample_rate=16000)
                logging.info(f"Ten VAD (sherpa-onnx) загружен: {model_path}")
            else:
                logging.warning(f"Ten VAD модель не найдена: {model_path}")
        except Exception as e:
            logging.warning(f"Ten VAD не загружен: {e}")

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
        # Проверка микрофона перед записью
        if not self.audio.mic_available:
            logging.warning("Запись не начата: микрофон не найден")
            return
        # Очистка старых чанков перед началом записи
        self._cleanup_chunks()
        # Логирование реальной частоты микрофона
        logging.info(f"Частота записи: {self.audio.rate} Гц (устройство: {self.audio.device_index})")
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
        logging.info("Запись остановлена")

    def _cleanup_chunks(self):
        """Удаляет все временные файлы из папок chunks/ и recordings/ (.wav + .txt)."""
        cleaned = 0
        # Чанки из chunks/
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
        # TXT из recordings/ (остатки от транскрипции)
        for f in RECORDINGS_DIR.glob("*.txt"):
            try:
                f.unlink()
                cleaned += 1
            except Exception as e:
                logging.warning(f"Не удалось удалить {f}: {e}")
        if cleaned:
            logging.info(f"Временные файлы очищены: {cleaned} файлов")

    def _process_remaining_buffer(self):
        if not self.full_buffer:
            return
        logging.info(f"Обработка остатка буфера: {len(self.full_buffer)} фреймов")
        self._process_chunk(self.full_buffer)
        self.full_buffer = []

    def _process_chunk(self, frames):
        if not frames:
            return
        # Фильтр: пропускаем слишком короткие чанки (< 300ms)
        total_bytes = sum(len(f) for f in frames)
        duration = total_bytes / 32000.0
        if duration < 0.3:
            logging.debug(f"Чанк слишком короткий ({duration:.1f} сек, {total_bytes} байт), пропускаем")
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
            self._paste_text(text + " ", streaming=True)
        # Чанк НЕ удаляется — оставляем для отладки (удалится при следующей записи)
        logging.info(f"Чанк сохранён для отладки: {chunk_path}")

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
            self._paste_text(text + " ", streaming=False)
            # WAV НЕ удаляется — оставляем для отладки (удалится при следующей записи)
            logging.info(f"WAV сохранён для отладки: {wav_path}")
        else:
            logging.warning("Файл пуст или не создан")
            self._paste_text("[Нет записанных данных]")

    def _streaming_record_no_vad(self):
        """Стриминговая запись БЕЗ VAD — весь буфер → один чанк при остановке."""
        audio_buffer = bytearray()
        stream = self.audio.p.open(
            format=self.audio.format,
            channels=1,
            rate=self.audio.rate,
            input=True,
            input_device_index=self.audio.device_index,
            frames_per_buffer=self.audio.chunk,
        )
        
        logging.info(f"VAD [streaming NO VAD]: запись без VAD, частота: {self.audio.rate} Гц")
        
        try:
            while not self.stop_event.is_set() and self.is_running:
                data = stream.read(self.audio.chunk, exception_on_overflow=False)
                # Применяем gain (как в record_to_file)
                if self.audio.gain != 1.0:
                    samples = struct.unpack(f"<{len(data)//2}h", data)
                    amplified = []
                    for s in samples:
                        ns = int(s * self.audio.gain)
                        if ns > 32767:
                            ns = 32767
                        elif ns < -32768:
                            ns = -32768
                        amplified.append(ns)
                    data = struct.pack(f"<{len(amplified)}h", *amplified)
                audio_buffer += data
        except Exception as e:
            logging.error(f"Ошибка в потоковой записи: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            # Отправка всего буфера
            if len(audio_buffer) > 0:
                buf_duration = len(audio_buffer) / 32000.0
                logging.info(f"VAD [streaming NO VAD]: остаток буфера, чанк {len(audio_buffer)} байт (~{buf_duration:.1f} сек)")
                self._process_chunk([bytes(audio_buffer)])

    def _streaming_record(self):
        """Стриминговая запись с Ten VAD (sherpa-onnx) — Вариант 2: всегда накапливать.
        
        Логика:
        - ВСЕГДА добавляем аудио в accumulate_buffer (речь, пауза — всё)
        - VAD используется ТОЛЬКО для определения моментов отправки (паузы)
        - Отправка происходит АСИНХРОННО (в отдельном потоке) — запись не блокируется
        
       accumulate_buffer содержит ВСЁ аудио (включая паузы).
        Внутренний VAD Whisper (Silero) уберёт паузы при распознавании.
        """
        use_vad_streaming = self.config.getboolean("Recognition", "use_vad_streaming", fallback=True)
        
        if not use_vad_streaming:
            logging.info("VAD [streaming]: выключен (use_vad_streaming=false), запись без VAD")
            return self._streaming_record_no_vad()
        
        if self._vad is None:
            logging.error("Ten VAD модель не загружена, стриминг недоступен")
            return
        
        vad = self._vad
        
        min_chunk_duration = self.config.getfloat("Recognition", "min_chunk_duration", fallback=3.0)
        
        logging.info(f"VAD [streaming Ten v2]: всегда накапливаем, VAD только для пауз, min_chunk={min_chunk_duration}s")
        
        last_send_time = time.time()
        FORCE_SEND_INTERVAL = 10.0
        
        # Буфер накопления ВСЕГО аудио (байты)
        accumulate_buffer = bytearray()
        chunk_count = 0
        
        # Lock для защиты accumulate_buffer при асинхронной отправке
        buffer_lock = threading.Lock()
        
        # Состояние VAD (только для определения пауз)
        silence_counter = 0
        
        stream = self.audio.p.open(
            format=self.audio.format,
            channels=1,
            rate=self.audio.rate,
            input=True,
            input_device_index=self.audio.device_index,
            frames_per_buffer=self.audio.chunk,
        )
        
        logging.info(f"VAD [streaming Ten v2]: частота записи: {self.audio.rate} Гц")
        
        debug_log_counter = 0
        try:
            while not self.stop_event.is_set() and self.is_running:
                data = stream.read(self.audio.chunk, exception_on_overflow=False)
                
                # ВСЕГДА добавляем аудио в буфер
                with buffer_lock:
                    accumulate_buffer += data
                
                # Конвертируем в float32 для VAD
                new_int16 = np.frombuffer(data, dtype=np.int16)
                
                # Применяем gain ДО VAD-обработки
                if self.audio.gain != 1.0:
                    new_float = (new_int16.astype(np.float32) * self.audio.gain).clip(-32768, 32767).astype(np.float32) / 32768.0
                else:
                    new_float = new_int16.astype(np.float32) / 32768.0
                
                # Подаём сэмплы в Ten VAD (только для определения пауз)
                vad.accept_waveform(new_float)
                
                speech_detected = vad.is_speech_detected()
                
                debug_log_counter += 1
                if debug_log_counter % 100 == 0:
                    with buffer_lock:
                        buf_len = len(accumulate_buffer)
                    logging.debug(f"VAD [streaming Ten v2]: accumulate={buf_len} байт, speech={speech_detected}, silence={silence_counter}")
                
                # Логика VAD: только определяем паузы
                if speech_detected:
                    # Речь — сбрасываем счётчик тишины
                    silence_counter = 0
                else:
                    # Тишина — считаем кадры
                    silence_counter += 1
                    
                    if silence_counter >= 3:  # пауза подтверждена
                        with buffer_lock:
                            acc_duration = len(accumulate_buffer) / (self.audio.rate * 2)
                        
                        if acc_duration >= min_chunk_duration:
                            # Копируем данные и очищаем буфер (под lock)
                            with buffer_lock:
                                chunk_data = bytes(accumulate_buffer)
                                accumulate_buffer = bytearray()
                            
                            logging.info(f"VAD [streaming Ten v2]: отправка при паузе, {len(chunk_data)} байт (~{acc_duration:.1f} сек)")
                            chunk_count += 1
                            
                            # АСИНХРОННАЯ отправка — запись не блокируется!
                            threading.Thread(
                                target=self._process_chunk,
                                args=([chunk_data],),
                                daemon=True
                            ).start()
                            
                            last_send_time = time.time()
                        else:
                            logging.debug(f"VAD [streaming Ten v2]: чанк короткий ({acc_duration:.2f} сек < {min_chunk_duration} сек), накапливаем")
                        
                        # Сбрасываем счётчик тишины
                        silence_counter = 0
                
                # Принудительная отправка через 10 сек
                with buffer_lock:
                    acc_len = len(accumulate_buffer)
                if acc_len > 0 and (time.time() - last_send_time) >= FORCE_SEND_INTERVAL:
                    with buffer_lock:
                        acc_duration = len(accumulate_buffer) / (self.audio.rate * 2)
                        chunk_data = bytes(accumulate_buffer)
                        accumulate_buffer = bytearray()
                    
                    logging.info(f"VAD [streaming Ten v2]: принудительная отправка, {len(chunk_data)} байт (~{acc_duration:.1f} сек)")
                    chunk_count += 1
                    
                    # АСИНХРОННАЯ отправка
                    threading.Thread(
                        target=self._process_chunk,
                        args=([chunk_data],),
                        daemon=True
                    ).start()
                    
                    last_send_time = time.time()
                    silence_counter = 0
                
                time.sleep(0.001)
                    
        except Exception as e:
            logging.error(f"Ошибка в потоковой записи: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            # Отправка остатка (БЕЗ проверки min_chunk_duration — отправляем всё!)
            with buffer_lock:
                if len(accumulate_buffer) > 0:
                    chunk_data = bytes(accumulate_buffer)
                    accumulate_buffer = bytearray()
            
            if chunk_data:
                acc_duration = len(chunk_data) / (self.audio.rate * 2)
                logging.info(f"VAD [streaming Ten v2]: остаток при остановке, {len(chunk_data)} байт (~{acc_duration:.1f} сек) — отправляем всё")
                self._process_chunk([chunk_data])

    def _paste_text(self, text, streaming=False):
        if not text or text.startswith("[Ошибка]"):
            logging.warning(f"Не вставляем текст: {text}")
            return
        # Логирование вставки (задача 16: добавлено время)
        mode_label = "streaming" if streaming else "record"
        preview = text[:50].replace('\n', ' ')
        ts = time.strftime("%H:%M:%S")
        logging.info(f"[{ts}] Вставка текста [{mode_label}]: '{preview}...' (длина {len(text)})")
        
        if streaming:
            # Стриминг — асинхронные потоки, нужна блокировка буфера обмена
            with clipboard_lock:
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
        else:
            # Обычный режим — без блокировки (как раньше)
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
        self.root.resizable(True, True)
        self.root.geometry("850x550")

        # Создаём hotkey_var ДО вызова _build_*_tab() чтобы _save_config() мог его использовать
        self.hotkey_var = tk.StringVar(value=self.config["Hotkeys"]["hotkey"])

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

    def _set_tooltip(self, widget, text):
        """Добавляет всплывающую подсказку к виджету."""
        def _on_enter(e):
            self._tooltip = tk.Toplevel(widget)
            self._tooltip.wm_overrideredirect(True)
            self._tooltip.wm_geometry(f"+{e.x_root+10}+{e.y_root+10}")
            label = tk.Label(self._tooltip, text=text, justify=tk.LEFT,
                            background="#ffffe0", relief="solid", borderwidth=1,
                            font=("Segoe UI", 8), wraplength=300)
            label.pack()
        def _on_leave(e):
            if hasattr(self, '_tooltip'):
                self._tooltip.destroy()
                del self._tooltip
        widget.bind("<Enter>", _on_enter)
        widget.bind("<Leave>", _on_leave)

    def _build_engine_tab(self):
        row = 0
        lbl = ttk.Label(self.tab_engine, text="Путь к whisper-cli (GPU):")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self._set_tooltip(lbl, "Путь к исполняемому файлу whisper-cli для GPU (Vulkan)")
        self.gpu_path_var = tk.StringVar(value=self.config["Paths"]["whisper_gpu"])
        ttk.Entry(self.tab_engine, textvariable=self.gpu_path_var, width=60).grid(row=row, column=1, padx=5)
        ttk.Button(self.tab_engine, text="Обзор", command=lambda: self._browse_file(self.gpu_path_var)).grid(row=row, column=2)
        row += 1

        lbl = ttk.Label(self.tab_engine, text="Путь к whisper-cli (CPU):")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self._set_tooltip(lbl, "Путь к исполняемому файлу whisper-cli для CPU")
        self.cpu_path_var = tk.StringVar(value=self.config["Paths"]["whisper_cpu"])
        ttk.Entry(self.tab_engine, textvariable=self.cpu_path_var, width=60).grid(row=row, column=1, padx=5)
        ttk.Button(self.tab_engine, text="Обзор", command=lambda: self._browse_file(self.cpu_path_var)).grid(row=row, column=2)
        row += 1

        lbl = ttk.Label(self.tab_engine, text="Путь к модели:")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self._set_tooltip(lbl, "Путь к файлу модели Whisper (.bin)")
        self.model_path_var = tk.StringVar(value=self.config["Paths"]["model"])
        ttk.Entry(self.tab_engine, textvariable=self.model_path_var, width=60).grid(row=row, column=1, padx=5)
        ttk.Button(self.tab_engine, text="Обзор", command=lambda: self._browse_file(self.model_path_var)).grid(row=row, column=2)
        row += 1

        lbl = ttk.Label(self.tab_engine, text="Индекс GPU (--device):")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self._set_tooltip(lbl, "0=первая GPU, 1=вторая GPU. Определяется автоматически")
        gpu_idx_value = self._detect_gpu_index()
        self.gpu_idx_var = tk.StringVar(value=gpu_idx_value)
        gpu_combo = ttk.Combobox(self.tab_engine, textvariable=self.gpu_idx_var,
                                 values=["0", "1"], state="readonly", width=5)
        gpu_combo.grid(row=row, column=1, sticky="w", padx=5)
        ttk.Label(self.tab_engine, text="(0=первая GPU)").grid(row=row, column=2, sticky="w", padx=5)
        row += 1

        lbl = ttk.Label(self.tab_engine, text="Устройство:")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self._set_tooltip(lbl, "GPU=быстрее, CPU=медленнее, но работает без GPU")
        self.device_var = tk.StringVar(value=self.config["Recognition"]["device"])
        combo = ttk.Combobox(self.tab_engine, textvariable=self.device_var,
                             values=["gpu", "cpu"], state="readonly", width=10)
        combo.grid(row=row, column=1, sticky="w", padx=5)
        row += 1

        lbl = ttk.Label(self.tab_engine, text="Метод транскрипции:")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self._set_tooltip(lbl, "whisper-cli: локальный процесс, загружает модель для каждого файла. whisper-server: локальный HTTP-сервер, модель всегда в GPU, генерация мгновенная. Сервер запускается автоматически")
        self.method_var = tk.StringVar(value=self.config["Recognition"].get("method", "whisper-cli"))
        method_combo = ttk.Combobox(self.tab_engine, textvariable=self.method_var,
                                    values=["whisper-cli", "whisper-server"], state="readonly", width=15)
        method_combo.grid(row=row, column=1, sticky="w", padx=5)
        row += 1

        lbl = ttk.Label(self.tab_engine, text="Путь к whisper-server (GPU):")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self._set_tooltip(lbl, "Путь к исполняемому файлу whisper-server для GPU (Vulkan). Используется при выборе устройства GPU + метода транскрипции whisper-server. Модель загружается в видеопамять GPU")
        self.server_gpu_path_var = tk.StringVar(value=self.config["Paths"].get("server_gpu_path", "whisper-vulkan-Server\\whisper-server.exe"))
        ttk.Entry(self.tab_engine, textvariable=self.server_gpu_path_var, width=60).grid(row=row, column=1, padx=5)
        ttk.Button(self.tab_engine, text="Обзор", command=lambda: self._browse_file(self.server_gpu_path_var)).grid(row=row, column=2)
        row += 1

        lbl = ttk.Label(self.tab_engine, text="Путь к whisper-server (CPU):")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self._set_tooltip(lbl, "Путь к исполняемому файлу whisper-server для CPU. Используется при выборе устройства CPU + метода транскрипции whisper-server. Модель загружается в оперативную память")
        self.server_cpu_path_var = tk.StringVar(value=self.config["Paths"].get("server_cpu_path", "whisper-CPU\\whisper-server.exe"))
        ttk.Entry(self.tab_engine, textvariable=self.server_cpu_path_var, width=60).grid(row=row, column=1, padx=5)
        ttk.Button(self.tab_engine, text="Обзор", command=lambda: self._browse_file(self.server_cpu_path_var)).grid(row=row, column=2)
        row += 1

        lbl = ttk.Label(self.tab_engine, text="URL сервера:")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self._set_tooltip(lbl, "Адрес локального HTTP-сервера whisper-server для транскрипции. По умолчанию http://127.0.0.1:18877/inference")
        self.server_url_var = tk.StringVar(value=self.config["Recognition"].get("server_url", "http://127.0.0.1:18877/inference"))
        ttk.Entry(self.tab_engine, textvariable=self.server_url_var, width=50).grid(row=row, column=1, sticky="w", padx=5)

        for var in (self.gpu_path_var, self.cpu_path_var, self.model_path_var, self.gpu_idx_var, self.server_gpu_path_var, self.server_cpu_path_var):
            var.trace_add("write", lambda *a: self._save_config())
        self.device_var.trace_add("write", lambda *a: self._save_config())
        self.method_var.trace_add("write", lambda *a: self._save_config())
        self.server_url_var.trace_add("write", lambda *a: self._save_config())

    def _build_audio_tab(self):
        lbl = ttk.Label(self.tab_audio, text="Режим работы:")
        lbl.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self._set_tooltip(lbl, "record_then_analyze: запись → стоп → распознавание. streaming: распознавание в реальном времени по паузам")
        self.mode_var = tk.StringVar(value=self.config["Recognition"]["mode"])
        combo = ttk.Combobox(self.tab_audio, textvariable=self.mode_var,
                             values=["record_then_analyze", "streaming"], state="readonly", width=20)
        combo.grid(row=0, column=1, sticky="w", padx=5)

        lbl = ttk.Label(self.tab_audio, text="Частота дискретизации:")
        lbl.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self._set_tooltip(lbl, "16000 Гц рекомендуется для Whisper. 48000 Гц — высокое качество, но медленнее")
        self.sample_rate_var = tk.StringVar(value=self.config["Audio"].get("sample_rate", "16000"))
        sr_combo = ttk.Combobox(self.tab_audio, textvariable=self.sample_rate_var,
                                values=["16000", "48000"], state="readonly", width=10)
        sr_combo.grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(self.tab_audio, text="(16000 рекомендуется)").grid(row=1, column=2, sticky="w", padx=5)

        lbl = ttk.Label(self.tab_audio, text="Усиление (Gain):")
        lbl.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self._set_tooltip(lbl, "Усиление микрофона. 1.0=без усиления, 10.0=максимум. Помогает при тихом микрофоне")
        self.gain_var = tk.DoubleVar(value=float(self.config["Audio"]["gain"]))
        scale = ttk.Scale(self.tab_audio, from_=1.0, to=10.0, variable=self.gain_var,
                          orient="horizontal", command=lambda v: self._save_config())
        scale.grid(row=2, column=1, sticky="we", padx=5)
        ttk.Label(self.tab_audio, textvariable=self.gain_var).grid(row=2, column=2, padx=5)

        self.mic_info = tk.StringVar(value=self._get_mic_info())
        lbl = ttk.Label(self.tab_audio, text="Микрофон:")
        lbl.grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self._set_tooltip(lbl, "Информация о микрофоне по умолчанию")
        ttk.Label(self.tab_audio, textvariable=self.mic_info).grid(row=3, column=1, sticky="w", padx=5)

        # === Внутренний VAD (Silero для whisper-cli/server) ===
        lbl = ttk.Label(self.tab_audio, text="VAD:")
        lbl.grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self._set_tooltip(lbl, "Включает Silero VAD в whisper-cli/server. Убирает тишину из аудио перед распознаванием. Включено по умолчанию")
        self.vad_var = tk.BooleanVar(value=self.config.getboolean("Recognition", "use_vad", fallback=True))
        chk = ttk.Checkbutton(self.tab_audio, text="VAD (внутренний, Silero для whisper-cli/server)", variable=self.vad_var)
        chk.grid(row=4, column=1, sticky="w", padx=5, pady=5)

        # Vad Threshold для внутреннего VAD
        lbl = ttk.Label(self.tab_audio, text="Vad Threshold:")
        lbl.grid(row=5, column=0, sticky="w", padx=5, pady=2)
        self._set_tooltip(lbl, "Vad Threshold — порог уверенности для Внутреннего VAD (Silero в whisper-cli/server).\n\n"
                               "Как это работает: VAD анализирует аудио и выставляет оценку от 0.0 до 1.0 для каждого фрагмента. Если оценка выше порога — фрагмент считается речью и отправляется Whisper для распознавания. Если ниже — считается тишиной и отбрасывается.\n\n"
                               "Меньше значение (0.3) = VAD более чувствительный, отправляет больше аудио (включая фоновый шум), но не пропускает тихую речь\n"
                               "Больше значение (0.8) = VAD строже, отправляет только уверенную речь, но может обрезать начало/конец фраз\n\n"
                               "Не влияет на внешний VAD (Ten VAD для стриминга) — у него свой Vad Threshold.\n\n"
                               "По умолчанию 0.5")
        self.vad_threshold_var = tk.DoubleVar(value=self.config.getfloat("Recognition", "vad_threshold", fallback=0.5))
        vad_threshold_scale = ttk.Scale(self.tab_audio, from_=0.0, to=1.0, variable=self.vad_threshold_var,
                                        orient="horizontal", command=lambda v: (self._round_vad_threshold(v), self._save_config()))
        vad_threshold_scale.grid(row=5, column=1, sticky="we", padx=5)
        ttk.Label(self.tab_audio, textvariable=self.vad_threshold_var).grid(row=5, column=2, sticky="w", padx=5)
        ttk.Label(self.tab_audio, text="0=чувствительный, 1=строгий", foreground="gray", font=("Segoe UI", 7)).grid(row=5, column=3, sticky="w", padx=2)

        # === Внешний VAD (Ten VAD для стриминга) ===
        row = 6
        lbl = ttk.Label(self.tab_audio, text="VAD (внешний, Ten VAD для стриминга):")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=(10, 2))
        self._set_tooltip(lbl, "Включает Ten VAD для определения пауз в режиме стриминга. Весь аудио всегда накапливается (включая паузы). VAD только решает, когда отправить чанк на распознавание. Внутренний VAD (Silero) уберёт паузы из аудио при распознавании")
        row += 1

        self.vad_streaming_var = tk.BooleanVar(value=self.config.getboolean("Recognition", "use_vad_streaming", fallback=True))
        chk2 = ttk.Checkbutton(self.tab_audio, text="VAD (внешний, Ten VAD для стриминга)", variable=self.vad_streaming_var)
        chk2.grid(row=row, column=1, columnspan=3, sticky="w", padx=5, pady=2)
        row += 1

        # Vad Threshold
        lbl = ttk.Label(self.tab_audio, text="Vad Threshold:")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self._set_tooltip(lbl, "Порог уверенности для детекции речи. Чем меньше — тем чувствительнее VAD к тихой речи, но может реагировать на шум. Чем больше — тем строже.\n\n"
                               "0.3 = очень чувствительный\n"
                               "0.5 = рекомендуемое значение\n"
                               "0.7 = строгий, только уверенную речь")
        self.vad_enter_threshold_var = tk.DoubleVar(value=self.config.getfloat("Recognition", "vad_threshold", fallback=0.5))
        vad_enter_scale = ttk.Scale(self.tab_audio, from_=0.3, to=0.7, variable=self.vad_enter_threshold_var,
                                    orient="horizontal", command=lambda v: (self._round_enter_threshold(v), self._save_config()))
        vad_enter_scale.grid(row=row, column=1, sticky="we", padx=5)
        ttk.Label(self.tab_audio, textvariable=self.vad_enter_threshold_var).grid(row=row, column=2, sticky="w", padx=5)
        ttk.Label(self.tab_audio, text="(0.3-0.7)", foreground="gray", font=("Segoe UI", 7)).grid(row=row, column=3, sticky="w", padx=2)
        row += 1

        # Min Silence Duration
        lbl = ttk.Label(self.tab_audio, text="Min Silence Duration:")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self._set_tooltip(lbl, "Минимальная пауза тишины (в секундах), после которой VAD фиксирует конец речи и отправляет чанк на распознавание.\n\n"
                               "Меньше = быстрее отправка, но возможны ложные паузы\n"
                               "Больше = медленнее, но текст более плавный\n\n"
                               "По умолчанию 0.7 сек.")
        self.vad_silence_var = tk.DoubleVar(value=self.config.getfloat("Recognition", "vad_min_silence_duration", fallback=0.7))
        vad_silence_scale = ttk.Scale(self.tab_audio, from_=0.3, to=2.0, variable=self.vad_silence_var,
                                      orient="horizontal", command=lambda v: (self._round_silence(v), self._save_config()))
        vad_silence_scale.grid(row=row, column=1, sticky="we", padx=5)
        ttk.Label(self.tab_audio, textvariable=self.vad_silence_var).grid(row=row, column=2, sticky="w", padx=5)
        ttk.Label(self.tab_audio, text="сек", foreground="gray", font=("Segoe UI", 7)).grid(row=row, column=3, sticky="w", padx=2)
        row += 1

        # Min Speech Duration
        lbl = ttk.Label(self.tab_audio, text="Min Speech Duration:")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self._set_tooltip(lbl, "Минимальная длительность речи для учёта. Если VAD обнаружил речь короче этого времени — она считается шумом и игнорируется.\n\n"
                               "Меньше = чувствительнее (ловит короткие слова)\n"
                               "Больше = строже (игнорирует короткие всплески шума)\n\n"
                               "По умолчанию 0.25 сек.")
        self.min_speech_frames_var = tk.DoubleVar(value=self.config.getfloat("Recognition", "vad_min_speech_duration", fallback=0.25))
        min_speech_scale = ttk.Scale(self.tab_audio, from_=0.1, to=1.0, variable=self.min_speech_frames_var,
                                     orient="horizontal", command=lambda v: (self._round_min_speech(v), self._save_config()))
        min_speech_scale.grid(row=row, column=1, sticky="we", padx=5)
        self.min_speech_label_var = tk.StringVar(value=f"{self.min_speech_frames_var.get() * 1000:.0f}мс")
        ttk.Label(self.tab_audio, textvariable=self.min_speech_label_var).grid(row=row, column=2, sticky="w", padx=5)
        ttk.Label(self.tab_audio, text="(100мс-1000мс)", foreground="gray", font=("Segoe UI", 7)).grid(row=row, column=3, sticky="w", padx=2)
        row += 1

        # Min Chunk Duration (НОВАЯ настройка)
        lbl = ttk.Label(self.tab_audio, text="Min Chunk Duration:")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self._set_tooltip(lbl, "Минимальная длительность накопленного чанка для отправки на распознавание. Если чанк короче — он накапливается к следующему.\n\n"
                               "Весь аудио всегда накапливается (речь + паузы). VAD определяет только моменты отправки.\n\n"
                               "Меньше = быстрее отправка, но «рваный» текст\n"
                               "Больше = плавнее текст, но задержка при коротких фразах\n\n"
                               "По умолчанию 3.0 сек.")
        self.vad_chunk_duration_var = tk.DoubleVar(value=self.config.getfloat("Recognition", "min_chunk_duration", fallback=3.0))
        vad_chunk_scale = ttk.Scale(self.tab_audio, from_=0.5, to=5.0, variable=self.vad_chunk_duration_var,
                                    orient="horizontal", command=lambda v: (self._round_chunk_duration(v), self._save_config()))
        vad_chunk_scale.grid(row=row, column=1, sticky="we", padx=5)
        ttk.Label(self.tab_audio, textvariable=self.vad_chunk_duration_var).grid(row=row, column=2, sticky="w", padx=5)
        ttk.Label(self.tab_audio, text="сек", foreground="gray", font=("Segoe UI", 7)).grid(row=row, column=3, sticky="w", padx=2)
        row += 1

        # Сохраняем ссылки на виджеты для активации/деактивации
        self._vad_enter_scale = vad_enter_scale
        self._vad_silence_scale = vad_silence_scale
        self._vad_threshold_scale = vad_threshold_scale
        self._vad_streaming_chk = chk2
        self._min_speech_scale = min_speech_scale
        self._vad_chunk_scale = vad_chunk_scale

        self.mode_var.trace_add("write", lambda *a: self._on_mode_change())
        self.sample_rate_var.trace_add("write", lambda *a: self._save_config())
        self.gain_var.trace_add("write", lambda *a: self._save_config())
        self.vad_var.trace_add("write", lambda *a: self._on_vad_change())
        self.vad_threshold_var.trace_add("write", lambda *a: self._save_config())
        self.vad_streaming_var.trace_add("write", lambda *a: self._on_vad_streaming_change())
        self.vad_enter_threshold_var.trace_add("write", lambda *a: self._save_config())
        self.vad_silence_var.trace_add("write", lambda *a: self._save_config())
        self.min_speech_frames_var.trace_add("write", lambda *a: self._save_config())
        self.vad_chunk_duration_var.trace_add("write", lambda *a: self._save_config())

        # Инициализация состояния VAD-виджетов
        self._update_vad_widgets()
        self._update_vad_streaming_widgets()

        # Авто-включение VAD при стриминге
        if self.mode_var.get() == "streaming":
            self.vad_var.set(True)
            self.vad_streaming_var.set(True)

        # Привязка изменения метода транскрипции — whisper-server = только GPU
        self.method_var.trace_add("write", lambda *a: self._on_method_change())

    def _get_mic_info(self):
        try:
            p = pyaudio.PyAudio()
            info = p.get_default_input_device_info()
            name = info['name']
            # Исправление кодировки: PyAudio может вернуть OEM-кодировку (CP866)
            if isinstance(name, str):
                # Проверяем: содержит ли имя не-ASCII символы?
                has_non_ascii = any(ord(c) > 127 for c in name)
                if has_non_ascii:
                    # Попробуем декодировать как CP866 → UTF-8
                    try:
                        decoded = name.encode('cp866', errors='strict').decode('utf-8', errors='strict')
                        if decoded.isprintable() and len(decoded) < 200:
                            name = decoded
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        # Если CP866 не подошло — пробуем CP1251 → UTF-8
                        try:
                            decoded = name.encode('cp1251', errors='strict').decode('utf-8', errors='strict')
                            if decoded.isprintable():
                                name = decoded
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            # Если ничего не подошло — пробуем CP1251 с игнором ошибок
                            try:
                                decoded = name.encode('cp1251', errors='ignore').decode('utf-8', errors='ignore')
                                if decoded.isprintable():
                                    name = decoded
                            except:
                                pass  # оставляем оригинальное имя
            txt = f"{name} ({int(info['defaultSampleRate'])} Гц, {info['maxInputChannels']} кан.)"
            p.terminate()
            return txt
        except:
            return "Не удалось получить"

    def _build_hotkeys_tab(self):
        lbl = ttk.Label(self.tab_hotkeys, text="Горячая клавиша:")
        lbl.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self._set_tooltip(lbl, "Клавиша или комбинация для начала/остановки записи. Нажатие=старт, отпускание=стоп+распознавание")
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
        """Переключение VAD при смене режима.
        
        record_then_analyze: внутренний VAD = ВКЛ, внешний VAD = ВЫКЛ
        streaming: внутренний VAD = ВКЛ (убирает паузы из аудио), внешний VAD = ВКЛ (определяет паузы)
        
        Внутренний VAD (Silero) включён ВСЕГДА — он убирает тишину из аудио перед распознаванием.
        Внешний VAD (Ten VAD) включён только в стриминге — определяет когда отправлять чанки.
        """
        is_streaming = self.mode_var.get() == "streaming"
        if is_streaming:
            self.vad_var.set(True)             # включить внутренний VAD (убирает паузы из аудио)
            self.vad_streaming_var.set(True)   # включить внешний VAD (определяет паузы)
        else:
            self.vad_var.set(True)             # включить внутренний VAD
            self.vad_streaming_var.set(False)  # выключить внешний VAD
        # Обновляем виджеты
        self._update_vad_widgets()
        self._update_vad_streaming_visibility()
        self._save_config()

    def _update_vad_streaming_visibility(self):
        """Показывает/скрывает настройки внешнего VAD в зависимости от режима."""
        is_streaming = self.mode_var.get() == "streaming"
        state = "normal" if is_streaming else "disabled"
        self._vad_streaming_chk.configure(state=state)
        self._vad_enter_scale.configure(state=state)
        self._vad_silence_scale.configure(state=state)
        self._min_speech_scale.configure(state=state)
        self._vad_chunk_scale.configure(state=state)
        # Если переключились на record_then_analyze — деактивируем галочку
        if not is_streaming:
            self.vad_streaming_var.set(False)

    def _on_vad_change(self):
        """При изменении галочки внутреннего VAD — активировать/деактивировать Vad Threshold."""
        self._update_vad_widgets()
        self._save_config()

    def _on_vad_streaming_change(self):
        """При изменении галочки внешнего VAD — активировать/деактивировать настройки стриминга."""
        self._update_vad_streaming_widgets()
        self._save_config()

    def _update_vad_widgets(self):
        """Активирует/деактивирует настройки внутреннего VAD в зависимости от галочки."""
        state = "normal" if self.vad_var.get() else "disabled"
        self._vad_threshold_scale.configure(state=state)

    def _update_vad_streaming_widgets(self):
        """Активирует/деактивирует настройки внешнего VAD в зависимости от галочки."""
        state = "normal" if self.vad_streaming_var.get() else "disabled"
        self._vad_enter_scale.configure(state=state)
        self._vad_silence_scale.configure(state=state)
        self._min_speech_scale.configure(state=state)
        self._vad_chunk_scale.configure(state=state)

    def _round_silence(self, value):
        """Округляет значение порога тишины до 2 знаков."""
        self.vad_silence_var.set(round(float(value), 2))

    def _round_vad_threshold(self, value):
        """Округляет значение Vad Threshold до 2 знаков."""
        self.vad_threshold_var.set(round(float(value), 2))

    def _round_min_speech(self, value):
        """Округляет значение vad_min_speech_duration до 2 знаков и обновляет label."""
        duration = round(float(value), 2)
        self.min_speech_frames_var.set(duration)
        self.min_speech_label_var.set(f"{duration * 1000:.0f}мс")

    def _round_enter_threshold(self, value):
        """Округляет значение Vad Enter Threshold до 2 знаков."""
        self.vad_enter_threshold_var.set(round(float(value), 2))

    def _round_chunk_duration(self, value):
        """Округляет значение min_chunk_duration до 2 знаков."""
        self.vad_chunk_duration_var.set(round(float(value), 2))

    def _on_method_change(self):
        """При смене метода транскрипции — сохранить конфиг.
        
        whisper-server теперь работает и на GPU, и на CPU — авто-переключение удалено.
        """
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
            # Собираем изменения для логирования
            changes = []
            settings_map = [
                ("Paths", "whisper_gpu", self.gpu_path_var),
                ("Paths", "whisper_cpu", self.cpu_path_var),
                ("Paths", "model", self.model_path_var),
                ("Paths", "gpu_device_index", lambda: self.gpu_idx_var.get().strip() or "1"),
                ("Paths", "server_gpu_path", lambda: self.server_gpu_path_var.get().strip()),
                ("Paths", "server_cpu_path", lambda: self.server_cpu_path_var.get().strip()),
                ("Recognition", "device", self.device_var),
                ("Recognition", "mode", self.mode_var),
                ("Recognition", "method", self.method_var),
                ("Recognition", "server_url", lambda: self.server_url_var.get().strip()),
                ("Audio", "gain", lambda: str(self.gain_var.get())),
                ("Audio", "sample_rate", self.sample_rate_var),
                ("Recognition", "use_vad", lambda: str(self.vad_var.get()).lower()),
                ("Recognition", "vad_threshold", lambda: str(self.vad_threshold_var.get())),
                ("Recognition", "use_vad_streaming", lambda: str(self.vad_streaming_var.get()).lower()),
                ("Recognition", "vad_min_silence_duration", lambda: str(self.vad_silence_var.get())),
                ("Recognition", "vad_min_speech_duration", lambda: str(self.min_speech_frames_var.get())),
                ("Recognition", "min_chunk_duration", lambda: str(self.vad_chunk_duration_var.get())),
                ("Hotkeys", "hotkey", lambda: self.hotkey_var.get().strip()),
            ]
            for section, key, getter in settings_map:
                new_val = getter() if callable(getter) else getter.get()
                old_val = self.config.get(section, key, fallback="<не задано>")
                if str(new_val) != str(old_val):
                    changes.append(f"  {section}.{key}: {old_val} → {new_val}")
                    self.config[section][key] = str(new_val)
                else:
                    self.config[section][key] = str(new_val)
            save_config(self.config)
            self.core.update_settings(self.config)
            self.hotkey_manager.restart_listeners()
            if changes:
                logging.info("Изменены настройки:\n" + "\n".join(changes))
            else:
                logging.info("Настройки сохранены")
            # Уведомление с автоскрытием через 0.5 сек
            self._show_temp_notification("Настройки сохранены и применены")
        except Exception as e:
            logging.error(f"Ошибка сохранения: {e}")
            messagebox.showerror("Ошибка", f"Не удалось сохранить настройки: {e}")

# ----------------------------------------------------------------------
# Блокировка буфера обмена для стриминга (асинхронные потоки)
# ----------------------------------------------------------------------
clipboard_lock = threading.Lock()

# ----------------------------------------------------------------------
# Конвертация аудио и шумоподавление
# ----------------------------------------------------------------------
CONVERTED_TEMP_WAV = RECORDINGS_DIR / "converted_temp.wav"

def convert_to_wav(input_path):
    """Конвертирует любой аудиофайл в WAV 16kHz mono 16bit через pydub + ffmpeg.
    
    Удаляет старый файл converted_temp.wav если существует.
    Возвращает путь к WAV файлу.
    """
    # Удаляем старый файл
    if CONVERTED_TEMP_WAV.exists():
        try:
            CONVERTED_TEMP_WAV.unlink()
        except Exception as e:
            logging.warning(f"Не удалось удалить {CONVERTED_TEMP_WAV}: {e}")
    
    try:
        audio = AudioSegment.from_file(input_path)
        # Конвертация: 16kHz, mono, 16-bit
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(str(CONVERTED_TEMP_WAV), format="wav")
        logging.info(f"Конвертация: {input_path} → {CONVERTED_TEMP_WAV} (16kHz, mono, 16bit)")
        return str(CONVERTED_TEMP_WAV)
    except Exception as e:
        logging.error(f"Ошибка конвертации {input_path}: {e}")
        raise

def apply_noise_reduction(audio, remove_rumble=False, rumble_freq=80, remove_hiss=False, hiss_freq=8000):
    """Применяет фильтры шумоподавления к аудио через pydub.
    
    Args:
        audio: AudioSegment объект
        remove_rumble: Убрать низкие частоты (гул, рокот) — high_pass_filter
        rumble_freq: Частота high_pass фильтра (40-200 Гц, по умолчанию 80)
        remove_hiss: Убрать высокие частоты (шипение) — low_pass_filter
        hiss_freq: Частота low_pass фильтра (4000-16000 Гц, по умолчанию 8000)
    
    Returns:
        AudioSegment с применёнными фильтрами
    """
    if remove_rumble:
        audio = audio.high_pass_filter(rumble_freq)
        logging.info(f"Фильтр: high_pass {rumble_freq} Гц (убран гул)")
    if remove_hiss:
        audio = audio.low_pass_filter(hiss_freq)
        logging.info(f"Фильтр: low_pass {hiss_freq} Гц (убрано шипение)")
    return audio

# ----------------------------------------------------------------------
# Окно "Выбрать аудиофайл"
# ----------------------------------------------------------------------
class AudioFileWindow:
    """Окно для распознавания аудиофайлов через Whisper.
    
    Поддерживает:
    - Конвертацию любого формата (MP3, FLAC, OGG) → WAV 16kHz mono
    - Шумоподавление (high_pass / low_pass фильтры)
    - Прослушивание обработанного файла
    """

    def __init__(self, root, runner, parent_root):
        self.root = root
        self.runner = runner
        self.parent_root = parent_root
        self.root.title("Whisper Dictation — Распознавание аудиофайла")
        self.root.geometry("650x550")
        self.root.resizable(True, True)

        # Row 0: панель выбора файла
        frame_top = ttk.Frame(root, padding=10)
        frame_top.grid(row=0, column=0, sticky="ew")

        ttk.Label(frame_top, text="Аудиофайл:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.file_path_var = tk.StringVar()
        ttk.Entry(frame_top, textvariable=self.file_path_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(frame_top, text="Выбрать файл", command=self._browse_file).grid(row=0, column=2, padx=2)

        # Row 1: шумоподавление
        frame_noise = ttk.LabelFrame(root, text="Шумоподавление", padding=5)
        frame_noise.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        # Убрать гул (низкие частоты)
        self.remove_rumble_var = tk.BooleanVar(value=False)
        chk_rumble = ttk.Checkbutton(frame_noise, text="Убрать гул (низкие частоты)", variable=self.remove_rumble_var)
        chk_rumble.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self._set_tooltip(chk_rumble, "High-pass фильтр — убирает низкие частоты (гул, рокот, шум кондиционера).\n\n"
                                       "Чем меньше частота — тем больше низких частот останется\n"
                                       "Чем больше частота — тем больше низких частот будет убрано\n\n"
                                       "Рекомендуется: 80 Гц")
        self.rumble_freq_var = tk.IntVar(value=80)
        scale_rumble = ttk.Scale(frame_noise, from_=40, to=200, variable=self.rumble_freq_var,
                  orient="horizontal", command=lambda v: self._update_rumble_label())
        scale_rumble.grid(row=0, column=1, sticky="we", padx=5)
        self._set_tooltip(scale_rumble, "Частота high-pass фильтра (40-200 Гц).\n\n"
                                       "40 Гц — только очень низкие частоты\n"
                                       "80 Гц — рекомендуемое значение\n"
                                       "200 Гц — убирает много низких частот, голос может стать тонким")
        self.rumble_label_var = tk.StringVar(value="80 Гц")
        ttk.Label(frame_noise, textvariable=self.rumble_label_var, foreground="gray",
                  font=("Segoe UI", 7)).grid(row=0, column=2, sticky="w", padx=2)

        # Убрать шипение (высокие частоты)
        self.remove_hiss_var = tk.BooleanVar(value=False)
        chk_hiss = ttk.Checkbutton(frame_noise, text="Убрать шипение (высокие частоты)", variable=self.remove_hiss_var)
        chk_hiss.grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self._set_tooltip(chk_hiss, "Low-pass фильтр — убирает высокие частоты (шипение, фон, свист).\n\n"
                                     "Чем меньше частота — тем больше высоких частот будет убрано\n"
                                     "Чем больше частота — тем больше высоких частот останется\n\n"
                                     "Рекомендуется: 8000 Гц")
        self.hiss_freq_var = tk.IntVar(value=8000)
        scale_hiss = ttk.Scale(frame_noise, from_=4000, to=16000, variable=self.hiss_freq_var,
                  orient="horizontal", command=lambda v: self._update_hiss_label())
        scale_hiss.grid(row=1, column=1, sticky="we", padx=5)
        self._set_tooltip(scale_hiss, "Частота low-pass фильтра (4000-16000 Гц).\n\n"
                                       "4000 Гц — убирает много высоких частот, голос может стать глухим\n"
                                       "8000 Гц — рекомендуемое значение\n"
                                       "16000 Гц — оставляет почти все высокие частоты")
        self.hiss_label_var = tk.StringVar(value="8000 Гц")
        ttk.Label(frame_noise, textvariable=self.hiss_label_var, foreground="gray",
                  font=("Segoe UI", 7)).grid(row=1, column=2, sticky="w", padx=2)

        frame_noise.grid_columnconfigure(1, weight=1)

        # Row 2: кнопки действий
        frame_btn = ttk.Frame(root, padding=(10, 5))
        frame_btn.grid(row=2, column=0, sticky="ew")

        self.btn_transcribe = ttk.Button(frame_btn, text="Распознать", command=self._transcribe_file)
        self.btn_transcribe.grid(row=0, column=0, sticky="w", padx=(0, 5))

        self.btn_play = ttk.Button(frame_btn, text="▶ Прослушать", command=self._play_processed)
        self.btn_play.grid(row=0, column=1, sticky="w", padx=(0, 5))
        self._set_tooltip(self.btn_play, "Открывает обработанный WAV файл в плеере Windows.\n\n"
                                          "Позволяет прослушать результат шумоподавления и оценить качество.\n\n"
                                          "Файл создаётся при распознавании с включённым шумоподавлением.")

        # Статус
        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(frame_btn, textvariable=self.status_var, foreground="gray",
                  font=("Segoe UI", 8)).grid(row=0, column=2, sticky="e", padx=(20, 0))

        # Row 3: текстовое поле + скролл
        frame_text = ttk.Frame(root, padding=(10, 5))
        frame_text.grid(row=3, column=0, sticky="nsew")

        self.text_var = tk.Text(frame_text, wrap="word", state="disabled", font=("Consolas", 10), height=15)
        scrollbar = ttk.Scrollbar(frame_text, orient="vertical", command=self.text_var.yview)
        self.text_var.configure(yscrollcommand=scrollbar.set)
        self.text_var.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Row 4: кнопки внизу
        frame_bottom = ttk.Frame(root, padding=10)
        frame_bottom.grid(row=4, column=0, sticky="ew")

        ttk.Button(frame_bottom, text="Скопировать всё", command=self._copy_all).grid(row=0, column=0, padx=5)
        ttk.Button(frame_bottom, text="Очистить", command=self._clear_text).grid(row=0, column=1, padx=5)

        # Настройка весов
        root.grid_rowconfigure(3, weight=1)
        root.grid_columnconfigure(0, weight=1)
        frame_text.grid_rowconfigure(0, weight=1)
        frame_text.grid_columnconfigure(0, weight=1)

    def _set_tooltip(self, widget, text):
        """Добавляет всплывающую подсказку к виджету."""
        def _on_enter(e):
            self._tooltip = tk.Toplevel(widget)
            self._tooltip.wm_overrideredirect(True)
            self._tooltip.wm_geometry(f"+{e.x_root+10}+{e.y_root+10}")
            label = tk.Label(self._tooltip, text=text, justify=tk.LEFT,
                            background="#ffffe0", relief="solid", borderwidth=1,
                            font=("Segoe UI", 8), wraplength=300)
            label.pack()
        def _on_leave(e):
            if hasattr(self, '_tooltip'):
                self._tooltip.destroy()
                del self._tooltip
        widget.bind("<Enter>", _on_enter)
        widget.bind("<Leave>", _on_leave)

    def _update_rumble_label(self):
        self.rumble_label_var.set(f"{self.rumble_freq_var.get()} Гц")

    def _update_hiss_label(self):
        self.hiss_label_var.set(f"{self.hiss_freq_var.get()} Гц")

    def _browse_file(self):
        filepath = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[
                ("Аудиофайлы", "*.wav *.mp3 *.flac *.ogg *.m4a *.wma"),
                ("Все файлы", "*.*"),
            ],
        )
        if filepath:
            self.file_path_var.set(filepath)

    def _transcribe_file(self):
        """Полный пайплайн: конвертация → фильтры → сохранение → транскрипция."""
        filepath = self.file_path_var.get().strip()
        if not filepath:
            self.status_var.set("Ошибка: не выбран файл")
            return
        if not os.path.exists(filepath):
            self.status_var.set(f"Ошибка: файл не найден — {filepath}")
            return

        self.status_var.set("Обработка...")
        self.btn_transcribe.configure(state="disabled")

        def _work():
            wav_path = filepath
            try:
                # 1. Конвертация (если не WAV)
                if not filepath.lower().endswith('.wav'):
                    self.root.after(0, lambda: self.status_var.set("Конвертация..."))
                    wav_path = convert_to_wav(filepath)
                
                # 2. Загрузка аудио для применения фильтров
                remove_rumble = self.remove_rumble_var.get()
                remove_hiss = self.remove_hiss_var.get()
                rumble_freq = self.rumble_freq_var.get()
                hiss_freq = self.hiss_freq_var.get()
                
                if remove_rumble or remove_hiss:
                    self.root.after(0, lambda: self.status_var.set("Применение фильтров..."))
                    audio = AudioSegment.from_file(wav_path)
                    audio = apply_noise_reduction(
                        audio,
                        remove_rumble=remove_rumble,
                        rumble_freq=rumble_freq,
                        remove_hiss=remove_hiss,
                        hiss_freq=hiss_freq,
                    )
                    # Сохраняем обработанный файл в converted_temp.wav
                    if CONVERTED_TEMP_WAV.exists():
                        try:
                            CONVERTED_TEMP_WAV.unlink()
                        except:
                            pass
                    audio.export(str(CONVERTED_TEMP_WAV), format="wav")
                    wav_path = str(CONVERTED_TEMP_WAV)
                    logging.info(f"Фильтры применены, сохранено: {CONVERTED_TEMP_WAV}")
                
                # 3. Транскрипция
                self.root.after(0, lambda: self.status_var.set("Распознаю..."))
                text = self.runner.transcribe(wav_path)
                self.root.after(0, lambda: self._append_text(text))
            except Exception as e:
                logging.error(f"Ошибка обработки файла {filepath}: {e}")
                self.root.after(0, lambda: self._append_text(f"[Ошибка: {e}]"))
            finally:
                self.root.after(0, lambda: (
                    self.btn_transcribe.configure(state="normal"),
                    self.status_var.set("Готово"),
                ))

        threading.Thread(target=_work, daemon=True).start()

    def _play_processed(self):
        """Открывает обработанный WAV файл в плеере Windows."""
        if CONVERTED_TEMP_WAV.exists():
            try:
                os.startfile(str(CONVERTED_TEMP_WAV))
                logging.info(f"Прослушивание: {CONVERTED_TEMP_WAV}")
            except Exception as e:
                logging.error(f"Не удалось открыть файл: {e}")
                messagebox.showerror("Ошибка", f"Не удалось открыть файл: {e}")
        else:
            messagebox.showinfo("Инфо", "Сначала распознайте файл с шумоподавлением, чтобы создать обработанный файл")

    def _append_text(self, text):
        self.text_var.configure(state="normal")
        current = self.text_var.get("1.0", "end-1c")
        if current.strip():
            self.text_var.insert("end", "\n---\n")
        self.text_var.insert("end", text.strip())
        self.text_var.see("end")
        self.text_var.configure(state="disabled")

    def _copy_all(self):
        text = self.text_var.get("1.0", "end-1c").strip()
        if text:
            pyperclip.copy(text)

    def _clear_text(self):
        self.text_var.configure(state="normal")
        self.text_var.delete("1.0", "end")
        self.text_var.configure(state="disabled")


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
        self.settings_window = None
        self.audio_file_window = None

        # Проверка микрофона при старте
        if not core.audio.mic_available:
            self.paused = True
            self.hotkey_manager.paused = True
            logging.warning("Микрофон не найден — программа запущена в режиме паузы")

        self.icon = pystray.Icon("whisper_dictation")
        self.icon.icon = self._create_image(active=not self.paused)
        self.icon.title = "Whisper Dictation (Paused)" if self.paused else "Whisper Dictation"
        self._setup_menu()
        self.icon.on_click = self._on_click

        # Показать уведомление если нет микрофона
        if not core.audio.mic_available:
            self._show_toast("Микрофон не найден", "Программа в режиме паузы. Нажмите правую кнопку мыши на иконке для настроек.", duration=3)

    def _show_toast(self, title, message, duration=2):
        """Показывает всплывающее уведомление (тоаст) с автозакрытием."""
        def _show():
            toast = tk.Toplevel(self.root)
            toast.title(title)
            toast.attributes("-topmost", True)
            toast.attributes("-toolwindow", True)
            toast.geometry(f"350x80+{self.root.winfo_screenwidth() - 380}+{self.root.winfo_screenheight() - 120}")
            toast.resizable(False, False)
            toast.configure(bg="#28a745")
            ttk.Label(toast, text=message, background="#28a745", foreground="white",
                      font=("Segoe UI", 9)).pack(padx=10, pady=10, fill="both", expand=True)
            self.root.after(duration * 1000, lambda: (toast.destroy()))
        self.root.after(0, _show)

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
            pystray.MenuItem("Выбрать аудиофайл", self._on_audio_file),
            pystray.MenuItem("Настройки", self._on_settings),
            pystray.MenuItem("Выход", self._on_exit),
        )
        self.icon.menu = menu

    def _on_audio_file(self, icon, item):
        """Открывает окно для распознавания аудиофайла."""
        if self.audio_file_window is not None:
            try:
                self.audio_file_window.deiconify()
                self.audio_file_window.lift()
                return
            except:
                self.audio_file_window = None
        win_root = tk.Toplevel(self.root)
        self.audio_file_window = win_root
        AudioFileWindow(win_root, self.core.runner, self.root)
        win_root.protocol("WM_DELETE_WINDOW", lambda: self._on_audio_file_close(win_root))

    def _on_audio_file_close(self, window):
        window.destroy()
        self.audio_file_window = None

    def _check_mic_and_resume(self, icon):
        """Проверяет микрофон перед снятием паузы. Если найден — снимает с паузы."""
        try:
            p = pyaudio.PyAudio()
            p.get_default_input_device_info()
            p.terminate()
            # Микрофон найден
            self.paused = False
            self.hotkey_manager.paused = False
            icon.icon = self._create_image(active=True)
            icon.title = "Whisper Dictation"
            self._show_toast("Микрофон найден", "Программа готова к работе. Нажмите среднюю кнопку мыши для начала диктовки.", duration=2)
            logging.info("Микрофон найден, пауза снята")
        except Exception as e:
            # Микрофон не найден — остаёмся на паузе
            self._show_toast("Микрофон не найден", "Подключите микрофон или выберите его по умолчанию в настройках Windows.", duration=2)
            logging.warning(f"Микрофон не найден при снятии паузы: {e}")

    def _toggle_pause(self, icon, item):
        if self.paused:
            # Пытаемся снять паузу — проверяем микрофон
            self._check_mic_and_resume(icon)
        else:
            self.paused = True
            self.hotkey_manager.paused = True
            icon.icon = self._create_image(active=False)
            icon.title = "Whisper Dictation (Paused)"
            logging.info("Пауза включена")

    def _on_click(self, icon, pos, button, action):
        if button == pystray.Button.LEFT:
            if self.paused:
                self._check_mic_and_resume(icon)
            else:
                self.paused = True
                self.hotkey_manager.paused = True
                icon.icon = self._create_image(active=False)
                icon.title = "Whisper Dictation (Paused)"
                logging.info("Пауза включена")

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