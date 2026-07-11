# Изменения: VAD + исправления (11.07.2026)

## VAD (Voice Activity Detection)
- **webrtcvad** — установлен, добавлен в requirements.txt
- **_streaming_record** — переписан с VAD: разделение по паузам речи (порог 0.8 сек), принудительная отправка через 10 сек
- **Silero VAD** — модель скачана (`models/for-tests-silero-v6.2.0-ggml.bin`, 885 КБ)
- **use_vad** — добавлен в DEFAULT_CONFIG, чекбокс в GUI на вкладке "Звук"
- **Флаги VAD** — добавлены в `_auto_start_server_if_needed` и `_transcribe_with_cli`

## Исправления
1. **Запись в моно** — везде `channels=1` (достаточно для речи, работает с VAD)
2. **whisper-cli диагностика** — логирование stderr/stdout, fallback из stdout если .txt не создан
3. **Авто-включение VAD** — при режиме `streaming` автоматически включается чекбокс VAD
4. **Индекс GPU** — Entry заменён на Combobox (0, 1), автоопределение через `--list-devices` (uma=0 = дискретная GPU)
5. **CPU без --device** — CPU-версия не поддерживает `--device`, убран флаг
6. **whisper-server = только GPU** — при выборе whisper-server автоматически `device=gnu`
7. **Уведомление** — messagebox заменён на автоскрытие (0.5 сек)

## Улучшения
- **Защита от дублирования** — named mutex через pywin32 (не запускает вторую копию)

## Тесты
- **test_vad.py** — тест webrtcvad (PASS), Silero модель (PASS), GPU автоопределение (PASS)

## Файлы
- `whisper_dictation.py` — основной код
- `test_vad.py` — тест VAD
- `models/for-tests-silero-v6.2.0-ggml.bin` — Silero VAD модель
- `requirements.txt` — добавлен webrtcvad