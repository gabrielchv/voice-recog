#!/usr/bin/env python3
"""
🎙️ Voz para Texto pt-BR — Leve com GPU CUDA
=============================================
Usa faster-whisper com aceleração GPU (CTranslate2 + CUDA)
Modelo otimizado para português brasileiro.

Uso:
  ./stt.py                  # Modo contínuo (escuta e transcreve automaticamente)
  ./stt.py --oneshot        # Grava uma vez e transcreve
  ./stt.py --file audio.wav # Transcreve um arquivo de áudio
  ./stt.py --model small    # Escolhe o modelo (tiny|base|small|medium)
"""

import argparse
import atexit
import logging
import os
import queue
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# ── Silencia logs verbosos ──────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

# ── Imports (com tratamento para dependências opcionais) ────────────
try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import pyperclip
except ImportError:
    pyperclip = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

# ── Constantes ──────────────────────────────────────────────────────
WHISPER_SR = 16000           # faster-whisper sample rate
CHANNELS = 1                 # Mono
DTYPE = "int16"              # Formato de áudio
BLOCK_SECONDS = 0.5          # 0.5s por bloco de áudio
SILENCE_THRESHOLD = 500      # RMS mínimo para considerar fala (16-bit)
SILENCE_SECONDS = 1.5        # Segundos de silêncio para finalizar frase
MIN_DURATION = 1.0           # Duração mínima para transcrever (segundos)
MAX_DURATION = 45            # Duração máxima de uma frase
VAD_WINDOW = 0.5             # Janela para detectar início da fala (segundos)

# Caminho dos modelos
MODEL_DIR = Path(__file__).parent / "models"

# Cores ANSI
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
ERASE_LINE = "\033[2K"
CURSOR_UP = "\033[A"


# ══════════════════════════════════════════════════════════════════════
#  DETECTOR DE ATIVIDADE DE VOZ (energy-based)
# ══════════════════════════════════════════════════════════════════════

class VAD:
    """Voice Activity Detection simples por energia."""

    def __init__(self, threshold=SILENCE_THRESHOLD, sample_rate=WHISPER_SR):
        self.threshold = threshold
        self.sample_rate = sample_rate

    def _preprocess(self, block: np.ndarray) -> np.ndarray:
        """Remove DC offset e converte para float32."""
        block_f = block.astype(np.float32)
        # Remove DC offset (filtro passa-alta simples)
        return block_f - block_f.mean()

    def is_speech(self, block: np.ndarray) -> bool:
        """Retorna True se o bloco contém fala.

        Usa RMS com remoção de DC offset para robustez.
        """
        clean = self._preprocess(block)
        rms = float(np.sqrt(np.mean(clean ** 2)))
        peak = float(np.abs(clean).max())
        # Fala típica: RMS 1000-8000 para int16. Silêncio: RMS 10-200
        return rms > self.threshold or peak > self.threshold * 4

    def speech_ratio(self, block: np.ndarray) -> float:
        """Proporção de amostras acima do threshold."""
        clean = self._preprocess(block)
        return float(np.mean(np.abs(clean) > self.threshold))


# ══════════════════════════════════════════════════════════════════════
#  GRAVADOR COM DETECÇÃO DE SILÊNCIO
# ══════════════════════════════════════════════════════════════════════

class Recorder:
    """Grava áudio do microfone com detecção de silêncio.

    Usa uma thread separada para ler o áudio de forma confiável,
    evitando problemas de overflow do callback.
    """

    def __init__(self, device=None):
        self.device = device
        self.device_sr = None
        self.audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self.audio_buffer: list[np.ndarray] = []
        self.is_recording = False
        self.silence_start: float | None = None
        self.speech_start: float | None = None
        self.recording_start: float | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Callbacks
        self.on_speech_start = lambda: None
        self.on_recording = lambda dur: None
        self.on_utterance = lambda audio, sr: None

    @staticmethod
    def _resample_to_whisper(audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Reamostra áudio para 16kHz (taxa do Whisper)."""
        if orig_sr == WHISPER_SR:
            return audio
        import scipy.signal
        num_samples = int(len(audio) * WHISPER_SR / orig_sr)
        return scipy.signal.resample(audio, num_samples).astype(np.int16)

    def _reader(self):
        """Thread que lê áudio do microfone e coloca na fila."""
        try:
            stream = sd.InputStream(
                device=self.device,
                samplerate=self.device_sr,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=0,  # usa tamanho padrão do sistema
            )
            stream.start()
            while not self._stop_event.is_set():
                block, overflowed = stream.read(int(self.device_sr * BLOCK_SECONDS))
                if overflowed:
                    pass  # overflow é comum no ALSA, ignorar
                if len(block) > 0:
                    self.audio_queue.put(block[:, 0])  # mono
            stream.stop()
            stream.close()
        except Exception as e:
            print(f"{RED}⚠ Erro no áudio: {e}{RESET}", file=sys.stderr)

    def _process_loop(self):
        """Loop principal: processa blocos da fila com VAD."""
        vad = VAD(threshold=SILENCE_THRESHOLD)

        while not self._stop_event.is_set():
            try:
                block = self.audio_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if block is None:
                break

            has_speech = vad.is_speech(block)
            now = time.time()

            if not self.is_recording:
                # ── Modo escuta (esperando fala) ──
                if has_speech:
                    if self.speech_start is None:
                        self.speech_start = now
                    elif (now - self.speech_start) >= VAD_WINDOW:
                        self.is_recording = True
                        self.audio_buffer = [block]
                        self.recording_start = now
                        self.silence_start = None
                        self.speech_start = None
                        self.on_speech_start()
                else:
                    self.speech_start = None
            else:
                # ── Modo gravando ──
                self.audio_buffer.append(block)
                elapsed = now - self.recording_start
                self.on_recording(elapsed)

                if not has_speech:
                    if self.silence_start is None:
                        self.silence_start = now
                    elif (now - self.silence_start) >= SILENCE_SECONDS:
                        if elapsed >= MIN_DURATION:
                            self._finalize_utterance()
                        else:
                            self._reset_recording()
                else:
                    self.silence_start = None

                if elapsed >= MAX_DURATION:
                    self._finalize_utterance()

    def _finalize_utterance(self):
        """Finaliza a gravação atual."""
        if not self.audio_buffer:
            self._reset_recording()
            return
        audio = np.concatenate(self.audio_buffer)
        self._reset_recording()
        audio_16k = self._resample_to_whisper(audio, self.device_sr)
        self.on_utterance(audio_16k, self.device_sr)

    def _reset_recording(self):
        """Reseta o estado de gravação."""
        self.is_recording = False
        self.audio_buffer = []
        self.silence_start = None
        self.recording_start = None
        self.speech_start = None

    def start(self):
        """Inicia a captura contínua do microfone."""
        if sd is None:
            print(f"{RED}❌ sounddevice não instalado.{RESET}")
            sys.exit(1)

        device_info = sd.query_devices(self.device, 'input')
        self.device_sr = int(device_info['default_samplerate'])
        print(f"{DIM}🎤 Microfone: {device_info['name']} ({self.device_sr} Hz){RESET}")

        self._stop_event.clear()
        # Limpa a fila de eventos anteriores
        while not self.audio_queue.empty():
            try: self.audio_queue.get_nowait()
            except queue.Empty: break

        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()

        # Processa em background (a menos que estejamos em modo oneshot)
        threading.Thread(target=self._process_loop, daemon=True).start()

    def stop(self):
        """Para a captura."""
        self._stop_event.set()

    def record_once(self, timeout=MAX_DURATION + 5):
        """Grava uma única frase (modo oneshot). Retorna áudio em 16kHz."""
        result = [None]

        def _done(audio, _):
            result[0] = audio
            self._stop_event.set()

        self.on_utterance = _done
        self.device_sr = int(sd.query_devices(self.device, 'input')['default_samplerate'])
        self._stop_event.clear()

        # Processa manualmente em modo oneshot (bloqueante)
        stream = sd.InputStream(
            device=self.device,
            samplerate=self.device_sr,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=0,
        )
        stream.start()

        vad = VAD(threshold=SILENCE_THRESHOLD)
        audio_buffer = []
        is_recording = False
        silence_start = None
        speech_start = None
        rec_start = None

        deadline = time.time() + timeout
        while time.time() < deadline and result[0] is None:
            block, overflowed = stream.read(int(self.device_sr * BLOCK_SECONDS))
            if overflowed:
                pass
            if len(block) == 0:
                continue
            block = block[:, 0]
            has_speech = vad.is_speech(block)
            now = time.time()

            if not is_recording:
                if has_speech:
                    if speech_start is None:
                        speech_start = now
                    elif (now - speech_start) >= VAD_WINDOW:
                        is_recording = True
                        audio_buffer = [block]
                        rec_start = now
                        silence_start = None
                        speech_start = None
                        sys.stdout.write(f"\n{GREEN}🔴 Gravando...{RESET}")
                        sys.stdout.flush()
                else:
                    speech_start = None
            else:
                audio_buffer.append(block)
                elapsed = now - rec_start

                bar_len = 20
                filled = min(int(elapsed / MAX_DURATION * bar_len), bar_len)
                bar = "▓" * filled + "░" * (bar_len - filled)
                sys.stdout.write(f"\r{GREEN}🔴 {elapsed:.1f}s {bar}{RESET}")
                sys.stdout.flush()

                if not has_speech:
                    if silence_start is None:
                        silence_start = now
                    elif (now - silence_start) >= SILENCE_SECONDS:
                        if elapsed >= MIN_DURATION and audio_buffer:
                            audio = np.concatenate(audio_buffer)
                            audio_16k = self._resample_to_whisper(audio, self.device_sr)
                            result[0] = audio_16k
                        break
                else:
                    silence_start = None

                if elapsed >= MAX_DURATION and audio_buffer:
                    audio = np.concatenate(audio_buffer)
                    audio_16k = self._resample_to_whisper(audio, self.device_sr)
                    result[0] = audio_16k
                    break

        stream.stop()
        stream.close()
        return result[0]


# ══════════════════════════════════════════════════════════════════════
#  TRANSCRITOR (faster-whisper)
# ══════════════════════════════════════════════════════════════════════

class Transcriber:
    """Transcrição de áudio usando faster-whisper."""

    MODELS = {
        "tiny":   {"size": "tiny",   "vram_gb": 0.4},
        "base":   {"size": "base",   "vram_gb": 0.7},
        "small":  {"size": "small",  "vram_gb": 1.2},
        "medium": {"size": "medium", "vram_gb": 2.5},
    }

    def __init__(self, model_name="base", device="cuda", compute_type="int8_float16",
                 cpu_threads=2, num_workers=1):
        if WhisperModel is None:
            print(f"{RED}❌ faster-whisper não instalado.{RESET}")
            print(f"   Rode: source venv/bin/activate && pip install faster-whisper")
            sys.exit(1)

        self.model_name = model_name
        model_info = self.MODELS.get(model_name, self.MODELS["base"])
        self.actual_model = model_info["size"]

        # Decide device
        if device == "auto":
            try:
                import ctranslate2
                if ctranslate2.get_cuda_device_count() > 0:
                    device = "cuda"
                else:
                    device = "cpu"
            except Exception:
                device = "cpu"

        print(f"{DIM}📦 Modelo: {BOLD}{self.actual_model}{RESET}{DIM} | Dispositivo: {device} | "
              f"Quant: {compute_type}{RESET}")

        os.environ["GGML_CUDA_NO_PINNED"] = "1"  # ajuda em GPU com pouca VRAM

        self.model = WhisperModel(
            model_size_or_path=self.actual_model,
            device=device,
            compute_type=compute_type,
            download_root=str(MODEL_DIR),
            cpu_threads=cpu_threads,
            num_workers=num_workers,
            local_files_only=False,
        )

    def transcribe(self, audio: np.ndarray, language="pt", task="transcribe") -> str | None:
        """Transcreve um array numpy de áudio (16kHz, mono, int16)."""
        if audio is None or len(audio) == 0:
            return None

        # faster-whisper espera float32 no range [-1, 1]
        audio_float = audio.astype(np.float32) / 32768.0

        segments, info = self.model.transcribe(
            audio_float,
            language=language,
            task=task,
            beam_size=3,
            best_of=3,
            temperature=0.0,
            vad_filter=False,        # VAD já feito no recording
            no_speech_threshold=0.35,
            log_prob_threshold=-2.0,
            condition_on_previous_text=False,
        )

        text_parts = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                text_parts.append(text)

        return " ".join(text_parts) if text_parts else None

    def transcribe_file(self, filepath: str, language="pt") -> str | None:
        """Transcreve um arquivo de áudio."""
        import soundfile as sf
        audio, sr = sf.read(filepath)
        if sr != WHISPER_SR:
            import scipy.signal
            from scipy.signal import resample
            # Reamostragem simples
            num_samples = int(len(audio) * WHISPER_SR / sr)
            audio = resample(audio, num_samples)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio_int16 = (audio * 32768).astype(np.int16)
        return self.transcribe(audio_int16, language=language)


# ══════════════════════════════════════════════════════════════════════
#  INTERFACE DO USUÁRIO
# ══════════════════════════════════════════════════════════════════════

class VoiceToTextApp:
    """Aplicação principal de voz para texto."""

    def __init__(self, model_name="base", copy_to_clipboard=True, device="cuda",
                 compute_type="int8_float16"):
        self.copy_to_clipboard = copy_to_clipboard and (pyperclip is not None)
        self.last_text = ""
        self.utterance_count = 0

        # Inicializa componentes
        print(f"\n{BOLD}🎙️  Voz para Texto pt-BR{RESET}")
        print(f"{DIM}{'─' * 45}{RESET}")

        self.transcriber = Transcriber(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
        )

        self.recorder = Recorder()
        self.recorder.on_speech_start = self._on_speech_start
        self.recorder.on_recording = self._on_recording
        self.recorder.on_utterance = self._on_utterance

        # Registra limpeza na saída
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _on_speech_start(self):
        """Chamado quando detecta início da fala."""
        print(f"\n{GREEN}🔴 Gravando...{RESET}", end="", flush=True)

    def _on_recording(self, elapsed: float):
        """Chamado durante a gravação para feedback visual."""
        # Barra de progresso simples
        bar_len = 20
        filled = min(int(elapsed / MAX_DURATION * bar_len), bar_len)
        bar = "▓" * filled + "░" * (bar_len - filled)
        dur_str = f"{elapsed:.1f}s"
        sys.stdout.write(f"\r{GREEN}🔴 Gravando {dur_str} {bar}{RESET}")
        sys.stdout.flush()

    def _on_utterance(self, audio: np.ndarray, orig_sr: int = WHISPER_SR):
        """Chamado quando uma frase completa é detectada."""
        dur = len(audio) / WHISPER_SR

        # Mostra feedback imediato
        sys.stdout.write(f"\r{CYAN}⚡ Transcrevendo... ({dur:.1f}s)   {RESET}")
        sys.stdout.flush()

        # Transcreve (bloqueante, mas rápido com GPU)
        text = self.transcriber.transcribe(audio)

        # Limpa a linha de progresso
        sys.stdout.write(f"\r{' ' * 60}\r")
        sys.stdout.flush()

        if text and text.strip():
            self.utterance_count += 1
            self.last_text = text

            # Mostra resultado
            print(f"{GREEN}┌─ [{self.utterance_count}] Transcrição:{RESET}")
            print(f"{GREEN}│{RESET} {text}")

            # Copia para área de transferência
            if self.copy_to_clipboard:
                try:
                    pyperclip.copy(text)
                    print(f"{GREEN}│{RESET} {DIM}📋 Copiado!{RESET}")
                except Exception:
                    pass

            print(f"{GREEN}└─ {DIM}⏱  {dur:.1f}s  |  Ctrl+C para sair{RESET}\n")
        else:
            print(f"{YELLOW}⚠  Nada detectado{RESET}\n")

    def _signal_handler(self, signum, frame):
        """Trata Ctrl+C."""
        self.cleanup()

    def cleanup(self):
        """Limpeza ao sair."""
        self.recorder.stop()

    def run(self):
        """Inicia o loop principal de escuta."""
        print(f"{DIM}🎧 Escutando... (fale algo){RESET}")
        print(f"{DIM}   Ctrl+C para sair{RESET}\n")
        self.recorder.start()

        # Mantém vivo
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
            print(f"\n{DIM}👋 Até logo!{RESET}")

    def run_oneshot(self):
        """Modo one-shot: grava uma vez e transcreve."""
        print(f"{CYAN}🎤 Grave uma frase (silêncio de {SILENCE_SECONDS}s finaliza)...{RESET}")

        audio = self.recorder.record_once()

        if audio is not None and len(audio) / WHISPER_SR >= MIN_DURATION:
            dur = len(audio) / WHISPER_SR
            print(f"\n{CYAN}⚡ Transcrevendo...{RESET}")
            text = self.transcriber.transcribe(audio)
            if text:
                print(f"\n{GREEN}📝 Transcrição:{RESET} {text}")
                if self.copy_to_clipboard:
                    try:
                        pyperclip.copy(text)
                        print(f"{DIM}📋 Copiado para área de transferência!{RESET}")
                    except Exception:
                        pass
            else:
                print(f"{YELLOW}⚠  Nada detectado.{RESET}")
        else:
            print(f"{YELLOW}⚠  Gravação muito curta ou vazia.{RESET}")


# ══════════════════════════════════════════════════════════════════════
#  PONTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════

def check_deps():
    """Verifica dependências e mostra ajuda se faltar algo."""
    missing = []
    if sd is None:
        missing.append("sounddevice")
    if WhisperModel is None:
        missing.append("faster-whisper")

    if missing:
        print(f"{RED}❌ Dependências faltando: {', '.join(missing)}{RESET}")
        print(f"\n   Para instalar:")
        print(f"   source {Path(__file__).parent / 'venv' / 'bin' / 'activate'}")
        print(f"   pip install {' '.join(missing)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="🎙️  Voz para Texto pt-BR — Leve com GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  %(prog)s                    # Modo contínuo (escuta e transcreve)
  %(prog)s --oneshot          # Grava uma frase e transcreve
  %(prog)s --model small      # Usa modelo small (melhor qualidade)
  %(prog)s --file audio.wav   # Transcreve um arquivo
  %(prog)s --device cpu       # Força CPU (mais leve ainda)
        """,
    )
    parser.add_argument(
        "--model", default="base", choices=["tiny", "base", "small", "medium"],
        help="Modelo Whisper (tiny=mais rápido, base~pequeno, medium=preciso)"
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu", "auto"],
        help="Dispositivo para inferência (cuda usa GPU)"
    )
    parser.add_argument(
        "--compute", default="int8_float16",
        choices=["int8", "int8_float16", "float16", "float32"],
        help="Tipo de quantização (int8_float16 = menor VRAM)"
    )
    parser.add_argument(
        "--oneshot", action="store_true",
        help="Grava apenas uma frase e sai"
    )
    parser.add_argument(
        "--file", type=str, metavar="AUDIO",
        help="Transcreve um arquivo de áudio (.wav, .mp3, .flac, .ogg)"
    )
    parser.add_argument(
        "--no-clipboard", action="store_true",
        help="Não copiar para área de transferência"
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="Lista dispositivos de áudio disponíveis"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Mostra informações de debug (RMS do áudio)"
    )

    args = parser.parse_args()

    # Lista dispositivos
    if args.list_devices:
        if sd is None:
            print(f"{RED}sounddevice não instalado.{RESET}")
            sys.exit(1)
        print(f"{BOLD}Dispositivos de áudio:{RESET}")
        print(sd.query_devices())
        return

    check_deps()

    # Modo arquivo
    if args.file:
        if not os.path.exists(args.file):
            print(f"{RED}❌ Arquivo não encontrado: {args.file}{RESET}")
            sys.exit(1)
        transcriber = Transcriber(
            model_name=args.model,
            device=args.device,
            compute_type=args.compute,
        )
        print(f"{CYAN}📄 Transcrevendo: {args.file}{RESET}")
        text = transcriber.transcribe_file(args.file)
        if text:
            print(f"\n{GREEN}📝 Transcrição:{RESET} {text}")
            if not args.no_clipboard and pyperclip:
                pyperclip.copy(text)
                print(f"{DIM}📋 Copiado!{RESET}")
        else:
            print(f"{YELLOW}⚠  Nada detectado.{RESET}")
        return

    # Modo app
    app = VoiceToTextApp(
        model_name=args.model,
        copy_to_clipboard=not args.no_clipboard,
        device=args.device,
        compute_type=args.compute,
    )

    if args.oneshot:
        app.run_oneshot()
    else:
        app.run()


if __name__ == "__main__":
    main()
