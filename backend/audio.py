"""Cross-platform audio dispatcher.

ONE worker thread owns the TTS engine and a bounded queue, so a siren request
every frame can no longer spawn thousands of threads, and a hung TTS driver
only stalls audio, never the video loop.
"""
import time
import queue
import threading

try:
    import winsound
except ImportError:          # Linux / macOS
    winsound = None
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class SoundAlertManager:
    def __init__(self, speech_cooldown=4.0, siren_cooldown=3.0):
        self.speech_cooldown = speech_cooldown
        self.siren_cooldown = siren_cooldown
        self._last_speech = 0.0
        self._last_siren = 0.0
        self._q = queue.Queue(maxsize=4)
        threading.Thread(target=self._run, name="audio", daemon=True).start()

    def _run(self):
        engine = None
        if pyttsx3 is not None:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                engine.setProperty("volume", 1.0)
            except Exception as e:
                print(f"[AUDIO] TTS unavailable: {e}")
        while True:
            kind, payload = self._q.get()
            try:
                if kind == "speak":
                    if engine:
                        engine.say(payload)
                        engine.runAndWait()
                    else:
                        print(f"[AUDIO] {payload}")
                elif winsound:
                    for _ in range(3):
                        winsound.Beep(2600, 120)
                        time.sleep(0.04)
                else:
                    print("\a[AUDIO] siren", flush=True)
            except Exception as e:
                print(f"[AUDIO] error: {e}")

    def speak(self, text):
        now = time.time()
        if not text or now - self._last_speech < self.speech_cooldown:
            return False
        try:
            self._q.put_nowait(("speak", text))
            self._last_speech = now
            return True
        except queue.Full:
            return False

    def siren(self):
        now = time.time()
        if now - self._last_siren < self.siren_cooldown:
            return False
        try:
            self._q.put_nowait(("siren", None))
            self._last_siren = now
            return True
        except queue.Full:
            return False