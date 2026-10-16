import pyaudio
import threading
import webrtcvad

class VADDetector:

    def __init__(self):
        self.vad = webrtcvad.Vad(2)
        self.voice_detected = False
        self.running = True

    def detect_voice(self):

        p = pyaudio.PyAudio()

        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=320)

        while self.running:

            frame = stream.read(320, exception_on_overflow=False)

            if self.vad.is_speech(frame, 16000):
                self.voice_detected = True
            else:
                self.voice_detected = False

    def start(self):

        thread = threading.Thread(target=self.detect_voice)
        thread.daemon = True
        thread.start()

        print("Voice detection started...")