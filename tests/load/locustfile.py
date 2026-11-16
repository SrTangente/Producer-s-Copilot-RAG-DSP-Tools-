"""Load test: `locust -f tests/load/locustfile.py --host https://<your-app>`."""
import io

import numpy as np
import soundfile as sf
from locust import HttpUser, between, task

SR = 44100
_t = np.arange(SR * 10) / SR
_buf = io.BytesIO()
sf.write(_buf, (0.3 * np.sin(2 * np.pi * 110 * _t)).astype(np.float32), SR, format="WAV")
CLIP = _buf.getvalue()


class ProducerUser(HttpUser):
    wait_time = between(1, 3)

    @task(5)
    def health(self):
        self.client.get("/health")

    @task(2)
    def analyze(self):
        self.client.post("/analyze", files={"file": ("clip.wav", CLIP, "audio/wav")})

    @task(1)
    def tips_rules_only(self):
        # use_llm=false keeps load tests from spending API credits
        self.client.post("/tips", files={"file": ("clip.wav", CLIP, "audio/wav")},
                         data={"target_id": "streaming", "use_llm": "false"})
