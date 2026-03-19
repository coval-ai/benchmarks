class TTS_Benchmark:
    def __init__(self, config):
        self.model = config.get("model")
        self.voice = config.get("voice")
    async def calculateTTFA(self, text):
        raise NotImplementedError("Subclasses must implement calculateTTFA method.")