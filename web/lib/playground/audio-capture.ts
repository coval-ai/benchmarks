export type AudioCaptureHandle = { stop: () => void };

export async function startAudioCapture(
  onChunk: (chunk: ArrayBuffer) => void,
): Promise<AudioCaptureHandle> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  const ctx = new AudioContext({ sampleRate: 16_000 });
  if (ctx.sampleRate !== 16_000) {
    stream.getTracks().forEach((t) => t.stop());
    await ctx.close();
    throw new Error(`Unsupported capture sample rate: ${ctx.sampleRate}. Expected 16000.`);
  }
  const source = ctx.createMediaStreamSource(stream);

  const processor = ctx.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = (e) => {
    const float32 = e.inputBuffer.getChannelData(0);
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]!));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    onChunk(int16.buffer);
  };

  const sink = ctx.createGain();
  sink.gain.value = 0;
  source.connect(processor);
  processor.connect(sink);
  sink.connect(ctx.destination);

  return {
    stop: () => {
      processor.disconnect();
      sink.disconnect();
      source.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      void ctx.close();
    },
  };
}
