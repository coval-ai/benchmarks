export type AudioCaptureHandle = { stop: () => void; analyser: AnalyserNode };

function resampleLinear(src: Float32Array, ratio: number): Float32Array {
  const dstLen = Math.round(src.length * ratio);
  const dst = new Float32Array(dstLen);
  for (let i = 0; i < dstLen; i++) {
    const pos = i / ratio;
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, src.length - 1);
    dst[i] = src[lo]! + (src[hi]! - src[lo]!) * (pos - lo);
  }
  return dst;
}

const TARGET_SAMPLE_RATE = 16_000;

export async function startAudioCapture(
  onChunk: (chunk: ArrayBuffer) => void,
): Promise<AudioCaptureHandle> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  const ctx = new AudioContext();
  const nativeRate = ctx.sampleRate;
  const source = ctx.createMediaStreamSource(stream);

  const processor = ctx.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = (e) => {
    const float32 = e.inputBuffer.getChannelData(0);
    const samples = nativeRate === TARGET_SAMPLE_RATE
      ? float32
      : resampleLinear(float32, TARGET_SAMPLE_RATE / nativeRate);
    const int16 = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]!));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    onChunk(int16.buffer);
  };

  const sink = ctx.createGain();
  sink.gain.value = 0;
  source.connect(processor);
  processor.connect(sink);
  sink.connect(ctx.destination);

  const analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.82;
  source.connect(analyser);

  return {
    analyser,
    stop: () => {
      processor.disconnect();
      sink.disconnect();
      source.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      void ctx.close();
    },
  };
}
