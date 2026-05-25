/** Wraps raw PCM s16le mono into a WAV container. Required by ElevenLabs REST. */
export function pcmToWav(pcm: ArrayBuffer, sampleRate = 16_000): ArrayBuffer {
  const dataLen = pcm.byteLength;
  const header = new ArrayBuffer(44);
  const v = new DataView(header);
  const str = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i));
  };

  str(0, "RIFF");
  v.setUint32(4, 36 + dataLen, true);
  str(8, "WAVE");
  str(12, "fmt ");
  v.setUint32(16, 16, true);              // subchunk size
  v.setUint16(20, 1, true);              // PCM format
  v.setUint16(22, 1, true);              // mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true); // byte rate (16-bit mono)
  v.setUint16(32, 2, true);              // block align
  v.setUint16(34, 16, true);             // bits per sample
  str(36, "data");
  v.setUint32(40, dataLen, true);

  const out = new Uint8Array(44 + dataLen);
  out.set(new Uint8Array(header), 0);
  out.set(new Uint8Array(pcm), 44);
  return out.buffer;
}
