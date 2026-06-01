export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    process.env.WS_NO_BUFFER_UTIL ??= "1";
    process.env.WS_NO_UTF_8_VALIDATE ??= "1";
  }
}
