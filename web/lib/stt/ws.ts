import { createRequire } from "node:module";

process.env.WS_NO_BUFFER_UTIL ??= "1";
process.env.WS_NO_UTF_8_VALIDATE ??= "1";

const require = createRequire(import.meta.url);
const WebSocket = require("ws") as typeof import("ws").default;

export default WebSocket;
