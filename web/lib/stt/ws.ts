import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const WebSocket = require("ws") as typeof import("ws").default;

export default WebSocket;
