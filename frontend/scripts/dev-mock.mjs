// `npm run dev:mock`: the mock API on :8787 plus `next dev` pointed at it.
import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { startMock } from "./mock-api.mjs";

const port = Number(process.env.MOCK_PORT ?? "8787");
const server = await startMock(port);

const require = createRequire(import.meta.url);
const nextBin = require.resolve("next/dist/bin/next");
const child = spawn(process.execPath, [nextBin, "dev", ...process.argv.slice(2)], {
  stdio: "inherit",
  env: { ...process.env, NEXT_PUBLIC_API_URL: `http://localhost:${port}` },
});

const stop = () => {
  child.kill();
  server.close();
};
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
child.on("exit", (code) => {
  server.close();
  process.exit(code ?? 0);
});
