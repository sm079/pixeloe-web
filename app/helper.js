// One thread of worker.js's pool: runs a single pixeloe.js task per message.
//
// in:  { id, task, args }     out: { id, result } or { id, error }

import { TASKS } from "./pixeloe.js";

onmessage = ({ data: { id, task, args } }) => {
  try {
    const result = TASKS[task](...args);
    const buf = (result.data ?? result).buffer;
    postMessage({ id, result }, buf ? [buf] : []);
  } catch (err) {
    postMessage({ id, error: String(err?.message ?? err) });
  }
};
