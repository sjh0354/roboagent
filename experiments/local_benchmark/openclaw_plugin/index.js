const endpoint = () => {
  const value = process.env.LOCAL_BENCHMARK_HTTP_URL;
  if (!value) throw new Error("LOCAL_BENCHMARK_HTTP_URL is required");
  return value.replace(/\/$/, "");
};

async function invoke(route, payload) {
  const response = await fetch(`${endpoint()}/${route}`, {
    method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(payload || {}),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(JSON.stringify(result));
  const content = [{type: "text", text: JSON.stringify(result)}];
  const observation = result.observation || result;
  for (const path of (observation.rgb_paths || []).slice(0, 2)) {
    const {readFile} = await import("node:fs/promises");
    content.push({type: "image", data: (await readFile(path)).toString("base64"), mimeType: "image/png"});
  }
  return {content};
}

const object = (properties, required = []) => ({type: "object", additionalProperties: false, properties, required});
const string = {type: "string"};

export default {
  id: "local-benchmark",
  name: "RoboAgent Local Benchmark",
  description: "Stateful simulated physical tools",
  register(api) {
    api.registerTool({name: "local_benchmark_observe", description: "Observe public state and current RGB views. Hidden outcomes and private scoring are never returned.", parameters: object({}), execute: (_id, params) => invoke("observe", params)});
    api.registerTool({name: "local_benchmark_start_skill", description: "Resolve one pick, place, navigate, or pick_and_place goal atomically using the assigned aggregate probability; inspect returned state/images. Monitoring and recovery are already integrated, so do not repeat the same goal.", parameters: object({skill: string, arguments: {type: "object", additionalProperties: true}}, ["skill", "arguments"]), execute: (_id, params) => invoke("start_skill", params)});
    api.registerTool({name: "local_benchmark_set_device", description: "Set a simulated device state; never touches a real account or device.", parameters: object({device: string, state: string}, ["device", "state"]), execute: (_id, params) => invoke("set_device", params)});
    api.registerTool({name: "local_benchmark_remember", description: "Store a user-stated preference inside this isolated simulated session.", parameters: object({text: string}, ["text"]), execute: (_id, params) => invoke("remember", params)});
  },
};
