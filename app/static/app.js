const el = (id) => document.getElementById(id);

let activeTab = "bench";
let es = null;
let lastResult = null;

function setTab(tab){
  activeTab = tab;
  document.querySelectorAll(".tab").forEach(b => {
    b.classList.toggle("tab--active", b.dataset.tab === tab);
  });
  el("benchFields").classList.toggle("hidden", tab !== "bench");
  el("loadFields").classList.toggle("hidden", tab !== "load");
  el("status").textContent = "Idle.";
}

document.querySelectorAll(".tab").forEach(b => b.addEventListener("click", () => setTab(b.dataset.tab)));

function fmt(x, digits=2){
  if (x === null || x === undefined) return "N/A";
  if (Number.isNaN(x)) return "N/A";
  if (typeof x === "number") return x.toFixed(digits);
  return String(x);
}
function msFromS(s){
  if (s === null || s === undefined) return "N/A";
  if (Number.isNaN(s)) return "N/A";
  return (s * 1000).toFixed(2) + " ms";
}

function renderHeader(h){
  const lines = [];
  lines.push("┌" + "─".repeat(86) + "┐");
  lines.push("│ " + "vLLM " + (h.kind === "load" ? "Load Test" : "Benchmark") + " ".repeat(86 - ("vLLM ".length + (h.kind === "load" ? "Load Test".length : "Benchmark".length))) + "│");
  lines.push("├" + "─".repeat(86) + "┤");
  const kv = [
    ["Model", h.model],
    ["Host", h.host],
    ["API", h.api],
    ["Models", h.models],
    ["Mode", h.stream ? "stream (TTFB/TTFT enabled)" : "non-stream"],
    ...(h.kind === "load" ? [["Requests", h.requests], ["Concurrency", h.concurrency]] : [["Runs", h.runs]]),
    ["Max tokens", h.max_tokens],
    ["Temp", h.temperature],
    ["Timeout", h.timeout + "s"],
    ["TLS", h.insecure ? "insecure (skip verify)" : "verify"],
    ["Auth", h.has_token ? "bearer token" : "none"],
  ];
  kv.forEach(([k,v]) => {
    const left = (k + " : ").padEnd(12, " ");
    const txt = left + String(v);
    lines.push("│ " + txt.padEnd(86, " ") + "│");
  });
  lines.push("└" + "─".repeat(86) + "┘");
  el("headerBox").textContent = lines.join("\n");
}

function renderSummary(s, kind){
  if (!s) { el("summaryBox").textContent = ""; return; }
  const lines = [];
  lines.push("┌" + "─".repeat(86) + "┐");
  lines.push("│ " + "Summary".padEnd(86, " ") + "│");
  lines.push("├" + "─".repeat(86) + "┤");

  if (kind === "load"){
    lines.push(("│ " + (`Wall time   : ${fmt(s.wall_time_s,3)}s | Achieved RPS ${fmt(s.achieved_rps,2)}`).padEnd(86," ") + "│"));
    lines.push(("│ " + (`Results     : ${s.ok} ok | ${s.failed} failed`).padEnd(86," ") + "│"));
  }

  function lineStat(label, stat, unit){
    if (!stat) return;
    const avg = fmt(stat.avg, unit === "s" ? 3 : 2);
    const p50 = fmt(stat.p50, unit === "s" ? 3 : 2);
    const p95 = fmt(stat.p95, unit === "s" ? 3 : 2);
    lines.push(("│ " + (`${label} : avg ${avg}${unit} | p50 ${p50}${unit} | p95 ${p95}${unit}`).padEnd(86," ") + "│"));
  }

  lineStat("Total time ", s.total_s, "s");

  if (s.ttfb_s){
    lines.push(("│ " + (`TTFB       : avg ${msFromS(s.ttfb_s.avg)} | p50 ${msFromS(s.ttfb_s.p50)} | p95 ${msFromS(s.ttfb_s.p95)}`).padEnd(86," ") + "│"));
  }
  if (s.ttft_s){
    lines.push(("│ " + (`TTFT       : avg ${msFromS(s.ttft_s.avg)} | p50 ${msFromS(s.ttft_s.p50)} | p95 ${msFromS(s.ttft_s.p95)}`).padEnd(86," ") + "│"));
  }
  if (s.e2e_tps){
    lines.push(("│ " + (`Speed(e2e) : avg ${fmt(s.e2e_tps.avg,2)} | p50 ${fmt(s.e2e_tps.p50,2)} | p95 ${fmt(s.e2e_tps.p95,2)}`).padEnd(86," ") + "│"));
  }
  if (s.gen_tps){
    lines.push(("│ " + (`Speed(gen) : avg ${fmt(s.gen_tps.avg,2)} | p50 ${fmt(s.gen_tps.p50,2)} | p95 ${fmt(s.gen_tps.p95,2)}`).padEnd(86," ") + "│"));
  }
  lines.push("└" + "─".repeat(86) + "┘");
  el("summaryBox").textContent = lines.join("\n");
}

function clearTable(){
  el("tbody").innerHTML = "";
  el("headerBox").textContent = "";
  el("summaryBox").textContent = "";
  lastResult = null;
  el("download").classList.add("disabled");
  if (es){ es.close(); es = null; }
}

el("clear").addEventListener("click", clearTable);

function addRow(r){
  const tr = document.createElement("tr");

  if (r.error){
    tr.innerHTML = `
      <td>${r.i ?? "?"}</td>
      <td colspan="8" class="err">ERROR: ${r.error}</td>
    `;
    el("tbody").appendChild(tr);
    return;
  }

  const ttfb = Number.isNaN(r.ttfb_s) ? "N/A" : msFromS(r.ttfb_s);
  const ttft = Number.isNaN(r.ttft_s) ? "N/A" : msFromS(r.ttft_s);

  tr.innerHTML = `
    <td>${r.i}</td>
    <td>${fmt(r.total_s,3)}</td>
    <td class="dim">${ttfb}</td>
    <td class="cyan">${ttft}</td>
    <td>${r.pTok ?? "?"}</td>
    <td>${r.cTok ?? "?"}</td>
    <td>${r.e2e_tps !== null && r.e2e_tps !== undefined ? fmt(r.e2e_tps,2) : "N/A"}</td>
    <td>${r.gen_tps !== null && r.gen_tps !== undefined ? fmt(r.gen_tps,2) : "N/A"}</td>
    <td class="ok">${r.http ?? ""}</td>
  `;
  el("tbody").appendChild(tr);
}

async function run(){
  clearTable();
  el("status").textContent = "Starting…";

  const payload = {
    server: el("server").value.trim(),
    model: el("model").value.trim() || null,
    prompt: el("prompt").value,
    max_tokens: Number(el("max_tokens").value),
    temperature: Number(el("temperature").value),
    timeout: Number(el("timeout").value),
    stream: el("stream").checked,
    insecure: el("insecure").checked,
    token: el("token").value.trim() || null,
  };

  let url = "/api/bench";
  if (activeTab === "bench"){
    payload.runs = Number(el("runs").value);
  } else {
    url = "/api/load";
    payload.requests = Number(el("requests").value);
    payload.concurrency = Number(el("concurrency").value);
  }

  if (!payload.server){
    el("status").textContent = "Server is required.";
    return;
  }

  const res = await fetch(url, {
    method:"POST",
    headers:{ "Content-Type":"application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok){
    el("status").textContent = "Error starting job: " + (data.error || res.statusText);
    return;
  }

  const jobId = data.job_id;
  el("status").textContent = "Running… job " + jobId.slice(0,8);

  es = new EventSource(`/api/jobs/${jobId}/events`);

  es.onmessage = (evt) => {
    const ev = JSON.parse(evt.data);
    if (ev.type === "header"){
      renderHeader(ev);
    } else if (ev.type === "row"){
      addRow(ev.row);
    } else if (ev.type === "done"){
      lastResult = ev.result;
      renderSummary(ev.result.summary, ev.result.kind);
      el("status").textContent = "Done.";
      el("download").classList.remove("disabled");
      el("download").href = "data:application/json;charset=utf-8," + encodeURIComponent(JSON.stringify(ev.result, null, 2));
      es.close();
      es = null;
    } else if (ev.type === "error"){
      el("status").textContent = "Error: " + ev.error;
      es.close();
      es = null;
    }
  };

  es.onerror = () => {
    el("status").textContent = "Stream error (job may still be running).";
  };
}

el("run").addEventListener("click", run);

// default tab
setTab("bench");
