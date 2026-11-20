const state = {
  latest: null,
  filter: "all",
  sortMode: "status_p",
  lastTickMs: 0,
};

const pollMs = 300;
const staleMs = 2000;

function getEl(id) {
  return document.getElementById(id);
}

function fmt(n, digits = 0) {
  const value = Number.isFinite(n) ? n : 0;
  return value.toFixed(digits);
}

function esc(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function setConnectionState(kind, text) {
  const dot = getEl("conn-dot");
  dot.classList.remove("dot-live", "dot-error", "dot-stale");
  dot.classList.add(`dot-${kind}`);
  getEl("conn-text").textContent = text;
}

function applySummary(s) {
  getEl("m-people").textContent = s.people_count ?? 0;
  getEl("m-emergency").textContent = s.emergency_count ?? 0;
  getEl("m-uncertain").textContent = s.uncertain_count ?? 0;
  getEl("m-fps").textContent = 0; //fmt(s.fps ?? 0, 1);
  getEl("m-latency").textContent = `${fmt(s.latency_ms ?? 0, 1)} ms`;
  getEl("m-reconnects").textContent = s.reconnects ?? 0;
  getEl("m-last-update").textContent = new Date().toLocaleTimeString();

  if (s.error) {
    setConnectionState("error", "Source error");
    getEl("stream-state").textContent = "error";
    return;
  }

  const ageMs = Date.now() - state.lastTickMs;
  if (ageMs > staleMs) {
    setConnectionState("stale", "Stale data");
    getEl("stream-state").textContent = "stale";
  } else {
    setConnectionState("live", "Live");
    getEl("stream-state").textContent = "healthy";
  }
}

function sortPeople(people) {
  const sorted = [...people];
  if (state.sortMode === "track_id") {
    sorted.sort((a, b) => (a.track_id ?? 0) - (b.track_id ?? 0));
    return sorted;
  }
  if (state.sortMode === "det_conf") {
    sorted.sort((a, b) => (b.det_conf ?? 0) - (a.det_conf ?? 0));
    return sorted;
  }
  sorted.sort((a, b) => (b.status_p ?? 0) - (a.status_p ?? 0));
  return sorted;
}

function filterPeople(people) {
  if (state.filter === "all") return people;
  return people.filter((p) => (p.status || "unknown") === state.filter);
}

function renderPeople() {
  const box = getEl("people-list");
  if (!state.latest) {
    box.innerHTML = '<div class="state">Loading tracks...</div>';
    return;
  }
  if (state.latest.error) {
    box.innerHTML = `<div class="state">Source error: ${esc(state.latest.error)}</div>`;
    return;
  }
  const peopleRaw = Array.isArray(state.latest.people) ? state.latest.people : [];
  const people = sortPeople(filterPeople(peopleRaw));
  if (!people.length) {
    box.innerHTML = '<div class="state">No tracks for this filter.</div>';
    return;
  }
  box.innerHTML = people
    .map((p) => {
      const status = p.status || "unknown";
      const cls = `sev-${status}`;
      return `<article class="person">
        <div class="person-top">
          <div><strong>ID ${p.track_id}</strong></div>
          <div class="${cls}"><strong>${esc(status)}</strong></div>
        </div>
        <div>Status confidence: <strong>${fmt(p.status_p ?? 0, 2)}</strong></div>
        <div>Posture: <strong>${esc(p.posture || "unknown")}</strong> (${fmt(p.posture_p ?? 0, 2)})</div>
        <div class="muted">Det ${fmt(p.det_conf ?? 0, 2)} | Kpt ${fmt(p.avg_kpt_conf ?? 0, 2)}</div>
      </article>`;
    })
    .join("");
}

async function refreshStatus() {
  try {
    const response = await fetch("/status", { cache: "no-store" });
    if (!response.ok) {
      setConnectionState("error", "Fetch error");
      getEl("stream-state").textContent = "error";
      return;
    }
    const payload = await response.json();
    state.lastTickMs = Date.now();
    state.latest = payload;
    applySummary(payload);
    renderPeople();
  } catch (_) {
    setConnectionState("error", "Fetch error");
    getEl("stream-state").textContent = "error";
  }
}

function wireControls() {
  const filterButtons = Array.from(document.querySelectorAll("[data-filter]"));
  filterButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      state.filter = btn.getAttribute("data-filter") || "all";
      filterButtons.forEach((x) => x.classList.remove("is-active"));
      btn.classList.add("is-active");
      renderPeople();
    });
  });
  getEl("sort-mode").addEventListener("change", (event) => {
    state.sortMode = event.target.value;
    renderPeople();
  });

  const stopBtn = getEl("stop-server-btn");
  if (stopBtn) {
    stopBtn.addEventListener("click", async () => {
      stopBtn.disabled = true;
      stopBtn.textContent = "Stopping...";
      try {
        await fetch("/shutdown", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        });
      } catch (_) {
        // Server may close before response is returned.
      }
      setConnectionState("stale", "Server stopped");
      getEl("stream-state").textContent = "stopped";
    });
  }
}

wireControls();
setConnectionState("stale", "Connecting...");
setInterval(refreshStatus, pollMs);
refreshStatus();
