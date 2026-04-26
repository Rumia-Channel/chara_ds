"use strict";

const REFRESH_MS = 1000;

const $ = (id) => document.getElementById(id);

let paused = false;
let activeTab = "actor";
let lastTimelineLen = 0;

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    activeTab = btn.dataset.tab;
    $("pane-" + activeTab).classList.add("active");
  });
});

$("pause-toggle").addEventListener("change", (e) => {
  paused = e.target.checked;
});

function fmtJSON(obj) {
  if (obj === null || obj === undefined) return "(none)";
  try {
    return JSON.stringify(obj, null, 2);
  } catch (e) {
    return String(obj);
  }
}

function clip(s, n) {
  if (s === null || s === undefined) return "";
  s = String(s);
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function setStatus(state) {
  const dot = $("status-dot");
  const text = $("status-text");
  const status = state.status || "idle";
  dot.classList.remove("running", "done", "error", "idle");
  if (status === "running") dot.classList.add("running");
  else if (status === "done") dot.classList.add("done");
  else if (status === "error") dot.classList.add("error");
  else dot.classList.add("idle");
  text.textContent = status;
}

function setProgress(state) {
  const s = state.summary || {};
  const written = s.written ?? 0;
  const total = s.total_requested ?? 0;
  const pct = total > 0 ? Math.min(100, Math.round((written / total) * 100)) : 0;
  $("progress-fill").style.width = pct + "%";
  $("progress-label").textContent = `${written} / ${total || "—"}  (${pct}%)`;

  $("meta-workers").textContent = "workers: " + (s.workers ?? "—");
  $("meta-out").textContent = "out: " + clip(s.out || "—", 40);
  $("meta-out").title = s.out || "";
  $("meta-updated").textContent = "updated: " + clip(state.updated_at || "—", 19);
}

function renderActive(state) {
  const list = $("active-list");
  const active = state.active || {};
  const ids = Object.keys(active);
  $("active-count").textContent = ids.length;
  list.innerHTML = "";
  if (ids.length === 0) {
    const d = document.createElement("div");
    d.className = "empty";
    d.textContent = "no active conversations";
    list.appendChild(d);
    return;
  }
  ids.sort();
  ids.forEach((id) => {
    const cur = active[id] || {};
    const card = document.createElement("div");
    card.className = "active-card";

    const idEl = document.createElement("div");
    idEl.className = "ac-id";
    idEl.textContent = id;
    card.appendChild(idEl);

    const row = document.createElement("div");
    row.className = "ac-row";
    const turn = cur.turn_index ?? cur.turn ?? "—";
    const stage = cur.stage || cur.phase || "—";
    const speaker = cur.speaker || cur.next_speaker || "";
    row.innerHTML =
      `<span>turn: ${turn}</span>` +
      `<span>stage: ${stage}</span>` +
      (speaker ? `<span>speaker: ${speaker}</span>` : "");
    card.appendChild(row);

    const extraText = cur.note || cur.scene_state || cur.public_event;
    if (extraText) {
      const ex = document.createElement("div");
      ex.className = "ac-extra";
      ex.textContent = clip(String(extraText), 240);
      card.appendChild(ex);
    }

    list.appendChild(card);
  });
}

function speakerClass(sp) {
  const k = String(sp || "").trim().toUpperCase();
  if (k === "A") return "speaker-a";
  if (k === "B") return "speaker-b";
  if (k === "C") return "speaker-c";
  if (k === "D") return "speaker-d";
  return "";
}

function renderTimeline(state) {
  const tl = $("timeline");
  const items = state.latest_public_timeline || [];
  $("timeline-count").textContent = items.length;

  const wasAtBottom =
    tl.scrollHeight - tl.scrollTop - tl.clientHeight < 40 || items.length !== lastTimelineLen;
  tl.innerHTML = "";

  if (items.length === 0) {
    const d = document.createElement("div");
    d.className = "empty";
    d.textContent = "timeline is empty";
    tl.appendChild(d);
    return;
  }

  items.forEach((m) => {
    const div = document.createElement("div");
    div.className = "msg " + speakerClass(m.speaker);

    const head = document.createElement("div");
    const sp = document.createElement("span");
    sp.className = "speaker";
    sp.textContent = (m.speaker || "?") + ":";
    head.appendChild(sp);

    const tx = document.createElement("span");
    tx.className = "utter";
    tx.textContent = m.utterance || m.public_utterance || "";
    head.appendChild(tx);
    div.appendChild(head);

    const action = m.visible_action || m.physical_action || m.action;
    if (action) {
      const ac = document.createElement("div");
      ac.className = "action";
      if (typeof action === "string") {
        ac.textContent = action;
      } else {
        ac.textContent = JSON.stringify(action);
      }
      div.appendChild(ac);
    }

    const metaBits = [];
    if (m.suggested_action) metaBits.push("action: " + m.suggested_action);
    if (m.pressure) {
      metaBits.push(`<span class="pressure-${m.pressure}">pressure: ${m.pressure}</span>`);
    }
    if (m.scene_state) metaBits.push("scene: " + clip(m.scene_state, 60));
    if (metaBits.length) {
      const meta = document.createElement("div");
      meta.className = "meta-row";
      meta.innerHTML = metaBits.join(" · ");
      div.appendChild(meta);
    }

    tl.appendChild(div);
  });

  if (wasAtBottom) tl.scrollTop = tl.scrollHeight;
  lastTimelineLen = items.length;
}

function renderDetail(state) {
  $("pane-actor").textContent = fmtJSON(state.last_actor);
  $("pane-controller").textContent = fmtJSON(state.last_controller);
  $("pane-persona").textContent = fmtJSON(state.last_persona);

  const eventsPane = $("pane-events");
  eventsPane.innerHTML = "";
  const events = (state.events || []).slice(-80).reverse();
  if (events.length === 0) {
    const d = document.createElement("div");
    d.className = "empty";
    d.textContent = "no events yet";
    eventsPane.appendChild(d);
  } else {
    events.forEach((ev) => {
      const item = document.createElement("div");
      item.className = "event-item";
      const head = document.createElement("div");
      const t = document.createElement("span");
      t.className = "ev-time";
      t.textContent = clip(ev.time || "", 19);
      const n = document.createElement("span");
      n.className = "ev-name";
      n.textContent = ev.event || "(event)";
      head.appendChild(t);
      head.appendChild(n);
      item.appendChild(head);

      const detail = { ...ev };
      delete detail.time;
      delete detail.event;
      if (Object.keys(detail).length) {
        const pre = document.createElement("pre");
        pre.style.margin = "4px 0 0";
        pre.style.fontSize = "10.5px";
        pre.style.color = "var(--fg-dim)";
        pre.style.whiteSpace = "pre-wrap";
        pre.textContent = fmtJSON(detail);
        item.appendChild(pre);
      }
      eventsPane.appendChild(item);
    });
  }

  const errorsPane = $("pane-errors");
  errorsPane.innerHTML = "";
  const errs = (state.errors || []).slice(-50).reverse();
  const badge = $("errors-badge");
  badge.textContent = (state.errors || []).length;
  badge.classList.toggle("zero", (state.errors || []).length === 0);

  if (errs.length === 0) {
    const d = document.createElement("div");
    d.className = "empty";
    d.textContent = "no errors";
    errorsPane.appendChild(d);
  } else {
    errs.forEach((er) => {
      const item = document.createElement("div");
      item.className = "error-item";
      const head = document.createElement("div");
      const t = document.createElement("span");
      t.className = "er-time";
      t.textContent = clip(er.created_at || er.time || "", 19);
      const ty = document.createElement("span");
      ty.className = "er-type";
      ty.textContent = er.error_type || er.type || "Error";
      head.appendChild(t);
      head.appendChild(ty);
      item.appendChild(head);

      const msg = document.createElement("div");
      msg.className = "er-msg";
      msg.textContent = er.error || er.message || "";
      item.appendChild(msg);

      if (er.context) {
        const c = document.createElement("div");
        c.className = "er-ctx";
        c.textContent = "context: " + fmtJSON(er.context);
        item.appendChild(c);
      }

      errorsPane.appendChild(item);
    });
  }
}

async function refresh() {
  if (paused) return;
  try {
    const res = await fetch("/state?t=" + Date.now());
    if (!res.ok) return;
    const state = await res.json();
    setStatus(state);
    setProgress(state);
    renderActive(state);
    renderTimeline(state);
    renderDetail(state);
  } catch (e) {
    // ignore network blips
  }
}

setInterval(refresh, REFRESH_MS);
refresh();
