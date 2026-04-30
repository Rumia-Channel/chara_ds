"use strict";

const REFRESH_MS = 1000;

const $ = (id) => document.getElementById(id);

let uiPaused = false;
let activeTab = "actor";
let lastTimelineLen = 0;
let selectedConversation = "__latest__";
let lastSelectedConversation = "__latest__";
const openAgentDetails = new Set();

// Completed-tab state
let completedOffset = 0;
let completedLimit = 50;
let completedOrder = "desc";
let completedTotal = 0;
let completedExpandedId = null;

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    activeTab = btn.dataset.tab;
    $("pane-" + activeTab).classList.add("active");
    if (activeTab === "completed") {
      loadCompleted();
    } else if (activeTab === "queue") {
      loadQueue();
    }
  });
});

$("ui-pause-toggle").addEventListener("change", (e) => {
  uiPaused = e.target.checked;
});

$("timeline-select").addEventListener("change", (e) => {
  selectedConversation = e.target.value || "__latest__";
  lastTimelineLen = 0;
  refresh();
});

async function postControl(action) {
  try {
    const res = await fetch("/control/" + action, { method: "POST" });
    if (!res.ok) {
      const txt = await res.text();
      alert("control " + action + " failed: " + txt);
    }
  } catch (e) {
    alert("control " + action + " failed: " + e);
  }
  refresh();
}

$("btn-pause").addEventListener("click", () => postControl("pause"));
$("btn-resume").addEventListener("click", () => postControl("resume"));
$("btn-stop").addEventListener("click", () => {
  if (confirm("Stop after currently-running conversations finish? This cannot be undone in this session.")) {
    postControl("stop");
  }
});

$("completed-refresh").addEventListener("click", () => loadCompleted());
$("completed-prev").addEventListener("click", () => {
  completedOffset = Math.max(0, completedOffset - completedLimit);
  loadCompleted();
});
$("completed-next").addEventListener("click", () => {
  if (completedOffset + completedLimit < completedTotal) {
    completedOffset += completedLimit;
    loadCompleted();
  }
});
$("completed-order").addEventListener("change", (e) => {
  completedOrder = e.target.value;
  completedOffset = 0;
  loadCompleted();
});
$("completed-limit").addEventListener("change", (e) => {
  completedLimit = parseInt(e.target.value, 10) || 50;
  completedOffset = 0;
  loadCompleted();
});

// ---- Queue tab ----
let queueHideConsumed = false;
$("queue-refresh").addEventListener("click", () => loadQueue());
$("queue-hide-consumed").addEventListener("change", (e) => {
  queueHideConsumed = e.target.checked;
  loadQueue();
});
$("queue-add-btn").addEventListener("click", async () => {
  const ta = $("queue-add-text");
  const text = (ta.value || "").trim();
  if (!text) return;
  try {
    const res = await fetch("/situations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      const t = await res.text();
      alert("add failed: " + t);
      return;
    }
    ta.value = "";
    loadQueue();
  } catch (e) {
    alert("add failed: " + e);
  }
});

async function loadQueue() {
  const list = $("queue-list");
  const info = $("queue-info");
  info.textContent = "loading…";
  try {
    const res = await fetch("/situations");
    if (!res.ok) {
      info.textContent = "error: " + res.status;
      list.innerHTML = "";
      return;
    }
    const payload = await res.json();
    const items = payload.items || [];
    const total = items.length;
    const generated = items.filter((it) => it.origin === "generated").length;
    const consumed = items.filter((it) => it.likely_consumed).length;
    $("queue-badge").textContent = total - consumed;
    info.textContent =
      `${total} total · ${generated} flash-generated · ${consumed} likely picked` +
      (payload.format_path ? "  ·  " + payload.format_path : "");
    list.innerHTML = "";
    const visible = queueHideConsumed
      ? items.filter((it) => !it.likely_consumed)
      : items;
    if (visible.length === 0) {
      const d = document.createElement("div");
      d.className = "empty";
      d.textContent = "queue is empty";
      list.appendChild(d);
      return;
    }
    visible.forEach((it) => {
      const card = document.createElement("div");
      card.className = "queue-card" + (it.likely_consumed ? " consumed" : "")
        + (it.origin === "generated" ? " generated" : " initial");
      const head = document.createElement("div");
      head.className = "qc-head";
      head.innerHTML =
        `<span class="qc-ln">#${it.line_number}</span>` +
        `<span class="qc-origin qc-origin-${it.origin}">${it.origin}</span>` +
        (it.likely_consumed ? `<span class="qc-status">picked</span>` : `<span class="qc-status qc-pending">pending</span>`);
      card.appendChild(head);

      const body = document.createElement("div");
      body.className = "qc-body";
      const view = document.createElement("div");
      view.className = "qc-text";
      view.textContent = it.text;
      body.appendChild(view);
      card.appendChild(body);

      const actions = document.createElement("div");
      actions.className = "qc-actions";
      const editBtn = document.createElement("button");
      editBtn.className = "ctrl-btn";
      editBtn.textContent = "edit";
      editBtn.addEventListener("click", () => beginEditQueueItem(card, it));
      actions.appendChild(editBtn);
      if (it.likely_consumed) {
        const note = document.createElement("span");
        note.className = "qc-warn";
        note.textContent = "note: edits to picked items only affect future picks";
        actions.appendChild(note);
      }
      card.appendChild(actions);

      list.appendChild(card);
    });
  } catch (e) {
    info.textContent = "error: " + e;
  }
}

function beginEditQueueItem(card, it) {
  const body = card.querySelector(".qc-body");
  body.innerHTML = "";
  const ta = document.createElement("textarea");
  ta.className = "qc-edit";
  ta.rows = 3;
  ta.value = it.text;
  body.appendChild(ta);

  const actions = card.querySelector(".qc-actions");
  actions.innerHTML = "";
  const save = document.createElement("button");
  save.className = "ctrl-btn";
  save.textContent = "save";
  save.addEventListener("click", async () => {
    const newText = (ta.value || "").trim();
    if (!newText) {
      alert("text required");
      return;
    }
    try {
      const res = await fetch("/situations/" + it.line_number, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: newText }),
      });
      if (!res.ok) {
        // Fallback for environments where PATCH is blocked.
        const fb = await fetch("/situations/" + it.line_number + "/edit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: newText }),
        });
        if (!fb.ok) {
          const t = await fb.text();
          alert("save failed: " + t);
          return;
        }
      }
      loadQueue();
    } catch (e) {
      alert("save failed: " + e);
    }
  });
  actions.appendChild(save);

  const cancel = document.createElement("button");
  cancel.className = "ctrl-btn";
  cancel.textContent = "cancel";
  cancel.addEventListener("click", () => loadQueue());
  actions.appendChild(cancel);
}

function fmtJSON(obj) {
  if (obj === null || obj === undefined) return "(none)";
  try {
    return JSON.stringify(obj, null, 2);
  } catch (e) {
    return String(obj);
  }
}

function agentKeyForTab(tab) {
  if (tab === "guard") return "actor_guard";
  if (tab === "grand") return "grand_controller";
  if (tab === "audit") return "conversation_audit";
  return tab;
}

function lastKeyForAgent(agent) {
  if (agent === "actor_guard") return "last_actor_guard";
  if (agent === "grand_controller") return "last_grand_controller";
  if (agent === "conversation_audit") return "last_conversation_audit";
  return "last_" + agent;
}

function collectConversationIdsWithHistory(state) {
  const active = state.active || {};
  return Object.keys(active)
    .filter((id) => {
      const h = active[id] && active[id].agent_history;
      return h && typeof h === "object";
    })
    .sort();
}

function entryLabel(entry, idx) {
  const bits = [];
  if (entry.turn_index !== undefined && entry.turn_index !== null) bits.push("turn " + entry.turn_index);
  if (entry.speaker) bits.push("speaker " + entry.speaker);
  if (entry.guard_round !== undefined && entry.guard_round !== null) bits.push("guard " + entry.guard_round);
  if (entry.stage) bits.push(entry.stage);
  if (bits.length === 0) bits.push("#" + (idx + 1));
  if (entry.time) bits.push(clip(entry.time, 19));
  return bits.join(" · ");
}

function renderAgentPane(state, tab, paneId) {
  const pane = $(paneId);
  pane.innerHTML = "";
  const agent = agentKeyForTab(tab);
  const active = state.active || {};
  const ids = collectConversationIdsWithHistory(state);
  let rendered = 0;

  ids.forEach((id) => {
    const history = ((active[id] || {}).agent_history || {})[agent] || [];
    if (!Array.isArray(history) || history.length === 0) return;
    rendered += history.length;

    const section = document.createElement("section");
    section.className = "agent-conversation";

    const head = document.createElement("div");
    head.className = "agent-conversation-head";
    const title = document.createElement("span");
    title.className = "agent-cid";
    title.textContent = id;
    const count = document.createElement("span");
    count.className = "agent-count";
    count.textContent = history.length + " entries";
    head.appendChild(title);
    head.appendChild(count);
    section.appendChild(head);

    history.forEach((entry, idx) => {
      const detailKey = `${id}:${agent}:${idx}`;
      const latest = idx === history.length - 1;
      const d = document.createElement("details");
      d.className = "agent-entry" + (latest ? " latest" : "");
      d.open = latest || openAgentDetails.has(detailKey);
      d.addEventListener("toggle", () => {
        if (latest) return;
        if (d.open) openAgentDetails.add(detailKey);
        else openAgentDetails.delete(detailKey);
      });

      const summary = document.createElement("summary");
      summary.textContent = (latest ? "latest · " : "") + entryLabel(entry, idx);
      d.appendChild(summary);

      const pre = document.createElement("pre");
      pre.textContent = fmtJSON(entry.content ?? entry);
      d.appendChild(pre);
      section.appendChild(d);
    });
    pane.appendChild(section);
  });

  if (rendered === 0) {
    const fallback = state[lastKeyForAgent(agent)];
    if (fallback !== undefined && fallback !== null) {
      const section = document.createElement("section");
      section.className = "agent-conversation";

      const head = document.createElement("div");
      head.className = "agent-conversation-head";
      const title = document.createElement("span");
      title.className = "agent-cid";
      title.textContent = "latest " + tab;
      const count = document.createElement("span");
      count.className = "agent-count";
      count.textContent = "fallback";
      head.appendChild(title);
      head.appendChild(count);
      section.appendChild(head);

      const d = document.createElement("details");
      d.className = "agent-entry latest";
      d.open = true;
      const summary = document.createElement("summary");
      summary.textContent = "latest global " + tab;
      d.appendChild(summary);
      const pre = document.createElement("pre");
      pre.textContent = fmtJSON(fallback);
      d.appendChild(pre);
      section.appendChild(d);
      pane.appendChild(section);
    } else {
      const d = document.createElement("div");
      d.className = "empty";
      d.textContent = "no " + tab + " history yet";
      pane.appendChild(d);
    }
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

function setControl(state) {
  const ctrl = state.control || {};
  const paused = !!ctrl.paused;
  const stopped = !!ctrl.stop_requested;
  $("btn-pause").hidden = paused || stopped;
  $("btn-resume").hidden = !paused || stopped;
  $("btn-stop").disabled = stopped;
  const badge = $("control-badge");
  if (stopped) {
    badge.textContent = "STOPPING";
    badge.className = "control-badge ctrl-stopping";
  } else if (paused) {
    badge.textContent = "PAUSED";
    badge.className = "control-badge ctrl-paused";
  } else {
    badge.textContent = "";
    badge.className = "control-badge";
  }
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

  // Completed-tab badge mirrors written count, which == lines in out file.
  $("completed-badge").textContent = written;
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
    if (id === selectedConversation) card.classList.add("selected");
    card.addEventListener("click", () => {
      selectedConversation = (selectedConversation === id) ? "__latest__" : id;
      $("timeline-select").value = selectedConversation;
      lastTimelineLen = 0;
      refresh();
    });

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

function syncTimelineSelector(state) {
  const sel = $("timeline-select");
  const active = state.active || {};
  const ids = Object.keys(active).sort();

  const desiredOptions = ["__latest__", ...ids];
  const currentOptions = Array.from(sel.options).map((o) => o.value);
  const same =
    desiredOptions.length === currentOptions.length &&
    desiredOptions.every((v, i) => v === currentOptions[i]);

  if (!same) {
    sel.innerHTML = "";
    const optLatest = document.createElement("option");
    optLatest.value = "__latest__";
    optLatest.textContent = "Latest update";
    sel.appendChild(optLatest);
    ids.forEach((id) => {
      const o = document.createElement("option");
      o.value = id;
      const short = id.length > 28 ? "…" + id.slice(-26) : id;
      o.textContent = short;
      sel.appendChild(o);
    });
  }

  if (selectedConversation !== "__latest__" && !(selectedConversation in active)) {
    selectedConversation = "__latest__";
  }
  if (sel.value !== selectedConversation) sel.value = selectedConversation;
}

function speakerClass(sp) {
  const k = String(sp || "").trim().toUpperCase();
  if (k === "A") return "speaker-a";
  if (k === "B") return "speaker-b";
  if (k === "C") return "speaker-c";
  if (k === "D") return "speaker-d";
  return "";
}

function renderTimelineMessages(items, container) {
  container.innerHTML = "";
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

    container.appendChild(div);
  });
}

function renderTimeline(state) {
  const tl = $("timeline");
  let items;
  let sourceLabel;
  if (selectedConversation === "__latest__") {
    items = state.latest_public_timeline || [];
    sourceLabel = state.latest_public_timeline_id
      ? "latest (" + state.latest_public_timeline_id + ")"
      : "latest";
  } else {
    const slot = (state.active || {})[selectedConversation];
    items = (slot && slot.public_timeline) || [];
    sourceLabel = selectedConversation;
  }
  $("timeline-count").textContent = items.length;

  const switched = selectedConversation !== lastSelectedConversation;
  const wasAtBottom =
    switched ||
    tl.scrollHeight - tl.scrollTop - tl.clientHeight < 40 ||
    items.length !== lastTimelineLen;

  if (items.length === 0) {
    tl.innerHTML = "";
    const d = document.createElement("div");
    d.className = "empty";
    d.textContent = "timeline is empty (" + sourceLabel + ")";
    tl.appendChild(d);
    lastSelectedConversation = selectedConversation;
    lastTimelineLen = 0;
    return;
  }

  renderTimelineMessages(items, tl);

  if (wasAtBottom) tl.scrollTop = tl.scrollHeight;
  lastTimelineLen = items.length;
  lastSelectedConversation = selectedConversation;
}

function renderDetail(state) {
  renderAgentPane(state, "actor", "pane-actor");
  renderAgentPane(state, "guard", "pane-guard");
  renderAgentPane(state, "audit", "pane-audit");
  renderAgentPane(state, "grand", "pane-grand");
  renderAgentPane(state, "controller", "pane-controller");
  renderAgentPane(state, "persona", "pane-persona");

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

async function loadCompleted() {
  const list = $("completed-list");
  const info = $("completed-info");
  info.textContent = "loading…";
  try {
    const url = `/completed?limit=${completedLimit}&offset=${completedOffset}&order=${completedOrder}`;
    const res = await fetch(url);
    if (!res.ok) {
      info.textContent = "error: " + res.status;
      list.innerHTML = "";
      return;
    }
    const payload = await res.json();
    completedTotal = payload.total || 0;
    info.textContent =
      `${payload.total || 0} total · showing ${completedOffset + 1}-${Math.min(completedTotal, completedOffset + (payload.items || []).length)}` +
      (payload.out ? "  ·  " + payload.out : "");
    list.innerHTML = "";
    const items = payload.items || [];
    if (items.length === 0) {
      const d = document.createElement("div");
      d.className = "empty";
      d.textContent = payload.exists === false
        ? "output file does not exist yet"
        : "no completed conversations in range";
      list.appendChild(d);
    } else {
      items.forEach((it) => {
        const card = document.createElement("div");
        card.className = "completed-card";
        const head = document.createElement("div");
        head.className = "cc-head";
        const idEl = document.createElement("span");
        idEl.className = "cc-id";
        idEl.textContent = it.conversation_id || "?";
        const turnsEl = document.createElement("span");
        turnsEl.className = "cc-turns";
        turnsEl.textContent = (it.turns || 0) + " turns";
        head.appendChild(idEl);
        head.appendChild(turnsEl);
        if (it.created_at) {
          const t = document.createElement("span");
          t.className = "cc-time";
          t.textContent = clip(it.created_at, 19);
          head.appendChild(t);
        }
        card.appendChild(head);
        if (it.personas && (it.personas.A || it.personas.B)) {
          const p = document.createElement("div");
          p.className = "cc-personas";
          p.textContent = `A: ${it.personas.A || "?"}  ·  B: ${it.personas.B || "?"}`;
          card.appendChild(p);
        }
        if (it.situation) {
          const s = document.createElement("div");
          s.className = "cc-situation";
          s.textContent = it.situation;
          card.appendChild(s);
        }
        if (it.last_utterance) {
          const u = document.createElement("div");
          u.className = "cc-last";
          u.textContent = "… " + it.last_utterance;
          card.appendChild(u);
        }
        card.addEventListener("click", () => openCompleted(it.conversation_id));
        list.appendChild(card);
      });
    }
  } catch (e) {
    info.textContent = "error: " + e;
  }
}

async function openCompleted(cid) {
  const detail = $("completed-detail");
  detail.hidden = false;
  detail.innerHTML = "<div class='empty'>loading " + cid + "…</div>";
  try {
    // Re-fetch a small window with full record to find the requested cid.
    // For simplicity we fetch the entire current page with include_record=1.
    const url = `/completed?limit=${completedLimit}&offset=${completedOffset}&order=${completedOrder}&include_record=1`;
    const res = await fetch(url);
    if (!res.ok) {
      detail.innerHTML = "<div class='empty'>fetch failed: " + res.status + "</div>";
      return;
    }
    const payload = await res.json();
    const found = (payload.items || []).find((it) => it.conversation_id === cid);
    if (!found || !found.record) {
      detail.innerHTML = "<div class='empty'>record not found in current page</div>";
      return;
    }
    const rec = found.record;
    detail.innerHTML = "";
    const header = document.createElement("div");
    header.className = "cc-detail-head";
    header.innerHTML =
      `<button class="ctrl-btn" id="cc-detail-close">× close</button>` +
      `<span class="cc-id">${rec.conversation_id || cid}</span>`;
    detail.appendChild(header);
    $("cc-detail-close").addEventListener("click", () => {
      detail.hidden = true;
      detail.innerHTML = "";
    });

    const sit = rec.situation || rec.scene;
    if (sit) {
      const s = document.createElement("div");
      s.className = "cc-situation";
      s.textContent = typeof sit === "string" ? sit : JSON.stringify(sit);
      detail.appendChild(s);
    }

    const tlBox = document.createElement("div");
    tlBox.className = "completed-timeline";
    detail.appendChild(tlBox);
    const items = rec.public_timeline || rec.timeline || rec.dialogue || [];
    if (Array.isArray(items) && items.length) {
      renderTimelineMessages(items, tlBox);
    } else {
      tlBox.innerHTML = "<div class='empty'>no public_timeline in record</div>";
    }

    const guardTurns = Array.isArray(rec.turns)
      ? rec.turns.filter((turn) => turn && turn.actor_guard)
      : [];
    if (guardTurns.length) {
      const guardBox = document.createElement("details");
      guardBox.className = "cc-dump";
      const sm = document.createElement("summary");
      sm.textContent = "actor guards · " + guardTurns.length + " turns";
      guardBox.appendChild(sm);
      const pre = document.createElement("pre");
      pre.textContent = fmtJSON(guardTurns.map((turn) => ({
        turn: turn.turn,
        speaker: turn.actor && turn.actor.speaker,
        provider: turn.actor_guard.provider,
        model: turn.actor_guard.model,
        content: turn.actor_guard.content,
      })));
      guardBox.appendChild(pre);
      detail.appendChild(guardBox);
    }

    if (rec.conversation_audit) {
      const audit = document.createElement("details");
      audit.className = "cc-dump";
      audit.open = true;
      const sm = document.createElement("summary");
      const content = rec.conversation_audit.content || {};
      const score = content.overall_score !== undefined ? `score ${content.overall_score}` : "audit";
      const action = content.recommended_action ? ` · ${content.recommended_action}` : "";
      sm.textContent = "conversation audit · " + score + action;
      audit.appendChild(sm);
      const pre = document.createElement("pre");
      pre.textContent = fmtJSON(rec.conversation_audit);
      audit.appendChild(pre);
      detail.appendChild(audit);
    }

    const dump = document.createElement("details");
    dump.className = "cc-dump";
    const sm = document.createElement("summary");
    sm.textContent = "raw record";
    dump.appendChild(sm);
    const pre = document.createElement("pre");
    pre.textContent = fmtJSON(rec);
    dump.appendChild(pre);
    detail.appendChild(dump);
  } catch (e) {
    detail.innerHTML = "<div class='empty'>error: " + e + "</div>";
  }
}

async function refresh() {
  if (uiPaused) return;
  try {
    const res = await fetch("/state?t=" + Date.now());
    if (!res.ok) return;
    const state = await res.json();
    setStatus(state);
    setControl(state);
    setProgress(state);
    syncTimelineSelector(state);
    renderActive(state);
    renderTimeline(state);
    renderDetail(state);
  } catch (e) {
    // ignore network blips
  }
}

setInterval(refresh, REFRESH_MS);
refresh();
