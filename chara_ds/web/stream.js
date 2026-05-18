"use strict";

// SSE streaming live display.
// Depends on variables defined in app.js:
//   sseLastId, sseConnection, liveStreams, liveStreamEl, liveStreamBody,
//   liveStreamKey, lastKnownEntryCount, sseRenderPending,
//   timelineMode, agentStreamAgent, selectedConversation,
//   $(), refresh().

function connectSSE() {
  if (sseConnection) return;
  sseConnection = new EventSource("/stream?last_id=" + sseLastId);
  sseConnection.onmessage = function (e) {
    try {
      const ev = JSON.parse(e.data);
      sseLastId = ev.id || sseLastId;
      const key = ev.conversation_id + "|" + ev.stage;
      liveStreams[key] = (liveStreams[key] || "") + ev.chunk;
      // Cull old entries (keep last 50)
      const keys = Object.keys(liveStreams);
      if (keys.length > 50) {
        const toDelete = keys.slice(0, keys.length - 50);
        toDelete.forEach(k => delete liveStreams[k]);
      }
      applySSEChunk(ev.conversation_id, ev.stage, key);
    } catch (_) { /* ignore parse errors */ }
  };
  sseConnection.onerror = function () {
    if (sseConnection.readyState === 2) { // CLOSED
      sseConnection = null;
    }
  };
}

function applySSEChunk(convId, stage, key) {
  // Only update if the current view matches this conversation+agent
  if (timelineMode !== "agent_stream") return;
  if (selectedConversation !== "__latest__" && selectedConversation !== convId) return;
  if (stage !== agentStreamAgent) return;

  const container = $("timeline");
  const text = liveStreams[key];
  if (!text || text.length === 0) return;

  const streamChanged = liveStreamKey !== key;
  if (streamChanged) {
    // Remove old live element, create new one
    if (liveStreamEl && liveStreamEl.parentNode) {
      liveStreamEl.remove();
    }
    liveStreamEl = null;
    liveStreamBody = null;
    liveStreamKey = key;
    lastKnownEntryCount = 0;
  }

  if (!liveStreamEl) {
    liveStreamEl = document.createElement("div");
    liveStreamEl.className = "stream-entry stream-live";

    const head = document.createElement("div");
    head.className = "stream-head";

    const cidSpan = document.createElement("span");
    cidSpan.className = "stream-cid";
    cidSpan.textContent = convId;
    head.appendChild(cidSpan);

    const labelSpan = document.createElement("span");
    labelSpan.className = "stream-label";
    labelSpan.innerHTML = '<span class="live-dot"></span> streaming...';
    head.appendChild(labelSpan);

    liveStreamEl.appendChild(head);

    liveStreamBody = document.createElement("pre");
    liveStreamBody.className = "stream-body";
    liveStreamEl.appendChild(liveStreamBody);

    container.appendChild(liveStreamEl);
  }

  liveStreamBody.textContent = text;

  // Auto-scroll to bottom
  const atBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 80;
  if (atBottom || streamChanged) {
    container.scrollTop = container.scrollHeight;
  }

  scheduleCompletionCheck();
}

function scheduleCompletionCheck() {
  if (sseRenderPending) return;
  sseRenderPending = true;
  requestAnimationFrame(() => {
    sseRenderPending = false;
    refresh().then((state) => {
      if (!state) return;
      const active = state.active || {};
      let currentCount = 0;
      if (selectedConversation === "__latest__") {
        for (const cid of Object.keys(active)) {
          const hist = ((active[cid] || {}).agent_history || {})[agentStreamAgent] || [];
          currentCount += hist.length;
        }
      } else {
        const slot = active[selectedConversation] || {};
        const hist = (slot.agent_history || {})[agentStreamAgent] || [];
        currentCount = hist.length;
      }
      if (currentCount > lastKnownEntryCount) {
        lastKnownEntryCount = currentCount;
        // Agent completed; remove live stream element and clear buffer
        if (liveStreamEl && liveStreamEl.parentNode) {
          liveStreamEl.remove();
        }
        liveStreamEl = null;
        liveStreamBody = null;
        liveStreamKey = null;
        // Clear buffers for completed entries
        for (const k of Object.keys(liveStreams)) {
          if (k.endsWith("|" + agentStreamAgent)) {
            delete liveStreams[k];
          }
        }
      }
    }).catch(() => {});
  });
}

connectSSE();
