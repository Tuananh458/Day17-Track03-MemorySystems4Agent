const baselineMessages = document.getElementById("baselineMessages");
const advancedMessages = document.getElementById("advancedMessages");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const userIdInput = document.getElementById("userId");
const threadIdInput = document.getElementById("threadId");
const profileMd = document.getElementById("profileMd");
const compactSummary = document.getElementById("compactSummary");
const baselineMetrics = document.getElementById("baselineMetrics");
const advancedMetrics = document.getElementById("advancedMetrics");
const providerPill = document.getElementById("providerPill");
const modePill = document.getElementById("modePill");
const offlineBanner = document.getElementById("offlineBanner");
const benchmarkBox = document.getElementById("benchmarkBox");
const benchmarkResult = document.getElementById("benchmarkResult");

let baselineTotals = { tokens: 0, prompt: 0 };
let advancedTotals = { tokens: 0, prompt: 0, compact: 0 };

function appendMessage(container, role, text, meta = "") {
  const empty = container.querySelector(".empty-state");
  if (empty) empty.remove();

  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = text;
  if (meta) {
    const metaNode = document.createElement("span");
    metaNode.className = "meta";
    metaNode.textContent = meta;
    node.appendChild(metaNode);
  }
  container.appendChild(node);
  container.scrollTop = container.scrollHeight;
  return node;
}

function showTyping(container) {
  const node = document.createElement("div");
  node.className = "message typing";
  node.textContent = "Đang trả lời...";
  node.dataset.typing = "1";
  container.appendChild(node);
  container.scrollTop = container.scrollHeight;
  return node;
}

function clearTyping(container) {
  container.querySelectorAll('[data-typing="1"]').forEach((node) => node.remove());
}

function setEmptyStates() {
  baselineMessages.innerHTML = '<div class="empty-state">Baseline chỉ nhớ trong thread hiện tại.</div>';
  advancedMessages.innerHTML = '<div class="empty-state">Advanced sẽ lưu facts vào User.md và nén hội thoại dài.</div>';
}

async function api(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      ...options,
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `HTTP ${response.status}`);
    }
    return response.json();
  } finally {
    clearTimeout(timeout);
  }
}

function updateMetrics(data) {
  baselineTotals.tokens += data.baseline.tokens || 0;
  baselineTotals.prompt += data.baseline.prompt_tokens || 0;
  advancedTotals.tokens += data.advanced.tokens || 0;
  advancedTotals.prompt += data.advanced.prompt_tokens || 0;
  advancedTotals.compact = data.compactions || 0;

  baselineMetrics.innerHTML = `
    <span>Tokens: ${baselineTotals.tokens}</span>
    <span>Prompt: ${baselineTotals.prompt}</span>
    <span>Mode: ${data.baseline.mode}</span>
  `;
  advancedMetrics.innerHTML = `
    <span>Tokens: ${advancedTotals.tokens}</span>
    <span>Prompt: ${advancedTotals.prompt}</span>
    <span>Compact: ${advancedTotals.compact}</span>
    <span>Mode: ${data.advanced.mode}</span>
  `;

  profileMd.textContent = data.profile_md || "# User Profile";
  compactSummary.textContent = data.compact_summary?.trim() || "(chưa có)";
}

async function sendMessage(message) {
  const payload = {
    user_id: userIdInput.value.trim(),
    thread_id: threadIdInput.value.trim(),
    message,
  };

  appendMessage(baselineMessages, "user", message);
  appendMessage(advancedMessages, "user", message);
  showTyping(baselineMessages);
  showTyping(advancedMessages);
  sendBtn.disabled = true;

  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    clearTyping(baselineMessages);
    clearTyping(advancedMessages);

    const baselineAnswer = data.baseline?.answer?.trim();
    const advancedAnswer = data.advanced?.answer?.trim();

    if (!baselineAnswer) {
      appendMessage(baselineMessages, "assistant", "Không nhận được phản hồi từ Baseline.");
    } else {
      const baselineMeta = `+${data.baseline.tokens} tok | prompt ${data.baseline.prompt_tokens}`;
      appendMessage(baselineMessages, "assistant", baselineAnswer, baselineMeta);
    }

    if (!advancedAnswer) {
      appendMessage(advancedMessages, "assistant", "Không nhận được phản hồi từ Advanced.");
    } else {
      const advancedMeta = `+${data.advanced.tokens} tok | prompt ${data.advanced.prompt_tokens}`;
      appendMessage(advancedMessages, "assistant", advancedAnswer, advancedMeta);
    }

    if (data.baseline?.error) {
      appendMessage(baselineMessages, "assistant", `LLM lỗi → fallback: ${data.baseline.error}`);
    }
    if (data.advanced?.error) {
      appendMessage(advancedMessages, "assistant", `LLM lỗi → fallback: ${data.advanced.error}`);
    }

    updateMetrics(data);
  } catch (error) {
    clearTyping(baselineMessages);
    clearTyping(advancedMessages);
    const msg = error.name === "AbortError"
      ? "Request quá 30s — thử lại hoặc bật FORCE_OFFLINE=true trong .env"
      : error.message;
    appendMessage(baselineMessages, "assistant", `Lỗi: ${msg}`);
    appendMessage(advancedMessages, "assistant", `Lỗi: ${msg}`);
  } finally {
    sendBtn.disabled = false;
  }
}

async function loadHealth() {
  try {
    const data = await api("/api/health");
    const base = data.base_url ? ` @ ${data.base_url}` : "";
    providerPill.textContent = `Provider: ${data.provider} / ${data.model}${base}`;
    modePill.textContent = `Baseline: ${data.baseline_mode} | Advanced: ${data.advanced_mode}`;

    if (!data.llm_live) {
      offlineBanner.hidden = false;
      offlineBanner.textContent = data.llm_reason || "Offline mode";
      offlineBanner.title = data.llm_reason || "";
    } else {
      offlineBanner.hidden = true;
    }
  } catch {
    providerPill.textContent = "Provider: offline";
    modePill.textContent = "Không kết nối server";
    offlineBanner.hidden = false;
    offlineBanner.textContent = "Không kết nối server — chạy: python demo_server.py";
  }
}

sendBtn.addEventListener("click", () => {
  const message = messageInput.value.trim();
  if (!message) return;
  messageInput.value = "";
  sendMessage(message);
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendBtn.click();
  }
});

document.getElementById("newThreadBtn").addEventListener("click", async () => {
  const data = await api("/api/new-thread", { method: "POST" });
  threadIdInput.value = data.thread_id;
  baselineTotals = { tokens: 0, prompt: 0 };
  advancedTotals = { tokens: 0, prompt: 0, compact: 0 };
  setEmptyStates();
});

document.getElementById("resetThreadBtn").addEventListener("click", async () => {
  await api("/api/reset", {
    method: "POST",
    body: JSON.stringify({
      user_id: userIdInput.value.trim(),
      thread_id: threadIdInput.value.trim(),
      clear_profile: false,
    }),
  });
  baselineTotals = { tokens: 0, prompt: 0 };
  advancedTotals = { tokens: 0, prompt: 0, compact: 0 };
  setEmptyStates();
  profileMd.textContent = "# User Profile";
  compactSummary.textContent = "(chưa có)";
});

document.getElementById("resetAllBtn").addEventListener("click", async () => {
  await api("/api/reset", {
    method: "POST",
    body: JSON.stringify({
      user_id: userIdInput.value.trim(),
      thread_id: null,
      clear_profile: true,
    }),
  });
  baselineTotals = { tokens: 0, prompt: 0 };
  advancedTotals = { tokens: 0, prompt: 0, compact: 0 };
  setEmptyStates();
  profileMd.textContent = "# User Profile";
  compactSummary.textContent = "(chưa có)";
});

document.getElementById("benchmarkBtn").addEventListener("click", async () => {
  benchmarkBox.hidden = false;
  benchmarkResult.textContent = "Đang chạy benchmark...";
  const data = await api("/api/benchmark/quick");
  benchmarkResult.textContent = data.table;
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const message = chip.dataset.message;
    if (message) sendMessage(message);
  });
});

setEmptyStates();
loadHealth();
