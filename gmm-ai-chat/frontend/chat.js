const API_URL = "/chat";

// Replace with real auth session in production
const SESSION = {
  user_id: "demo_user",
  subscription_tier: "enterprise",
  licensed_industries: null,
  licensed_geographies: null,
};

let conversationHistory = [];

async function sendMessage() {
  const input = document.getElementById("user-input");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  hideSuggestions();
  appendMessage("user", message);
  const thinkingId = appendThinking();

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json", "ngrok-skip-browser-warning": "1" },
      body: JSON.stringify({ message, conversation_history: conversationHistory, ...SESSION }),
    });
    if (!res.ok) {
      const errText = await res.text();
      console.error("Server error", res.status, errText);
      removeThinking(thinkingId);
      appendMessage("assistant", `Server error (${res.status}): ${errText}`);
      return;
    }
    const data = await res.json();
    removeThinking(thinkingId);
    conversationHistory.push({ role: "user", content: message });
    conversationHistory.push({ role: "assistant", content: data.response });
    appendMessage("assistant", data.response, data.tool_calls_made);
  } catch (err) {
    console.error("Fetch error", err);
    removeThinking(thinkingId);
    appendMessage("assistant", `Connection error: ${err.message}. Is the backend running on port 8000?`);
  }
}

function appendMessage(role, content, toolCalls = []) {
  const msgs = document.getElementById("messages");
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.innerHTML = renderContent(content);
  if (toolCalls.length > 0) {
    const trace = document.createElement("div");
    trace.className = "tool-trace";
    trace.textContent = `Tools used: ${toolCalls.join(" → ")}`;
    div.appendChild(trace);
  }
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function appendThinking() {
  const msgs = document.getElementById("messages");
  const div = document.createElement("div");
  const id = `thinking-${Date.now()}`;
  div.id = id;
  div.className = "message assistant thinking";
  div.innerHTML = `<span class="dot"></span><span class="dot"></span><span class="dot"></span>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return id;
}

function removeThinking(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function buildTable(lines) {
  const rows = lines.filter(l => !l.match(/^\|[-| ]+\|$/));
  let html = "<table>";
  rows.forEach((row, i) => {
    const cells = row.split("|").filter(c => c.trim() !== "");
    const tag = i === 0 ? "th" : "td";
    html += "<tr>" + cells.map(c => `<${tag}>${c.trim()}</${tag}>`).join("") + "</tr>";
  });
  return html + "</table>";
}

function renderContent(text) {
  const lines = text.split("\n");
  let html = "", tableLines = [], inTable = false;
  for (const line of lines) {
    if (line.trim().startsWith("|")) {
      inTable = true; tableLines.push(line);
    } else {
      if (inTable) { html += buildTable(tableLines); tableLines = []; inTable = false; }
      const fmt = line.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/\*(.*?)\*/g, "<em>$1</em>");
      html += fmt ? `<p>${fmt}</p>` : "<br/>";
    }
  }
  if (inTable) html += buildTable(tableLines);
  return html;
}

function hideSuggestions() { document.getElementById("suggestions").style.display = "none"; }
function sendSuggestion(btn) { document.getElementById("user-input").value = btn.textContent; sendMessage(); }

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("user-input").addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
});