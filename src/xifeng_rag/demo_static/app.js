const token = new URLSearchParams(location.search).get("token") || "";
let questionHistory = [];

async function api(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Demo-Token": token },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `请求失败 (${response.status})`);
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

function setBusy(form, busy) {
  const button = form.querySelector("button[type=submit]");
  button.disabled = busy;
  button.dataset.label ||= button.innerHTML;
  button.innerHTML = busy ? "处理中…" : button.dataset.label;
}

function setStatus(id, text, error = false) {
  const node = document.getElementById(id);
  node.textContent = text;
  node.classList.toggle("error", error);
}

function renderQuestionHistory(items) {
  questionHistory = items;
  const section = document.getElementById("question-history");
  const list = document.getElementById("question-history-list");
  section.classList.toggle("hidden", items.length === 0);
  list.innerHTML = items.map((item, index) => `
    <div class="history-item">
      <button class="history-question" type="button" data-index="${index}" title="${escapeHtml(item.question)}">${escapeHtml(item.question)}</button>
      <button class="history-delete" type="button" data-index="${index}" aria-label="删除这个问题">删除</button>
    </div>`).join("");
  list.querySelectorAll(".history-question").forEach(button => {
    button.addEventListener("click", () => {
      const question = questionHistory[Number(button.dataset.index)].question;
      const input = document.getElementById("answer-query");
      input.value = question;
      input.focus();
    });
  });
  list.querySelectorAll(".history-delete").forEach(button => {
    button.addEventListener("click", async () => {
      try {
        const question = questionHistory[Number(button.dataset.index)].question;
        const data = await api("/api/questions/delete", { question });
        renderQuestionHistory(data.questions);
      } catch (error) {
        setStatus("answer-status", `删除最近问题失败：${error.message}`, true);
      }
    });
  });
}

async function loadQuestionHistory() {
  try {
    const data = await api("/api/questions/history", {});
    renderQuestionHistory(data.questions);
  } catch (error) {
    setStatus("answer-status", `最近问题读取失败：${error.message}`, true);
  }
}

document.querySelectorAll(".tab").forEach(button => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab, .panel").forEach(node => node.classList.remove("active"));
    button.classList.add("active");
    document.getElementById(button.dataset.panel).classList.add("active");
  });
});

document.getElementById("discover-form").addEventListener("submit", async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const query = document.getElementById("discover-query").value.trim();
  if (!query) return setStatus("discover-status", "请先描述你记得的内容。", true);
  setBusy(form, true);
  setStatus("discover-status", "正在从文章概括开始逐层反查…");
  document.getElementById("discover-results").innerHTML = "";
  try {
    const data = await api("/api/articles/search", { query, top_k: 5 });
    const layers = data.activated_layers.join(" → ") || "overview";
    setStatus("discover-status", `找到 ${data.results.length} 篇候选 · 实际触发层：${layers}`);
    document.getElementById("discover-results").innerHTML = data.results.map((item, index) => `
      <article class="article-card">
        <div class="rank">${index + 1}</div>
        <div>
          <h3>${escapeHtml(item.title)}</h3>
          <p>${escapeHtml(item.article_summary)}</p>
          <div class="chips">
            ${(item.matched_topics || []).map(value => `<span class="chip">${escapeHtml(value)}</span>`).join("")}
            ${(item.matched_subtopics || []).map(value => `<span class="chip">细话题 · ${escapeHtml(value)}</span>`).join("")}
          </div>
          <div class="meta">
            <span>${escapeHtml(item.published_at || "发布时间未知")}</span>
            <span>${escapeHtml((item.sources || []).join(" · "))}</span>
            ${item.rrf_score == null ? "" : `<span>RRF ${Number(item.rrf_score).toFixed(5)}</span>`}
          </div>
        </div>
      </article>`).join("") || `<p>没有找到相关文章。</p>`;
  } catch (error) {
    setStatus("discover-status", error.message, true);
  } finally {
    setBusy(form, false);
  }
});

document.getElementById("answer-form").addEventListener("submit", async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const question = document.getElementById("answer-query").value.trim();
  if (!question) return setStatus("answer-status", "请先输入问题。", true);
  setBusy(form, true);
  setStatus("answer-status", "正在本地检索并组装证据…");
  document.getElementById("disclosure").classList.add("hidden");
  document.getElementById("answer-result").classList.add("hidden");
  try {
    const data = await api("/api/answers/prepare", { question });
    await loadQuestionHistory();
    renderDisclosure(data);
    setStatus("answer-status", data.evidence_count ? "证据准备完成，请检查外发范围。" : "没有检索到证据，将在本地拒答。");
  } catch (error) {
    setStatus("answer-status", error.message, true);
  } finally {
    setBusy(form, false);
  }
});

function renderDisclosure(data) {
  const node = document.getElementById("disclosure");
  node.classList.remove("hidden");
  node.innerHTML = `
    <h3>本次固定证据范围</h3>
    <div class="scope-grid">
      <div class="scope-item"><b>${data.evidence_count}</b><span>证据块</span></div>
      <div class="scope-item"><b>${Number(data.original_text_chars).toLocaleString()}</b><span>原文字符</span></div>
      <div class="scope-item"><b>${data.max_external_requests}</b><span>最大请求</span></div>
      <div class="scope-item"><b>${escapeHtml(data.model)}</b><span>${escapeHtml(data.provider)}</span></div>
    </div>
    <div class="evidence-list">${data.evidence.map(item => `
      <div class="evidence-preview">
        <strong>${escapeHtml(item.citation_id)} · ${escapeHtml(item.article_title)} · ¶${item.paragraph_range[0]}–${item.paragraph_range[1]}</strong>
        <p>${escapeHtml(item.preview)}</p>
      </div>`).join("") || `<div class="evidence-preview">本次没有证据，不会调用外部服务。</div>`}
    </div>
    <p class="confirm-note">${data.will_call_external_generator
      ? `确认后，上述 ${data.evidence_count} 块证据将发送到 ${escapeHtml(data.provider_url)}。`
      : "确认后将在本地返回证据不足，不读取 API 密钥。"}</p>
    <button id="confirm-answer" class="primary">${data.will_call_external_generator ? "确认范围并生成回答" : "查看本地拒答"} <b>→</b></button>`;
  document.getElementById("confirm-answer").addEventListener("click", event => confirmAnswer(data.prepared_id, event.currentTarget));
}

async function confirmAnswer(preparedId, button) {
  button.disabled = true;
  button.textContent = "正在生成…";
  setStatus("answer-status", "正在生成结构化回答，请稍候…");
  try {
    const data = await api("/api/answers/confirm", { prepared_id: preparedId });
    renderAnswer(data);
    document.getElementById("disclosure").classList.add("hidden");
    setStatus("answer-status", "回答完成。点击引用可核对本地原文。" );
  } catch (error) {
    setStatus("answer-status", error.message, true);
    button.disabled = false;
    button.textContent = "准备结果已失效，请重新检索";
  }
}

function renderAnswer(data) {
  const node = document.getElementById("answer-result");
  const answer = data.answer;
  const labels = { answered: "已回答", partially_answered: "部分回答", insufficient_evidence: "证据不足" };
  node.classList.remove("hidden");
  node.innerHTML = `
    <span class="answer-status">${labels[answer.status] || escapeHtml(answer.status)}</span>
    <h3>基于本地文章的回答</h3>
    ${(answer.grounded_sections || []).map(section => `
      <div class="answer-section">
        ${escapeHtml(section.text)}
        ${(section.evidence_ids || []).map(id => `<span class="citation">[${escapeHtml(id)}]</span>`).join("")}
      </div>`).join("")}
    ${(answer.limitations || []).length ? `<div class="limitations"><strong>资料限制</strong><ul>${answer.limitations.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : ""}
    <div class="sources">
      <h3>原文引用</h3>
      ${data.sources.map(source => `<details>
        <summary>${escapeHtml(source.citation_id)} · ${escapeHtml(source.article_title)} · ¶${source.paragraph_range[0]}–${source.paragraph_range[1]}</summary>
        <p>${escapeHtml(source.original_text)}</p>
      </details>`).join("") || `<p>本次回答没有引用原文。</p>`}
    </div>
    <div class="run-meta">${data.request_count} 次外部请求 · ${Math.round(data.latency_ms).toLocaleString()} ms · ${escapeHtml(data.model || "本地拒答")}</div>`;
}

document.getElementById("shutdown").addEventListener("click", async () => {
  try {
    await api("/api/shutdown", {});
    document.body.innerHTML = `<div class="page-shell"><h1>演示服务已关闭</h1><p class="subtitle">可以关闭这个页面了。</p></div>`;
  } catch (error) {
    alert(error.message);
  }
});

document.getElementById("clear-history").addEventListener("click", async () => {
  try {
    const data = await api("/api/questions/clear", {});
    renderQuestionHistory(data.questions);
  } catch (error) {
    setStatus("answer-status", `清空最近问题失败：${error.message}`, true);
  }
});

loadQuestionHistory();
