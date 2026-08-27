const token = new URLSearchParams(location.search).get("token") || "";
const headers = {"Content-Type": "application/json", "X-Evaluation-Token": token};
let state = null;
let currentDataset = "";
let currentDocument = null;
let points = [];
let selectionStart = null;
let activePointIndex = null;
let editingQuestionId = null;

async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: {...headers, ...(options.headers || {})}});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function notice(message, error = false) {
  const node = document.getElementById("notice");
  node.textContent = message;
  node.className = error ? "error" : "";
}
function selectedDataset() { return state.datasets.find(item => item.id === currentDataset); }

async function refresh() {
  state = await api("/api/state");
  const datasets = document.getElementById("dataset-list");
  datasets.innerHTML = state.datasets.map(item =>
    `<option value="${item.id}">${escapeHtml(item.name)} ${escapeHtml(item.version)} · ${item.status} · ${item.reviewed_count}/${item.question_count}</option>`
  ).join("");
  if (!currentDataset && state.datasets.length) currentDataset = state.datasets[0].id;
  datasets.value = currentDataset;
  document.getElementById("article-list").innerHTML = state.articles.map(item =>
    `<option value="${item.version_id}">${escapeHtml(item.title)} · ${item.chunk_count} 块</option>`
  ).join("");
  renderQuestions();
  updateDatasetControls();
  renderJob(state.ai_job);
}
function renderQuestions() {
  const rows = state.questions.filter(item => item.dataset_id === currentDataset);
  document.getElementById("question-list").innerHTML = rows.map(item =>
    `<button class="question-item ${item.review_status}" data-id="${item.id}">${item.ordinal}. ${escapeHtml(item.query)}</button>`
  ).join("");
  document.querySelectorAll(".question-item").forEach(button => button.addEventListener("click", () => loadQuestion(button.dataset.id)));
}
function updateDatasetControls() {
  const dataset = selectedDataset();
  const frozen = !dataset || dataset.status === "frozen";
  document.querySelector("#question-form button").disabled = frozen;
  document.getElementById("freeze").disabled = frozen;
  document.getElementById("question-split").disabled = dataset?.purpose === "pilot";
}

async function loadQuestion(id) {
  currentDocument = await api(`/api/questions/${id}`);
  points = structuredClone(currentDocument.answer_points);
  activePointIndex = points.length ? 0 : null;
  selectionStart = null;
  document.getElementById("workspace").classList.remove("hidden");
  document.getElementById("workspace").classList.toggle("frozen", currentDocument.dataset.status === "frozen");
  document.getElementById("article-title").textContent = `${currentDocument.article.title} · ${currentDocument.article.version_id}`;
  document.getElementById("current-question").textContent = currentDocument.question.query;
  document.getElementById("reference-answer").value = currentDocument.question.reference_answer;
  document.getElementById("paragraphs").innerHTML = currentDocument.paragraphs.map(item =>
    `<div class="paragraph" data-number="${item.number}"><span class="unit-number">¶${item.number}</span>${escapeHtml(item.text)}</div>`
  ).join("");
  document.querySelectorAll(".paragraph").forEach(node => node.addEventListener("click", () => selectParagraph(Number(node.dataset.number))));
  renderPoints();
  renderProposals();
  const frozen = currentDocument.dataset.status === "frozen";
  for (const id of ["edit-question", "ai-assist", "add-point", "save-draft", "save-reviewed"]) document.getElementById(id).disabled = frozen;
  document.querySelectorAll(".review-pane input, .review-pane textarea, .review-pane button").forEach(control => control.disabled = frozen);
}
function selectParagraph(number) {
  if (currentDocument.dataset.status === "frozen") return;
  if (selectionStart === null) {
    selectionStart = number;
    document.getElementById("selection-help").textContent = `起点 ¶${number}；再点一次选择终点`;
    highlightRange(number, number);
    return;
  }
  const start = Math.min(selectionStart, number);
  const end = Math.max(selectionStart, number);
  selectionStart = null;
  if (!points.length) addPoint();
  const target = activePointIndex === null ? points.length - 1 : activePointIndex;
  points[target].evidence_ranges.push({start_paragraph: start, end_paragraph: end});
  document.getElementById("selection-help").textContent = `已把 ¶${start}–${end} 加到 ${points[target].claim_group}`;
  highlightRange(start, end);
  renderPoints();
}
function highlightRange(start, end) {
  document.querySelectorAll(".paragraph").forEach(node => node.classList.toggle("selected", Number(node.dataset.number) >= start && Number(node.dataset.number) <= end));
}
function addPoint() {
  points.push({claim_group: `point-${points.length + 1}`, claim: "", uncertainty: "", evidence_ranges: []});
  activePointIndex = points.length - 1;
  renderPoints();
}
function renderPoints() {
  document.getElementById("answer-points").innerHTML = points.map((point, index) => `
    <div class="point ${index === activePointIndex ? "active" : ""}" data-index="${index}">
      <div class="point-head"><input class="group" value="${escapeAttr(point.claim_group)}"><textarea class="claim" placeholder="答案要点">${escapeHtml(point.claim)}</textarea><button class="remove-point danger">删除</button></div>
      <label>不确定性<input class="uncertainty" value="${escapeAttr(point.uncertainty || "")}"></label>
      <div class="ranges">${point.evidence_ranges.map((range, rangeIndex) => `<span class="range">¶${range.start_paragraph}–${range.end_paragraph} <button data-range="${rangeIndex}">×</button></span>`).join("")}</div>
    </div>`).join("");
  document.querySelectorAll(".point").forEach(node => {
    const index = Number(node.dataset.index);
    node.addEventListener("click", () => { activePointIndex = index; renderPoints(); });
    node.querySelectorAll("input, textarea, button").forEach(control => control.addEventListener("click", event => event.stopPropagation()));
    node.querySelector(".group").addEventListener("input", event => points[index].claim_group = event.target.value);
    node.querySelector(".claim").addEventListener("input", event => points[index].claim = event.target.value);
    node.querySelector(".uncertainty").addEventListener("input", event => points[index].uncertainty = event.target.value);
    node.querySelector(".remove-point").addEventListener("click", () => { points.splice(index, 1); activePointIndex = points.length ? Math.min(index, points.length - 1) : null; renderPoints(); });
    node.querySelectorAll(".range button").forEach(button => button.addEventListener("click", () => { points[index].evidence_ranges.splice(Number(button.dataset.range), 1); renderPoints(); }));
  });
}
function renderProposals() {
  document.getElementById("proposals").innerHTML = currentDocument.ai_proposals.filter(item => item.status === "pending_review").map(item => `
    <div class="proposal"><strong>Codex 候选 · ${escapeHtml(item.model_id)}</strong><pre>${escapeHtml(JSON.stringify(item.proposal, null, 2))}</pre><button data-proposal="${item.id}" data-accept="true">接受到草案</button> <button class="danger" data-proposal="${item.id}" data-accept="false">拒绝</button></div>`).join("");
  document.querySelectorAll("[data-proposal]").forEach(button => button.addEventListener("click", async () => {
    await api(`/api/proposals/${button.dataset.proposal}/decision`, {method: "POST", body: JSON.stringify({accept: button.dataset.accept === "true"})});
    notice(button.dataset.accept === "true" ? "候选已复制到人工草案，请继续审核" : "候选已拒绝");
    await loadQuestion(currentDocument.question.id);
    await refresh();
  }));
}
async function saveReview(reviewed) {
  await api(`/api/questions/${currentDocument.question.id}/review`, {method: "POST", body: JSON.stringify({reference_answer: document.getElementById("reference-answer").value, answer_points: points, reviewed})});
  notice(reviewed ? "已保存并标记人工审核完成" : "审核草案已保存");
  await loadQuestion(currentDocument.question.id);
  await refresh();
}
async function aiAssist() {
  const disclosure = await api(`/api/questions/${currentDocument.question.id}/ai-disclosure`, {method: "POST", body: "{}"});
  const message = `问题：${disclosure.question}\n文章：${disclosure.article_title}\n文章 ID：${disclosure.article_id}\n版本：${disclosure.article_version}\n正文哈希：${disclosure.canonical_hash}\n清洗后字符数：${disclosure.character_count}\n原子单元数：${disclosure.unit_count}\nProvider：${disclosure.provider}\n模型：${disclosure.model}\n最大请求次数：${disclosure.max_requests}\n\n${disclosure.disclosure}\n\n确认发送？`;
  if (!window.confirm(message)) return;
  await api("/api/ai-confirm", {method: "POST", body: JSON.stringify({confirmation: disclosure.confirmation})});
  notice("Codex 正在生成候选；页面会自动刷新状态");
  pollJob();
}
async function pollJob() {
  const job = await api("/api/ai-job");
  renderJob(job);
  if (job.status === "running") return setTimeout(pollJob, 1500);
  if (currentDocument && job.question_id === currentDocument.question.id) await loadQuestion(currentDocument.question.id);
  await refresh();
}
function renderJob(job) {
  document.getElementById("ai-status").textContent = job.status === "running" ? "Codex 运行中…" : job.status === "failed" ? `失败：${job.error_message || job.error_type}` : job.status === "succeeded" ? "候选已生成" : "";
  document.getElementById("ai-assist").disabled = job.status === "running" || currentDocument?.dataset.status === "frozen";
}

document.getElementById("dataset-form").addEventListener("submit", async event => {
  event.preventDefault();
  const result = await api("/api/datasets", {method: "POST", body: JSON.stringify({name: document.getElementById("dataset-name").value, version: document.getElementById("dataset-version").value, purpose: document.getElementById("dataset-purpose").value})});
  currentDataset = result.id; notice("draft 数据集已创建"); await refresh();
});
document.getElementById("dataset-list").addEventListener("change", event => { currentDataset = event.target.value; renderQuestions(); updateDatasetControls(); });
document.getElementById("question-form").addEventListener("submit", async event => {
  event.preventDefault();
  const result = await api("/api/questions", {method: "POST", body: JSON.stringify({dataset_id: currentDataset, question_id: editingQuestionId, query: document.getElementById("question-query").value, version_id: document.getElementById("article-list").value, split: document.getElementById("question-split").value})});
  editingQuestionId = null; document.querySelector("#question-form button").textContent = "保存为新问题"; document.getElementById("question-query").value = ""; notice("问题已保存；修改题目会清空原人工答案和证据"); await refresh(); await loadQuestion(result.id);
});
document.getElementById("edit-question").addEventListener("click", () => {
  if ((currentDocument.question.reference_answer || points.length) && !window.confirm("编辑题目会在保存时清空现有人工答案和证据，继续？")) return;
  editingQuestionId = currentDocument.question.id;
  document.getElementById("question-query").value = currentDocument.question.query;
  document.getElementById("article-list").value = currentDocument.article.version_id;
  document.getElementById("question-split").value = currentDocument.question.split;
  document.querySelector("#question-form button").textContent = "更新当前问题";
  document.getElementById("question-query").focus();
});
document.getElementById("freeze").addEventListener("click", async () => {
  const dataset = selectedDataset();
  if (!dataset || !window.confirm(`冻结 ${dataset.name} ${dataset.version}？冻结后不可修改。`)) return;
  const result = await api(`/api/datasets/${dataset.id}/freeze`, {method: "POST", body: "{}"});
  notice(`已冻结：${result.content_hash}`); await refresh(); if (currentDocument) await loadQuestion(currentDocument.question.id);
});
document.getElementById("add-point").addEventListener("click", addPoint);
document.getElementById("save-draft").addEventListener("click", () => saveReview(false).catch(showError));
document.getElementById("save-reviewed").addEventListener("click", () => saveReview(true).catch(showError));
document.getElementById("ai-assist").addEventListener("click", () => aiAssist().catch(showError));
document.getElementById("shutdown").addEventListener("click", async () => {
  await api("/api/shutdown", {method: "POST", body: "{}"}); document.body.innerHTML = "<main><h1>评测工作台已关闭，可以安全关闭此标签页</h1></main>";
});
function showError(error) { notice(error.message, true); }
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]); }
function escapeAttr(value) { return escapeHtml(value); }
window.addEventListener("unhandledrejection", event => showError(event.reason));
refresh().catch(showError);
