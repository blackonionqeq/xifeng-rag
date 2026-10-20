const token = new URLSearchParams(location.search).get("token") || "";
const headers = {"Content-Type": "application/json", "X-Evaluation-Token": token};
let state = null;
let currentDataset = "";
let currentDocument = null;
let points = [];
let selectionStart = null;
let activePointIndex = null;
let editingQuestionId = null;
let currentRun = null;
let currentRuns = [];
let currentComparison = null;

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
  renderEvaluationJob(state.evaluation_job);
  renderRunControls();
}
function renderQuestions() {
  const rows = state.questions.filter(item => item.dataset_id === currentDataset);
  document.getElementById("question-list").innerHTML = rows.map(item =>
    `<button class="question-item ${item.review_status}" data-id="${item.id}"><span class="question-meta">${escapeHtml(item.split)} · ${escapeHtml(item.difficulty)} · ${escapeHtml(item.origin)} · ${item.answerable ? "可回答" : "应拒答"}</span>${item.ordinal}. ${escapeHtml(item.query)}</button>`
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
  document.getElementById("current-question").textContent = `[${currentDocument.question.split} · ${currentDocument.question.difficulty} · ${currentDocument.question.origin} · ${currentDocument.question.answerable ? "可回答" : "应拒答"}] ${currentDocument.question.query}`;
  document.getElementById("reference-answer").value = currentDocument.question.reference_answer;
  document.getElementById("reference-answer-label").textContent = currentDocument.question.answerable ? "人工确认的参考答案" : "预期拒答说明";
  document.getElementById("paragraphs").innerHTML = currentDocument.paragraphs.map(item =>
    `<div class="paragraph" data-number="${item.number}"><span class="unit-number">¶${item.number}</span>${escapeHtml(item.text)}</div>`
  ).join("");
  document.querySelectorAll(".paragraph").forEach(node => node.addEventListener("click", () => selectParagraph(Number(node.dataset.number))));
  renderPoints();
  renderProposals();
  const frozen = currentDocument.dataset.status === "frozen";
  for (const id of ["edit-question", "ai-assist", "add-point", "save-draft", "save-reviewed"]) document.getElementById(id).disabled = frozen;
  document.querySelectorAll(".review-pane input, .review-pane textarea, .review-pane button").forEach(control => control.disabled = frozen);
  document.getElementById("add-point").disabled = frozen || !currentDocument.question.answerable;
  document.getElementById("ai-assist").disabled = frozen || !currentDocument.question.answerable;
  document.getElementById("selection-help").textContent = currentDocument.question.answerable ? "点击起点和终点选择证据" : "不可回答题不标注证据";
}
function selectParagraph(number) {
  if (currentDocument.dataset.status === "frozen" || !currentDocument.question.answerable) return;
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
  document.getElementById("ai-assist").disabled = job.status === "running" || currentDocument?.dataset.status === "frozen" || currentDocument?.question.answerable === false;
}

function renderRunControls() {
  const frozen = state.datasets.filter(item => item.status === "frozen");
  const datasetSelect = document.getElementById("run-dataset");
  const previous = datasetSelect.value;
  datasetSelect.innerHTML = frozen.map(item => `<option value="${item.id}">${escapeHtml(item.name)} ${escapeHtml(item.version)}</option>`).join("");
  if (frozen.some(item => item.id === previous)) datasetSelect.value = previous;
  else if (frozen.some(item => item.id === currentDataset)) datasetSelect.value = currentDataset;
  renderCompatibleBuilds();
}
function renderCompatibleBuilds() {
  const dataset = state.datasets.find(item => item.id === document.getElementById("run-dataset").value);
  const builds = state.builds.filter(item => dataset && item.corpus_snapshot === dataset.corpus_snapshot);
  const buildSelect = document.getElementById("run-build");
  const previous = buildSelect.value;
  buildSelect.innerHTML = builds.map(item => `<option value="${item.id}">${item.active ? "活动 · " : ""}${escapeHtml(item.id)} · ${escapeHtml(item.profile_id)}</option>`).join("");
  if (builds.some(item => item.id === previous)) buildSelect.value = previous;
  else if (builds.some(item => item.active)) buildSelect.value = builds.find(item => item.active).id;
}
async function loadRuns(selectRunId = "") {
  const datasetId = document.getElementById("run-dataset").value;
  currentRuns = datasetId ? await api(`/api/datasets/${encodeURIComponent(datasetId)}/runs`) : [];
  const list = document.getElementById("run-list");
  list.innerHTML = currentRuns.map(item => `<option value="${item.id}">${formatDate(item.created_at)} · ${escapeHtml(item.split)} · ${escapeHtml(item.retrieval_mode || "—")} · K=${item.k}</option>`).join("");
  const baseline = document.getElementById("baseline-run");
  baseline.innerHTML = `<option value="">不对比</option>${currentRuns.map(item => `<option value="${item.id}">${formatDate(item.created_at)} · ${escapeHtml(item.retrieval_mode || "—")}</option>`).join("")}`;
  if (selectRunId || currentRuns.length) {
    list.value = selectRunId || currentRuns[0].id;
    await loadRun(list.value);
  } else {
    currentRun = null;
    document.getElementById("run-results").classList.add("hidden");
    document.getElementById("empty-results").classList.remove("hidden");
  }
}
async function loadRun(runId) {
  if (!runId) return;
  currentRun = await api(`/api/runs/${encodeURIComponent(runId)}`);
  document.getElementById("empty-results").classList.add("hidden");
  document.getElementById("run-results").classList.remove("hidden");
  document.getElementById("result-title").textContent = `${currentRun.retrieval_mode || "未知模式"} · ${currentRun.split} · K=${currentRun.k}`;
  document.getElementById("result-meta").textContent = `运行 ${currentRun.id} · Build ${currentRun.build_id} · Profile ${currentRun.profile_id || "—"} · Git ${currentRun.code_commit || "未知"}${currentRun.worktree_dirty ? "（dirty）" : ""}`;
  currentComparison = await renderComparison();
  renderGroupedMetrics(currentRun.metrics);
  renderQuestionResults(currentRun.questions, currentComparison);
}
async function renderComparison() {
  const baselineId = document.getElementById("baseline-run").value;
  const status = document.getElementById("comparison-status");
  let comparison = null;
  if (baselineId && baselineId !== currentRun.id) comparison = await api(`/api/runs/compare?baseline=${encodeURIComponent(baselineId)}&current=${encodeURIComponent(currentRun.id)}`);
  if (comparison) {
    status.textContent = comparison.compatible ? "可直接比较" : `不可直接比较：${comparison.reasons.join("、")}`;
    status.className = `badge ${comparison.compatible ? "good" : "warn"}`;
  } else {
    status.textContent = "";
    status.className = "badge";
  }
  const labels = {precision_at_k: "Precision@K", recall_at_k: "Recall@K", mrr: "MRR", ndcg_at_k: "nDCG@K", evidence_coverage_at_k: "证据覆盖率", latency_mean_ms: "平均延迟", latency_p95_ms: "P95 延迟"};
  document.getElementById("metric-cards").innerHTML = Object.entries(labels).map(([key, label]) => {
    const value = currentRun.metrics[key];
    const delta = comparison?.metrics[key]?.delta;
    const latency = key.includes("latency");
    const deltaClass = delta == null || delta === 0 ? "" : ((latency ? delta < 0 : delta > 0) ? "improved" : "regressed");
    return `<div class="metric-card"><span>${label}</span><strong>${formatMetric(value, latency)}</strong>${delta == null ? "" : `<small class="${deltaClass}">${delta >= 0 ? "+" : ""}${formatMetric(delta, latency)}</small>`}</div>`;
  }).join("");
  return comparison;
}
function renderGroupedMetrics(metrics) {
  const groups = [["难度", metrics.by_difficulty], ["来源", metrics.by_origin]];
  document.getElementById("grouped-metrics").innerHTML = groups.map(([label, values]) => `<div class="group-block"><h4>${label}</h4><table><thead><tr><th>分组</th><th>题数</th><th>Recall</th><th>MRR</th><th>证据覆盖</th></tr></thead><tbody>${Object.entries(values || {}).map(([name, item]) => `<tr><td>${escapeHtml(name)}</td><td>${item.question_count}</td><td>${formatMetric(item.recall_at_k)}</td><td>${formatMetric(item.mrr)}</td><td>${formatMetric(item.evidence_coverage_at_k)}</td></tr>`).join("")}</tbody></table></div>`).join("");
}
function renderQuestionResults(questions, comparison) {
  const comparisons = new Map((comparison?.questions || []).map(item => [item.question_id, item]));
  const priority = {regressed: 0, improved: 1, incompatible: 2, unchanged: 3};
  const sorted = [...questions].sort((a, b) => (priority[comparisons.get(a.question_id)?.status] ?? 2) - (priority[comparisons.get(b.question_id)?.status] ?? 2) || Number(a.should_abstain) - Number(b.should_abstain) || Number(a.evidence_coverage_at_k) - Number(b.evidence_coverage_at_k) || a.question_id.localeCompare(b.question_id));
  document.getElementById("question-results").innerHTML = sorted.map(item => {
    const compared = comparisons.get(item.question_id);
    const comparisonLabel = compared?.status === "regressed" ? "退化" : compared?.status === "improved" ? "改善" : compared?.status === "unchanged" ? "不变" : "";
    const comparisonClass = compared?.status === "regressed" ? "regressed" : compared?.status === "improved" ? "improved" : "";
    return `<details class="question-result ${Number(item.evidence_coverage_at_k) < 1 && !item.should_abstain ? "failed" : ""}">
    <summary><span><b>${escapeHtml(item.query || item.question_id)}</b><small>${escapeHtml(item.difficulty)} · ${escapeHtml(item.origin)}${item.should_abstain ? " · 应拒答" : ""}${comparisonLabel ? ` · <em class="${comparisonClass}">${comparisonLabel}</em>` : ""}</small></span><span>Recall ${formatMetric(item.recall_at_k)}${compared?.metrics.recall_at_k.delta == null ? "" : ` (${formatSigned(compared.metrics.recall_at_k.delta)})`} · 覆盖 ${formatMetric(item.evidence_coverage_at_k)}${compared?.metrics.evidence_coverage_at_k.delta == null ? "" : ` (${formatSigned(compared.metrics.evidence_coverage_at_k.delta)})`} · ${formatMetric(item.latency_ms, true)}</span></summary>
    <div class="diagnostic-grid"><div><h4>金标准证据</h4>${item.gold_evidence.length ? item.gold_evidence.map(ev => `<div class="gold-evidence"><b>${escapeHtml(ev.claim_group)}</b> · ¶${ev.start_paragraph}–${ev.end_paragraph}<p>${escapeHtml(ev.claim)}</p></div>`).join("") : "<p>不可回答题没有金标准证据。</p>"}</div>
    <div><h4>Top-K 候选</h4>${item.retrieval_candidates.map(candidate => `<details class="candidate ${candidate.relevant ? "relevant" : ""}"><summary>#${candidate.rank} ${escapeHtml(candidate.article_title || "未知文章")} · ${escapeHtml(candidate.topic_title || candidate.chunk_id)} ${candidate.relevant ? "✓" : ""}</summary><p class="help">¶${candidate.paragraph_range?.join("–") || "—"} · 来源 ${escapeHtml((candidate.source || []).join(" + "))}</p><pre>${escapeHtml(candidate.original_text || "该候选不是普通 TopicChunk，无法直接显示原文。")}</pre><pre>${escapeHtml(JSON.stringify(candidate.scores || {}, null, 2))}</pre></details>`).join("") || "<p>没有候选。</p>"}</div></div>
  </details>`; }).join("");
}
async function startEvaluation(event) {
  event.preventDefault();
  const split = document.getElementById("run-split").value;
  let confirmHoldout = false;
  if (split === "holdout") {
    confirmHoldout = window.confirm("holdout 应只用于阶段性最终验证，不能根据结果反复调参。确认运行？");
    if (!confirmHoldout) return;
  }
  const body = {dataset_id: document.getElementById("run-dataset").value, build_id: document.getElementById("run-build").value, split, mode: document.getElementById("run-mode").value, k: Number(document.getElementById("run-k").value), rerank: document.getElementById("run-rerank").checked, confirm_holdout: confirmHoldout};
  await api("/api/evaluation-runs", {method: "POST", body: JSON.stringify(body)});
  notice("检索评测正在本地运行；完成后会自动打开结果");
  pollEvaluationJob();
}
async function pollEvaluationJob() {
  const job = await api("/api/evaluation-job");
  renderEvaluationJob(job);
  if (job.status === "running") return setTimeout(pollEvaluationJob, 1500);
  await refresh();
  if (job.status === "succeeded") await loadRuns(job.run_id);
}
function renderEvaluationJob(job) {
  const status = document.getElementById("evaluation-status");
  status.textContent = job.status === "running" ? `运行中：${job.mode} · ${job.split} · K=${job.k}` : job.status === "failed" ? `失败：${job.error_message || job.error_type}` : job.status === "succeeded" ? `已完成：${job.run_id}` : "";
  status.classList.toggle("error", job.status === "failed");
  document.getElementById("run-submit").disabled = job.status === "running";
}
function formatMetric(value, latency = false) { return value == null ? "—" : latency ? `${Number(value).toFixed(1)} ms` : Number(value).toFixed(3); }
function formatSigned(value) { return `${value >= 0 ? "+" : ""}${Number(value).toFixed(3)}`; }
function formatDate(value) { return new Date(value).toLocaleString("zh-CN", {month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit"}); }

document.getElementById("dataset-form").addEventListener("submit", async event => {
  event.preventDefault();
  const result = await api("/api/datasets", {method: "POST", body: JSON.stringify({name: document.getElementById("dataset-name").value, version: document.getElementById("dataset-version").value, purpose: document.getElementById("dataset-purpose").value})});
  currentDataset = result.id; notice("draft 数据集已创建"); await refresh();
});
document.getElementById("dataset-list").addEventListener("change", event => { currentDataset = event.target.value; renderQuestions(); updateDatasetControls(); });
document.getElementById("run-dataset").addEventListener("change", async () => { renderCompatibleBuilds(); await loadRuns(); });
document.getElementById("run-mode").addEventListener("change", event => { if (event.target.value !== "hybrid") document.getElementById("run-rerank").checked = false; });
document.getElementById("run-form").addEventListener("submit", event => startEvaluation(event).catch(showError));
document.getElementById("run-list").addEventListener("change", event => loadRun(event.target.value).catch(showError));
document.getElementById("baseline-run").addEventListener("change", async () => { if (!currentRun) return; currentComparison = await renderComparison(); renderQuestionResults(currentRun.questions, currentComparison); });
document.getElementById("show-curation").addEventListener("click", () => switchView("curation"));
document.getElementById("show-results").addEventListener("click", () => { switchView("results"); loadRuns().catch(showError); });
document.getElementById("question-split").addEventListener("change", event => {
  document.getElementById("question-origin").value = event.target.value === "holdout" ? "new_holdout" : "new_dev";
});
document.getElementById("question-form").addEventListener("submit", async event => {
  event.preventDefault();
  const result = await api("/api/questions", {method: "POST", body: JSON.stringify({dataset_id: currentDataset, question_id: editingQuestionId, query: document.getElementById("question-query").value, version_id: document.getElementById("article-list").value, split: document.getElementById("question-split").value, difficulty: document.getElementById("question-difficulty").value, origin: document.getElementById("question-origin").value, answerable: document.getElementById("question-answerable").checked})});
  editingQuestionId = null; document.querySelector("#question-form button").textContent = "保存为新问题"; document.getElementById("question-query").value = ""; notice("问题已保存；修改题目会清空原人工答案和证据"); await refresh(); await loadQuestion(result.id);
});
document.getElementById("edit-question").addEventListener("click", () => {
  if ((currentDocument.question.reference_answer || points.length) && !window.confirm("编辑题目会在保存时清空现有人工答案和证据，继续？")) return;
  editingQuestionId = currentDocument.question.id;
  document.getElementById("question-query").value = currentDocument.question.query;
  document.getElementById("article-list").value = currentDocument.article.version_id;
  document.getElementById("question-split").value = currentDocument.question.split;
  document.getElementById("question-difficulty").value = currentDocument.question.difficulty;
  document.getElementById("question-origin").value = currentDocument.question.origin;
  document.getElementById("question-answerable").checked = currentDocument.question.answerable;
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
function switchView(name) {
  document.getElementById("curation-view").classList.toggle("hidden", name !== "curation");
  document.getElementById("results-view").classList.toggle("hidden", name !== "results");
  document.getElementById("show-curation").classList.toggle("active", name === "curation");
  document.getElementById("show-results").classList.toggle("active", name === "results");
}
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]); }
function escapeAttr(value) { return escapeHtml(value); }
window.addEventListener("unhandledrejection", event => showError(event.reason));
refresh().catch(showError);
