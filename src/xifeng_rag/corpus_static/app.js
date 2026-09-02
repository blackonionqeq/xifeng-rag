"use strict";

const token = new URLSearchParams(window.location.search).get("token") || "";
const state = { articles: [], stats: null, busy: false };
const statusLabels = {
  discovered: "已发现",
  needs_cleanup_review: "待清洗审核",
  pending_chunking: "待分块",
  pending_review: "待审核",
  approved: "已批准待建索引",
  indexed: "已索引",
  inactive: "来源缺失",
};

function request(path, options = {}) {
  return fetch(path, {
    ...options,
    headers: { ...(options.headers || {}), "X-Corpus-Token": token },
  });
}

function formatTime(value) {
  if (!value) return "—";
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  }).format(new Date(value));
}

function shortId(value) {
  if (!value) return "—";
  return value.length > 22 ? `${value.slice(0, 12)}…${value.slice(-6)}` : value;
}

function renderStats() {
  const entries = [
    [state.stats.active, "活动文章"],
    [state.stats.indexed, "已索引"],
    [state.stats.pending, "待处理"],
    [state.stats.needs_cleanup_review, "待清洗审核"],
    [state.stats.approved_overviews, "概括已批准"],
    [state.stats.pending_overviews, "概括待审核"],
    [`${state.stats.coverage_percent}%`, "索引覆盖率"],
  ];
  const container = document.getElementById("stats");
  const template = document.getElementById("stat-template");
  container.replaceChildren();
  entries.forEach(([value, label]) => {
    const fragment = template.content.cloneNode(true);
    fragment.querySelector("strong").textContent = value;
    fragment.querySelector("span").textContent = label;
    container.append(fragment);
  });
  document.getElementById("summary-line").textContent =
    `数据库共 ${state.stats.total} 篇 · 来源缺失 ${state.stats.missing} 篇`;
}

function matchesFilters(article) {
  const query = document.getElementById("search-input").value.trim().toLocaleLowerCase();
  const status = document.getElementById("status-filter").value;
  const index = document.getElementById("index-filter").value;
  const summary = article.overview ? article.overview.article_summary : "";
  if (query && !`${article.title} ${article.article_id} ${summary}`.toLocaleLowerCase().includes(query)) return false;
  if (status && article.latest_status !== status) return false;
  if (index === "indexed" && !article.is_indexed) return false;
  if (index === "unindexed" && article.is_indexed) return false;
  if (index === "pending-update" && !article.has_pending_update) return false;
  if (index === "missing" && article.source_active) return false;
  return true;
}

function sortedArticles() {
  const field = document.getElementById("sort-select").value;
  return state.articles.filter(matchesFilters).sort((left, right) => {
    const a = left[field] || "";
    const b = right[field] || "";
    if (field === "title" || field === "latest_status") return String(a).localeCompare(String(b), "zh-CN");
    if (!a) return 1;
    if (!b) return -1;
    return String(b).localeCompare(String(a));
  });
}

function statusBadge(article) {
  const wrapper = document.createElement("div");
  const badge = document.createElement("span");
  badge.className = `badge status-${article.latest_status}`;
  badge.textContent = statusLabels[article.latest_status] || article.latest_status;
  wrapper.append(badge);
  if (article.has_pending_update) {
    const update = document.createElement("span");
    update.className = "badge update";
    update.textContent = "有更新";
    wrapper.append(update);
  }
  if (!article.source_active) {
    const missing = document.createElement("span");
    missing.className = "badge missing";
    missing.textContent = "来源缺失";
    wrapper.append(missing);
  }
  return wrapper;
}

async function openReview(article, proposalId, tab) {
  const status = document.getElementById("operation-status");
  try {
    const response = await request("/api/reviews", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ version_id: article.latest_version_id, proposal_id: proposalId }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "审核页面启动失败");
    tab.location = payload.url;
    status.textContent = `${payload.message}：${article.title}`;
  } catch (error) {
    tab.close();
    status.textContent = error.message;
  }
}

function reviewAction(article) {
  const wrapper = document.createElement("div");
  wrapper.className = "review-action";
  if (article.pending_proposals.length) {
    const group = document.createElement("div");
    const select = document.createElement("select");
    select.setAttribute("aria-label", `选择 ${article.title} 的待审提案`);
    article.pending_proposals.forEach((proposal) => {
      const option = document.createElement("option");
      option.value = proposal.proposal_id;
      option.textContent = `${proposal.provider_id}/${proposal.model_id}`;
      option.title = `${proposal.prompt_version} · ${formatTime(proposal.created_at)}`;
      select.append(option);
    });
    const button = document.createElement("button");
    button.type = "button";
    button.className = "small";
    button.textContent = "审核分块";
    button.addEventListener("click", () => {
      const tab = window.open("about:blank", "_blank");
      if (!tab) {
        document.getElementById("operation-status").textContent = "浏览器阻止了新标签页，请允许弹窗后重试";
        return;
      }
      openReview(article, select.value, tab);
    });
    group.append(select, button);
    wrapper.append(group);
  }
  if (article.is_indexed) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "small secondary";
    button.textContent = article.subtopics?.status === "pending_review"
      ? "继续审核细话题"
      : article.subtopics?.status === "approved" ? "重新生成细话题" : "补充细话题";
    button.addEventListener("click", () => openSubtopicReview(article, button));
    wrapper.append(button);
    if (article.subtopics) {
      const meta = document.createElement("small");
      meta.textContent = `${article.subtopics.status === "approved" ? "已批准" : "待审核"} · ${article.subtopics.count} 个`;
      wrapper.append(meta);
    }
  }
  if (!wrapper.children.length) {
    wrapper.textContent = "—";
    wrapper.classList.add("muted");
  }
  return wrapper;
}

async function openSubtopicReview(article, button) {
  const status = document.getElementById("operation-status");
  button.disabled = true;
  const tab = window.open("about:blank", "_blank");
  if (!tab) {
    status.textContent = "浏览器阻止了新标签页，请允许弹窗后重试";
    button.disabled = false;
    return;
  }
  try {
    const disclosureResponse = await request("/api/subtopics/disclosure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ version_id: article.indexed_version_id }),
    });
    const disclosure = await disclosureResponse.json();
    if (!disclosureResponse.ok) throw new Error(disclosure.error || "无法读取外发范围");
    const message = disclosure.has_pending_draft
      ? `已有待审细话题草案，不会再次发送正文。\n\n文章：${disclosure.article_title}\n版本：${disclosure.version_id}\n\n继续审核？`
      : [
          `文章：${disclosure.article_title}`,
          `版本：${disclosure.version_id}`,
          `清洗后正文：${disclosure.character_count} 字符，${disclosure.unit_count} 个原子单元`,
          `外部服务：OpenAI Codex CLI（复用当前 ChatGPT 登录）`,
          `模型：${disclosure.model}`,
          `本次最多发送 ${disclosure.max_requests} 次完整清洗后正文。`,
          "训练使用取决于当前 ChatGPT 账户的数据控制设置。",
          "",
          "确认发送并生成细话题草案？",
        ].join("\n");
    if (!window.confirm(message)) {
      tab.close();
      status.textContent = "已取消，未发送正文。";
      return;
    }
    status.textContent = disclosure.has_pending_draft ? "正在启动细话题审核…" : "正在调用 Codex 生成细话题…";
    const response = await request("/api/subtopics/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ version_id: article.indexed_version_id, confirmed: true }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "细话题生成失败");
    tab.location = payload.url;
    status.textContent = `${payload.message}：${article.title}`;
    await loadArticles();
  } catch (error) {
    tab.close();
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

async function approveOverview(article, textarea, button) {
  const status = document.getElementById("operation-status");
  button.disabled = true;
  try {
    const response = await request("/api/overviews/approve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        overview_id: article.overview.overview_id,
        article_summary: textarea.value,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "文章概括批准失败");
    status.textContent = `${payload.message}：${article.title}`;
    await loadArticles();
  } catch (error) {
    status.textContent = error.message;
    button.disabled = false;
  }
}

function overviewCell(article) {
  const wrapper = document.createElement("div");
  wrapper.className = "overview-cell";
  if (!article.overview) {
    wrapper.classList.add("muted");
    wrapper.textContent = "尚未生成";
    return wrapper;
  }
  const textarea = document.createElement("textarea");
  textarea.value = article.overview.article_summary;
  textarea.readOnly = article.overview.status === "approved";
  textarea.setAttribute("aria-label", `${article.title}的文章概括`);
  const meta = document.createElement("small");
  meta.textContent = `${article.overview.status === "approved" ? "已批准" : "待审核"} · ${article.overview.provider_id}/${article.overview.model_id}`;
  wrapper.append(textarea, meta);
  if (article.overview.status === "pending_review") {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "small";
    button.textContent = "批准概括";
    button.addEventListener("click", () => approveOverview(article, textarea, button));
    wrapper.append(button);
  }
  return wrapper;
}

function renderRows() {
  const rows = document.getElementById("article-rows");
  rows.replaceChildren();
  const filtered = sortedArticles();
  filtered.forEach((article) => {
    const row = document.createElement("tr");
    const title = document.createElement("td");
    title.innerHTML = `<strong></strong><small></small>`;
    title.querySelector("strong").textContent = article.title;
    title.querySelector("small").textContent = article.article_id;
    const captured = document.createElement("td");
    const published = document.createElement("span");
    published.textContent = `发布 ${formatTime(article.published_at)}`;
    const capturedTime = document.createElement("span");
    capturedTime.textContent = `抓取 ${formatTime(article.captured_at)}`;
    captured.append(published, capturedTime);
    const modified = document.createElement("td");
    modified.textContent = formatTime(article.source_modified_at);
    const status = document.createElement("td");
    status.append(statusBadge(article));
    const proposals = document.createElement("td");
    proposals.textContent = `${article.pending_proposals.length} 待审 / ${article.proposal_count}`;
    const chunks = document.createElement("td");
    chunks.textContent = article.is_indexed ? String(article.indexed_chunk_count) : "—";
    const versions = document.createElement("td");
    versions.className = "version-cell";
    const latestVersion = document.createElement("span");
    latestVersion.title = article.latest_version_id || "";
    latestVersion.textContent = `最新 ${shortId(article.latest_version_id)}`;
    const indexedVersion = document.createElement("span");
    indexedVersion.title = article.indexed_version_id || "";
    indexedVersion.textContent = `索引 ${shortId(article.indexed_version_id)}`;
    versions.append(latestVersion, indexedVersion);
    const overview = document.createElement("td");
    overview.append(overviewCell(article));
    const action = document.createElement("td");
    action.append(reviewAction(article));
    row.append(title, captured, modified, status, proposals, chunks, versions, overview, action);
    rows.append(row);
  });
  document.getElementById("empty-state").hidden = filtered.length > 0;
}

function populateStatusFilter() {
  const select = document.getElementById("status-filter");
  const selected = select.value;
  const statuses = [...new Set(state.articles.map((article) => article.latest_status))].sort();
  select.replaceChildren(new Option("全部状态", ""));
  statuses.forEach((status) => select.append(new Option(statusLabels[status] || status, status)));
  select.value = selected;
}

async function loadArticles() {
  const status = document.getElementById("operation-status");
  try {
    const response = await request("/api/articles");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "列表加载失败");
    state.articles = payload.articles;
    state.stats = payload.stats;
    renderStats();
    populateStatusFilter();
    renderRows();
  } catch (error) {
    status.textContent = error.message;
  }
}

async function scanCorpus() {
  if (state.busy) return;
  state.busy = true;
  const button = document.getElementById("scan-button");
  const includeModified = document.getElementById("include-modified");
  const status = document.getElementById("operation-status");
  button.disabled = true;
  includeModified.disabled = true;
  button.textContent = "正在扫描…";
  status.textContent = "正在只读扫描规范 Markdown 并更新本地状态…";
  try {
    const response = await request("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ include_modified: includeModified.checked }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "扫描失败");
    status.textContent = `扫描完成：发现 ${payload.discovered}，排除 ${payload.excluded}，新增 ${payload.added}，已导入正文更新 ${payload.modified}，已跳过正文更新 ${payload.skipped_modified}，元数据更新 ${payload.metadata_updated}，未变化 ${payload.unchanged}，缺失 ${payload.missing}，待清洗审核 ${payload.needs_cleanup_review}`;
    await loadArticles();
  } catch (error) {
    status.textContent = error.message;
  } finally {
    state.busy = false;
    button.disabled = false;
    includeModified.disabled = false;
    button.textContent = "扫描语料更新";
  }
}

async function shutdownServer() {
  const button = document.getElementById("shutdown-button");
  const status = document.getElementById("operation-status");
  button.disabled = true;
  try {
    const response = await request("/api/shutdown", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "关闭失败");
    document.querySelector("main").hidden = true;
    document.querySelector(".header-actions").hidden = true;
    document.getElementById("summary-line").textContent = payload.message;
  } catch (error) {
    button.disabled = false;
    status.textContent = error.message;
  }
}

["search-input", "status-filter", "index-filter", "sort-select"].forEach((id) => {
  document.getElementById(id).addEventListener("input", renderRows);
  document.getElementById(id).addEventListener("change", renderRows);
});
document.getElementById("scan-button").addEventListener("click", scanCorpus);
document.getElementById("refresh-button").addEventListener("click", loadArticles);
document.getElementById("shutdown-button").addEventListener("click", shutdownServer);
loadArticles();
