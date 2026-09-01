"use strict";

const token = new URLSearchParams(window.location.search).get("token") || "";
const state = {
  document: null,
  articleSummary: null,
  topics: [],
  locatedTopicIndex: null,
  draftSaved: false,
  approved: false,
  indexed: false,
  requestInFlight: false,
};
const topicColors = ["#687d72", "#88705d", "#60788d", "#826876", "#7d794f", "#597c7e"];

function topicAt(number) {
  return state.topics.findIndex((topic) => number >= topic.start_paragraph && number <= topic.end_paragraph);
}

function syncFields() {
  if (state.articleSummary !== null) {
    state.articleSummary = document.getElementById("article-summary").value;
  }
  document.querySelectorAll(".topic-card").forEach((card, index) => {
    const topic = state.topics[index];
    topic.title = card.querySelector(".topic-title").value;
    topic.summary = card.querySelector(".topic-summary").value;
    topic.keywords = card.querySelector(".topic-keywords").value
      .split(/[,，]/).map((item) => item.trim()).filter(Boolean);
    topic.include_in_index = card.querySelector(".topic-index").checked;
  });
}

function toggleBoundary(number) {
  syncFields();
  const boundaryIndex = state.topics.findIndex((topic) => topic.end_paragraph === number);
  if (boundaryIndex >= 0 && boundaryIndex < state.topics.length - 1) {
    state.topics[boundaryIndex].end_paragraph = state.topics[boundaryIndex + 1].end_paragraph;
    state.topics.splice(boundaryIndex + 1, 1);
    state.locatedTopicIndex = boundaryIndex;
  } else {
    const index = topicAt(number);
    const topic = state.topics[index];
    if (!topic || number < topic.start_paragraph || number >= topic.end_paragraph) return;
    const next = {
      start_paragraph: number + 1,
      end_paragraph: topic.end_paragraph,
      title: "新话题",
      summary: "请补充概括",
      keywords: ["待补充"],
      include_in_index: topic.include_in_index,
    };
    topic.end_paragraph = number;
    state.topics.splice(index + 1, 0, next);
    state.locatedTopicIndex = index;
  }
  markDirty();
  render();
}

function mergeWithPrevious(index) {
  if (index <= 0) return;
  syncFields();
  state.topics[index - 1].end_paragraph = state.topics[index].end_paragraph;
  state.topics.splice(index, 1);
  state.locatedTopicIndex = index - 1;
  markDirty();
  render();
}

function markDirty() {
  if (state.approved) return;
  state.draftSaved = false;
  document.getElementById("save-status").textContent = "有尚未保存的修改";
  updateActions();
}

function updateActions() {
  const save = document.getElementById("save-button");
  const approve = document.getElementById("approve-button");
  save.disabled = state.approved || state.requestInFlight;
  approve.disabled = !state.draftSaved || state.indexed || state.requestInFlight;
  if (state.indexed) {
    approve.textContent = "已批准并更新索引";
  } else if (state.approved) {
    approve.textContent = "重试更新索引";
  } else {
    approve.textContent = "批准并更新索引";
  }
}

function lockEditing() {
  document.querySelectorAll(
    "#article-summary, .topic-title, .topic-summary, .topic-keywords, .topic-index, .merge-button, .boundary"
  ).forEach((element) => { element.disabled = true; });
}

function showCompletion(payload) {
  document.getElementById("review-actions").hidden = true;
  document.getElementById("review-main").hidden = true;
  document.getElementById("completion-included").textContent = `${payload.included_count} 块`;
  document.getElementById("completion-excluded").textContent = `${payload.excluded_count} 块`;
  document.getElementById("completion-manifest").textContent = payload.manifest_id;
  document.getElementById("completion-index").textContent = payload.index_id;
  if (payload.article_index_id) {
    document.getElementById("completion-article-index").textContent = payload.article_index_id;
    document.getElementById("completion-article-index-row").hidden = false;
  }
  document.getElementById("completion-panel").hidden = false;
}

function locateTopic(index) {
  syncFields();
  state.locatedTopicIndex = index;
  renderParagraphs();
  renderTopics();
  const topic = state.topics[index];
  const first = document.querySelector(`[data-paragraph="${topic.start_paragraph}"]`);
  first?.scrollIntoView({ behavior: "smooth", block: "center" });
}

function renderParagraphs() {
  const container = document.getElementById("paragraphs");
  container.replaceChildren();
  const lastNumber = state.document.paragraphs.length;
  state.document.paragraphs.forEach((paragraph) => {
    const index = topicAt(paragraph.number);
    const wrapper = document.createElement("div");
    wrapper.className = "paragraph-wrapper";
    wrapper.dataset.paragraph = String(paragraph.number);
    wrapper.style.setProperty("--accent", topicColors[index % topicColors.length]);
    if (index === state.locatedTopicIndex) wrapper.classList.add("located");
    const row = document.createElement("div");
    row.className = "paragraph";
    const number = document.createElement("span");
    number.className = "paragraph-number";
    number.textContent = String(paragraph.number);
    const text = document.createElement("p");
    text.textContent = paragraph.text;
    row.append(number, text);
    wrapper.append(row);
    if (paragraph.number < lastNumber) {
      const boundary = document.createElement("button");
      const isBoundary = state.topics.some((topic) => topic.end_paragraph === paragraph.number);
      boundary.type = "button";
      boundary.className = `boundary ${isBoundary ? "active" : ""}`;
      boundary.textContent = isBoundary ? "移除此边界" : "在此分块";
      boundary.addEventListener("click", () => toggleBoundary(paragraph.number));
      wrapper.append(boundary);
    }
    container.append(wrapper);
  });
}

function renderTopics() {
  const container = document.getElementById("topics");
  const template = document.getElementById("topic-template");
  container.replaceChildren();
  state.topics.forEach((topic, index) => {
    const fragment = template.content.cloneNode(true);
    fragment.querySelector(".topic-number").textContent = `第 ${index + 1} 块`;
    fragment.querySelector(".topic-range").textContent = `${topic.start_paragraph}–${topic.end_paragraph}`;
    fragment.querySelector(".topic-title").value = topic.title;
    fragment.querySelector(".topic-summary").value = topic.summary;
    fragment.querySelector(".topic-keywords").value = topic.keywords.join("，");
    fragment.querySelector(".topic-index").checked = topic.include_in_index;
    const locate = fragment.querySelector(".locate-button");
    locate.setAttribute("aria-pressed", String(index === state.locatedTopicIndex));
    locate.addEventListener("click", () => locateTopic(index));
    const merge = fragment.querySelector(".merge-button");
    merge.hidden = index === 0;
    merge.addEventListener("click", () => mergeWithPrevious(index));
    container.append(fragment);
  });
  document.getElementById("topic-count").textContent = `${state.topics.length} 个话题块`;
}

function render() {
  renderParagraphs();
  renderTopics();
}

async function loadReview() {
  const response = await fetch("/api/review", { headers: { "X-Review-Token": token } });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "加载失败");
  state.document = payload;
  state.topics = structuredClone(payload.topics);
  if (typeof payload.article_summary === "string") {
    state.articleSummary = payload.article_summary;
    document.getElementById("article-summary").value = state.articleSummary;
    document.getElementById("summary-count").textContent = String(state.articleSummary.length);
    document.getElementById("overview-editor").hidden = false;
  }
  document.getElementById("article-title").textContent = payload.article_title;
  document.getElementById("metadata").textContent =
    `${payload.provider_id}/${payload.model_id} · ${payload.article_version} · ${payload.paragraphs.length} 个单元`;
  render();
}

async function saveDraft() {
  syncFields();
  const button = document.getElementById("save-button");
  const status = document.getElementById("save-status");
  state.requestInFlight = true;
  updateActions();
  status.textContent = "正在保存…";
  try {
    const response = await fetch("/api/draft", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Review-Token": token },
      body: JSON.stringify({
        topics: state.topics,
        article_summary: state.articleSummary,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "保存失败");
    state.draftSaved = true;
    status.textContent = `${payload.message}：${payload.path}`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    state.requestInFlight = false;
    updateActions();
  }
}

async function approveAndIndex() {
  syncFields();
  const included = state.topics.filter((topic) => topic.include_in_index).length;
  const excluded = state.topics.length - included;
  const prompt = [
    `总分块：${state.topics.length}`,
    `进入索引：${included}`,
    `排除索引：${excluded}`,
    ...(state.articleSummary !== null ? ["文章整体概括：一起批准"] : []),
    "批准后清单不可修改。确认批准并更新活动索引？",
  ].join("\n");
  if (!window.confirm(prompt)) return;
  const status = document.getElementById("save-status");
  state.requestInFlight = true;
  updateActions();
  status.textContent = state.approved ? "正在重试更新索引…" : "正在批准并更新索引…";
  try {
    const response = await fetch("/api/approve-and-index", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Review-Token": token },
      body: JSON.stringify({ confirmed: true }),
    });
    const payload = await response.json();
    if (!response.ok) {
      if (payload.approval_status === "approved") {
        state.approved = true;
        lockEditing();
        throw new Error(`清单 ${payload.manifest_id} 已批准，但${payload.error}`);
      }
      throw new Error(payload.error || "批准失败");
    }
    state.approved = true;
    state.indexed = true;
    lockEditing();
    status.textContent =
      `${payload.message}：${payload.manifest_id} · ${payload.index_id} · ${payload.chunk_count} 块` +
      (payload.article_index_id ? ` · ${payload.article_index_id}` : "");
    showCompletion(payload);
  } catch (error) {
    status.textContent = error.message;
  } finally {
    state.requestInFlight = false;
    updateActions();
  }
}

async function shutdownReviewServer() {
  const button = document.getElementById("shutdown-button");
  const status = document.getElementById("shutdown-status");
  button.disabled = true;
  status.textContent = "正在关闭审核服务…";
  try {
    const response = await fetch("/api/shutdown", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Review-Token": token },
      body: JSON.stringify({}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "关闭审核服务失败");
    button.hidden = true;
    status.textContent = payload.message;
  } catch (error) {
    button.disabled = false;
    status.textContent = error.message;
  }
}

document.getElementById("save-button").addEventListener("click", saveDraft);
document.getElementById("approve-button").addEventListener("click", approveAndIndex);
document.getElementById("shutdown-button").addEventListener("click", shutdownReviewServer);
document.getElementById("topics").addEventListener("input", markDirty);
document.getElementById("topics").addEventListener("change", markDirty);
document.getElementById("article-summary").addEventListener("input", (event) => {
  document.getElementById("summary-count").textContent = String(event.target.value.length);
  markDirty();
});
loadReview().catch((error) => {
  document.getElementById("article-title").textContent = "审核页面加载失败";
  document.getElementById("save-status").textContent = error.message;
});
