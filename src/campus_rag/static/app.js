const form = document.querySelector("#question-form");
const input = document.querySelector("#question");
const submitButton = document.querySelector("#submit-button");
const conversation = document.querySelector("#conversation");
const template = document.querySelector("#message-template");

function addMessage(label, content, type = "assistant", evidence = []) {
  const node = template.content.firstElementChild.cloneNode(true);
  node.classList.add(type);
  node.querySelector(".message-label").textContent = label;
  node.querySelector(".message-content").textContent = content;

  if (evidence.length) {
    const details = document.createElement("details");
    details.className = "sources";
    const summary = document.createElement("summary");
    summary.textContent = `查看检索证据（${evidence.length} 条）`;
    details.append(summary);
    evidence.forEach((item) => {
      const card = document.createElement("section");
      card.className = "source-card";
      const meta = document.createElement("p");
      meta.className = "source-meta";
      meta.textContent = `${item.source} · ${item.context || "未标注章节"}`;
      const text = document.createElement("p");
      text.className = "source-text";
      text.textContent = item.text;
      card.append(meta, text);
      details.append(card);
    });
    node.append(details);
  }
  conversation.append(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
}

async function ask(question) {
  addMessage("你", question, "user");
  submitButton.disabled = true;
  submitButton.textContent = "回答中…";
  try {
    const response = await fetch("/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 3 }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "请求失败，请稍后再试。");
    addMessage("知识库助手", payload.answer, "assistant", payload.evidence);
  } catch (error) {
    addMessage("请求失败", error.message, "error");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "发送问题";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const questions = input.value.split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
  if (!questions.length) return;
  input.value = "";
  for (const question of questions) {
    await ask(question);
  }
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.question;
    form.requestSubmit();
  });
});
