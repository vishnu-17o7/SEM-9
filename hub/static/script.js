/* ── SEM 9 Hub — Run button + SSE output ── */

function startRun(btn) {
  const projectId = btn.dataset.project;
  const script = btn.dataset.script;
  const args = btn.dataset.args || "";
  const outputId = `output-${projectId}`;
  const outputEl = document.getElementById(outputId);
  if (!outputEl) return;

  /* disable all run buttons in this section */
  const section = btn.closest(".run-section");
  const buttons = section.querySelectorAll(".btn--primary");
  buttons.forEach(b => b.disabled = true);

  outputEl.textContent = "";
  outputEl.scrollTop = 0;

  const url = `/api/run/${projectId}?script=${encodeURIComponent(script)}` +
    (args ? `&args=${encodeURIComponent(args)}` : "");

  const evtSource = new EventSource(url);

  evtSource.onmessage = function (e) {
    try {
      const data = JSON.parse(e.data);
      const span = document.createElement("span");
      span.className = data.stream || "stdout";
      span.textContent = data.text;
      outputEl.appendChild(span);
      outputEl.scrollTop = outputEl.scrollHeight;
    } catch (_) {
      /* ignore parse errors */
    }
  };

  evtSource.addEventListener("done", function () {
    evtSource.close();
    buttons.forEach(b => b.disabled = false);
  });

  evtSource.onerror = function () {
    evtSource.close();
    buttons.forEach(b => b.disabled = false);
    const span = document.createElement("span");
    span.className = "stderr";
    span.textContent = "\n[error] Connection closed\n";
    outputEl.appendChild(span);
  };
}

function stopRun(projectId) {
  const outputId = `output-${projectId}`;
  const outputEl = document.getElementById(outputId);
  if (!outputEl) return;
  outputEl.textContent = "";
}
