function existing_tickers() {
  const chips = document.querySelectorAll("#chipContainer span[data-ticker]");
  return Array.from(chips).map(c => c.dataset.ticker);
}

function update_submit_state() {
  const disabled = existing_tickers().length === 0;
  document.getElementById("compareBtn").disabled = disabled;
  document.getElementById("exportBtn").disabled = disabled;
}

function add_chip(ticker) {
  const container = document.getElementById("chipContainer");

  const chip = document.createElement("span");
  chip.className = "badge text-bg-secondary font-monospace d-inline-flex align-items-center gap-1";
  chip.dataset.ticker = ticker;
  chip.textContent = ticker + " ";

  const remove_btn = document.createElement("button");
  remove_btn.type = "button";
  remove_btn.className = "btn-close btn-close-white chip-remove";
  remove_btn.setAttribute("aria-label", "Remove");
  chip.appendChild(remove_btn);

  const hidden = document.createElement("input");
  hidden.type = "hidden";
  hidden.name = "tickers";
  hidden.value = ticker;
  hidden.dataset.ticker = ticker;

  container.appendChild(chip);
  container.appendChild(hidden);
  update_submit_state();
}

function remove_chip(ticker) {
  document.querySelectorAll(`[data-ticker="${ticker}"]`).forEach(el => el.remove());
  update_submit_state();
}

function add_from_input() {
  const input = document.getElementById("tickerInput");
  const ticker = input.value.trim().toUpperCase();
  if (ticker && !existing_tickers().includes(ticker)) {
    add_chip(ticker);
  }
  input.value = "";
}

document.getElementById("addBtn").addEventListener("click", add_from_input);

document.getElementById("tickerInput").addEventListener("keydown", e => {
  if (e.key === "Enter") {
    e.preventDefault();
    add_from_input();
  }
});

document.getElementById("chipContainer").addEventListener("click", e => {
  if (e.target.classList.contains("chip-remove")) {
    const ticker = e.target.closest("[data-ticker]").dataset.ticker;
    remove_chip(ticker);
  }
});

update_submit_state();
