const state = {
  overview: null,
  queryIndex: 0,
  position: 0,
  targetOffset: 0,
  pageSize: 5,
  entry: null,
};

const elements = {
  testSize: document.querySelector("#test-size"),
  querySelect: document.querySelector("#query-select"),
  sourceInput: document.querySelector("#source-input"),
  jumpButton: document.querySelector("#jump-button"),
  previousButton: document.querySelector("#previous-button"),
  nextButton: document.querySelector("#next-button"),
  positionLabel: document.querySelector("#position-label"),
  queryChips: document.querySelector("#query-chips"),
  mappingLabel: document.querySelector("#mapping-label"),
  targetCount: document.querySelector("#target-count"),
  sourceGallery: document.querySelector("#source-gallery"),
  targetGallery: document.querySelector("#target-gallery"),
  targetSlider: document.querySelector("#target-slider"),
  targetRangeLabel: document.querySelector("#target-range-label"),
  previousTargets: document.querySelector("#previous-targets"),
  nextTargets: document.querySelector("#next-targets"),
  sameIdentityButton: document.querySelector("#same-identity-button"),
  matrix: document.querySelector("#attribute-matrix"),
  errorBox: document.querySelector("#error-box"),
  cardTemplate: document.querySelector("#image-card-template"),
};

async function getJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Richiesta fallita: ${response.status}`);
  }
  return payload;
}

function showError(error) {
  elements.errorBox.textContent = error.message;
  elements.errorBox.hidden = false;
  window.setTimeout(() => {
    elements.errorBox.hidden = true;
  }, 5000);
}

async function initialize() {
  state.overview = await getJson("/api/overview");
  elements.testSize.textContent = state.overview.test_size.toLocaleString("it-IT");
  state.overview.queries.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.index;
    option.textContent = `${item.index}. ${item.query} · ${item.source_count.toLocaleString("it-IT")} sorgenti`;
    elements.querySelect.append(option);
  });
  bindEvents();
  await loadEntry();
}

function bindEvents() {
  elements.querySelect.addEventListener("change", async (event) => {
    state.queryIndex = Number(event.target.value);
    state.position = 0;
    state.targetOffset = 0;
    await loadEntry();
  });

  elements.previousButton.addEventListener("click", () => move(-1));
  elements.nextButton.addEventListener("click", () => move(1));
  elements.jumpButton.addEventListener("click", jumpToSource);
  elements.sourceInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") jumpToSource();
  });
  elements.previousTargets.addEventListener("click", () => moveTargets(-state.pageSize));
  elements.nextTargets.addEventListener("click", () => moveTargets(state.pageSize));
  elements.targetSlider.addEventListener("input", (event) => {
    state.targetOffset = Number(event.target.value);
    renderTargetWindow();
  });
  elements.sameIdentityButton.addEventListener("click", jumpToSameIdentity);

  window.addEventListener("keydown", (event) => {
    if (document.activeElement.matches("input, select")) return;
    if (event.key === "ArrowLeft") move(-1);
    if (event.key === "ArrowRight") move(1);
  });
}

async function move(delta) {
  if (!state.entry) return;
  state.position = (state.entry.source_position + delta + state.entry.source_count) % state.entry.source_count;
  await loadEntry();
}

async function jumpToSource() {
  const source = Number(elements.sourceInput.value);
  if (!Number.isInteger(source)) return;
  await loadEntry(source);
}

function moveTargets(delta) {
  if (!state.entry) return;
  const maxOffset = Math.max(0, state.entry.targets.length - state.pageSize);
  state.targetOffset = Math.min(maxOffset, Math.max(0, state.targetOffset + delta));
  renderTargetWindow();
}

function jumpToSameIdentity() {
  const firstRank = state.entry?.same_identity_target_ranks[0];
  if (!firstRank) return;
  state.targetOffset = Math.floor((firstRank - 1) / state.pageSize) * state.pageSize;
  renderTargetWindow();
}

async function loadEntry(sourceIndex = null) {
  try {
    document.body.classList.add("loading");
    const params = new URLSearchParams({ query: state.queryIndex });
    if (sourceIndex === null) {
      params.set("position", state.position);
    } else {
      params.set("source", sourceIndex);
    }
    state.entry = await getJson(`/api/entry?${params}`);
    state.position = state.entry.source_position;
    state.targetOffset = 0;
    render();
  } catch (error) {
    showError(error);
  } finally {
    document.body.classList.remove("loading");
  }
}

function render() {
  const entry = state.entry;
  elements.sourceInput.value = entry.source.dataset_index;
  elements.positionLabel.textContent = `${entry.source_position + 1} / ${entry.source_count}`;
  elements.mappingLabel.textContent = `${entry.source.dataset_index} → ${entry.source.filename}`;
  const sameIdentityText = entry.same_identity_target_count
    ? `${entry.same_identity_target_count} della stessa persona`
    : "nessun target della stessa persona";
  elements.targetCount.textContent = `${entry.available_target_count} target validi · ${sameIdentityText}`;
  renderQueryChips(entry.modifiers);
  renderSource(entry);
  configureTargetNavigation(entry);
  renderTargetWindow();
}

function renderQueryChips(modifiers) {
  elements.queryChips.replaceChildren();
  modifiers.forEach((modifier) => {
    const chip = document.createElement("span");
    chip.className = `chip ${modifier.operator === "+" ? "positive" : "negative"}`;
    chip.textContent = `${modifier.operator}${humanize(modifier.attribute)}`;
    elements.queryChips.append(chip);
  });
}

function renderSource(entry) {
  elements.sourceGallery.replaceChildren(createImageCard(entry.source, entry, true, 0));
}

function configureTargetNavigation(entry) {
  const maxOffset = Math.max(0, entry.targets.length - state.pageSize);
  elements.targetSlider.max = maxOffset;
  elements.targetSlider.value = state.targetOffset;
  elements.sameIdentityButton.disabled = entry.same_identity_target_count === 0;
  elements.sameIdentityButton.textContent = entry.same_identity_target_count
    ? `Trova stesso ID (${entry.same_identity_target_count})`
    : "Nessuno stesso ID";
}

function visibleTargets() {
  return state.entry.targets.slice(state.targetOffset, state.targetOffset + state.pageSize);
}

function renderTargetWindow() {
  const entry = state.entry;
  if (!entry) return;
  const records = visibleTargets();
  elements.targetGallery.replaceChildren();
  records.forEach((record, index) => {
    elements.targetGallery.append(createImageCard(record, entry, false, index));
  });
  const first = records.length ? state.targetOffset + 1 : 0;
  const last = state.targetOffset + records.length;
  elements.targetRangeLabel.textContent = `Target ${first}–${last} di ${entry.targets.length}`;
  elements.targetSlider.value = state.targetOffset;
  elements.previousTargets.disabled = state.targetOffset === 0;
  elements.nextTargets.disabled = last >= entry.targets.length;
  renderMatrix(entry, records);
}

function createImageCard(record, entry, isSource, animationIndex) {
    const fragment = elements.cardTemplate.content.cloneNode(true);
    const card = fragment.querySelector(".image-card");
    card.style.animationDelay = `${animationIndex * 45}ms`;
    if (isSource) card.classList.add("source-card");
    if (!isSource && record.identity_id === entry.source.identity_id) {
      card.classList.add("same-identity-card");
    }

    fragment.querySelector(".card-role").textContent = isSource ? "Sorgente" : `Target valido #${record.rank}`;
    fragment.querySelector(".card-index").textContent = `index ${record.dataset_index}`;
    const identityBadge = fragment.querySelector(".identity-badge");
    identityBadge.textContent = record.identity_id ? `ID persona ${record.identity_id}` : "ID n/d";
    if (!isSource && record.identity_id === entry.source.identity_id) {
      identityBadge.textContent = `STESSA PERSONA · ID ${record.identity_id}`;
      identityBadge.classList.add("same-identity-badge");
    }
    fragment.querySelector(".filename").textContent = record.filename;

    const image = fragment.querySelector(".face-image");
    image.src = record.image_url;
    image.alt = `${isSource ? "Sorgente" : "Target"} CelebA, dataset index ${record.dataset_index}`;

    renderQuality(fragment.querySelector(".quality-row"), record, isSource);
    renderActiveAttributes(fragment.querySelector(".active-attributes"), record, entry.modifiers);
    return fragment;
}

function renderQuality(container, record, isSource) {
  if (isSource) {
    const chip = document.createElement("span");
    chip.className = "metric-chip";
    chip.textContent = "riferimento";
    container.append(chip);
    return;
  }

  const querySatisfied = record.query_checks.every((check) => check.satisfied);
  const queryChip = document.createElement("span");
  queryChip.className = `metric-chip ${querySatisfied ? "good" : "warn"}`;
  queryChip.textContent = querySatisfied ? "query ✓" : "query ✗";
  container.append(queryChip);

  const distanceChip = document.createElement("span");
  distanceChip.className = `metric-chip ${record.hamming_distance <= 2 ? "good" : "warn"}`;
  distanceChip.textContent = `dH resto = ${record.hamming_distance}`;
  distanceChip.title = record.differing_attributes.length
    ? `Differenze: ${record.differing_attributes.map(humanize).join(", ")}`
    : "Nessuna differenza sugli attributi non interrogati";
  container.append(distanceChip);
}

function renderActiveAttributes(container, record, modifiers) {
  const modifierByAttribute = new Map(modifiers.map((item) => [item.attribute, item]));
  record.active_attributes.forEach((attribute) => {
    const chip = document.createElement("span");
    const modifier = modifierByAttribute.get(attribute);
    chip.className = "attribute-chip";
    if (modifier?.operator === "+") chip.classList.add("query-positive");
    chip.textContent = humanize(attribute);
    container.append(chip);
  });

  modifiers
    .filter((modifier) => modifier.operator === "-" && record.attribute_values[modifier.attribute] === -1)
    .forEach((modifier) => {
      const chip = document.createElement("span");
      chip.className = "attribute-chip query-negative";
      chip.textContent = `no ${humanize(modifier.attribute)}`;
      container.append(chip);
    });
}

function renderMatrix(entry, targets) {
  const table = document.createElement("table");
  table.className = "attribute-table";
  const modifierByAttribute = new Map(entry.modifiers.map((item) => [item.attribute, item]));

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  const imageHeader = document.createElement("th");
  imageHeader.textContent = "Immagine";
  headRow.append(imageHeader);

  state.overview.attribute_names.forEach((attribute) => {
    const th = document.createElement("th");
    const modifier = modifierByAttribute.get(attribute);
    if (modifier) th.classList.add(modifier.operator === "+" ? "query-positive" : "query-negative");
    const label = document.createElement("span");
    label.textContent = humanize(attribute);
    th.append(label);
    headRow.append(th);
  });
  thead.append(headRow);
  table.append(thead);

  const tbody = document.createElement("tbody");
  [entry.source, ...targets].forEach((record, index) => {
    const row = document.createElement("tr");
    const label = document.createElement("td");
    label.textContent = `${index === 0 ? "Sorgente" : `Target ${record.rank}`} · ${record.dataset_index}`;
    row.append(label);

    state.overview.attribute_names.forEach((attribute) => {
      const value = record.attribute_values[attribute];
      const cell = document.createElement("td");
      const modifier = modifierByAttribute.get(attribute);
      if (modifier) cell.classList.add(modifier.operator === "+" ? "query-positive" : "query-negative");
      cell.classList.add(value === 1 ? "value-on" : "value-off");
      cell.textContent = value === 1 ? "+1" : "−1";
      row.append(cell);
    });
    tbody.append(row);
  });
  table.append(tbody);
  elements.matrix.replaceChildren(table);
}

function humanize(value) {
  return value.replaceAll("_", " ");
}

initialize().catch(showError);
