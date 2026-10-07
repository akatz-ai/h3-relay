import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "H3RelayPersonRemover";
const mounts = new Set();
const rootGraph = () => app.rootGraph ?? app.graph;
const workflowId = () => String(rootGraph()?.id ?? "");
const viewUrl = (file) => api.apiURL(`/view?${new URLSearchParams(file)}`);

function element(tag, className, text) {
    const el = document.createElement(tag);
    el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
}

function mount(node) {
    if (node._h3RemovalPreview) return;
    const root = element("div", "h3-removal-preview");
    const style = document.createElement("style");
    style.textContent = `
      .h3-removal-preview{font:12px/1.45 system-ui,sans-serif;color:#e3ebe7;background:#18201e;
        border:1px solid #40554b;border-radius:8px;padding:10px;box-sizing:border-box;
        width:100%;height:100%;overflow:auto;pointer-events:auto}
      .h3-removal-preview *{box-sizing:border-box}
      .h3-removal-status{display:flex;align-items:center;justify-content:space-between;gap:8px;margin:8px 0}
      .h3-removal-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
      .h3-removal-card{border:1px solid #41564c;border-radius:6px;overflow:hidden;background:#101613}
      .h3-removal-card button{display:block;border:0;padding:0;width:100%;background:#101613;cursor:pointer}
      .h3-removal-card video{display:block;width:100%;aspect-ratio:2/1;object-fit:contain;pointer-events:none}
      .h3-removal-caption{padding:6px 8px;font-size:11px;color:#b8c9bf}
      .h3-removal-caption strong{display:block;color:#e3ebe7;font-weight:600}
      .h3-removal-empty{grid-column:1/-1;padding:25px 12px;text-align:center;color:#a1b5a9;
        border:1px dashed #40554b;border-radius:6px}
      .h3-removal-hint{color:#9caea4;margin-top:8px;font-size:11px}
    `;
    const status = element("div", "h3-removal-status");
    const progress = element("span", "", "Window previews");
    status.append(progress);
    const grid = element("div", "h3-removal-grid");
    grid.append(element("div", "h3-removal-empty", "Each completed window appears here before the next one starts."));
    root.append(style, status, grid, element("div", "h3-removal-hint",
        "Hover to play · tap on touch screens. Preview clips are temporary; final output is saved separately."));
    const state = { node, root, progress, grid, run: null, revision: 0, cards: new Map() };
    node._h3RemovalPreview = state;
    mounts.add(state);
    const widget = node.addDOMWidget("window_previews", "div", root, {
        serialize: false, hideOnZoom: false,
        getMinHeight: () => 230, getMaxHeight: () => 470,
    });
    widget.serialize = false;
    node.setSize([Math.max(node.size[0], 550), Math.max(node.size[1], 800)]);
    const removed = node.onRemoved;
    node.onRemoved = function () {
        mounts.delete(state);
        for (const video of root.querySelectorAll("video")) { video.pause(); video.removeAttribute("src"); video.load(); }
        return removed?.apply(this, arguments);
    };
    const configured = node.onConfigure;
    node.onConfigure = function () {
        const result = configured?.apply(this, arguments);
        node.setSize([Math.max(node.size[0], 550), Math.max(node.size[1], 800)]);
        window.setTimeout(() => refresh(state), 100);
        return result;
    };
    const executed = node.onExecuted;
    node.onExecuted = function (message) {
        const result = executed?.apply(this, arguments);
        // ComfyUI replays this UI metadata when a completed render is cached.
        // Restore its own cards, even if another seed/window size ran afterwards.
        const runId = message?.h3_removal_run?.[0];
        if (typeof runId === "string") {
            node.properties.h3_removal_preview_run = runId;
            void refresh(state);
        }
        return result;
    };
    window.setTimeout(() => refresh(state), 300);
}

function render(state, run) {
    if (!run || String(run.node_id) !== String(state.node.id) || run.workflow_id !== workflowId()) return;
    if (state.run?.run_id !== run.run_id) {
        for (const video of state.grid.querySelectorAll("video")) video.pause();
        state.grid.replaceChildren();
        state.cards.clear();
    }
    state.run = run;
    state.revision++;
    state.node.properties.h3_removal_preview_run = run.run_id;
    state.progress.textContent = `${run.segments.length}/${run.total} windows · ${run.window_frames}f · seed ${run.seed}` +
        (run.status === "complete" ? " · complete" : run.status === "stopped" ? " · stopped" : "");
    for (const segment of run.segments) {
        if (state.cards.has(segment.index)) continue;
        const card = element("div", "h3-removal-card");
        card.dataset.windowIndex = segment.index;
        if (segment.video) {
            const play = element("button", "");
            play.type = "button";
            play.setAttribute("aria-label", `Play window ${segment.index + 1}`);
            const video = document.createElement("video");
            video.src = viewUrl(segment.video);
            video.poster = viewUrl(segment.poster);
            video.muted = true; video.loop = true; video.playsInline = true; video.preload = "metadata";
            const start = () => {
                for (const other of state.grid.querySelectorAll("video")) if (other !== video) other.pause();
                void video.play().catch(() => {});
            };
            play.addEventListener("pointerenter", e => { if (e.pointerType !== "touch") start(); });
            play.addEventListener("pointerleave", e => {
                if (e.pointerType !== "touch") { video.pause(); video.currentTime = 0; }
            });
            play.addEventListener("click", () => video.paused ? start() : video.pause());
            play.append(video); card.append(play);
        }
        const caption = element("div", "h3-removal-caption");
        caption.append(element("strong", "", `Window ${segment.index + 1} · ${(segment.frames / 24).toFixed(2)}s`));
        caption.append(document.createTextNode(segment.error ||
            `Source ${segment.start}–${segment.start + segment.frames - 1} · ${segment.discarded_overlap} overlap frame(s)`));
        card.append(caption);
        state.grid.append(card);
        state.cards.set(segment.index, card);
    }
    if (!run.segments.length && !state.grid.children.length) {
        state.grid.append(element("div", "h3-removal-empty", run.status === "stopped"
            ? "Stopped before the first window finished." : "Rendering the first window…"));
    } else if (run.segments.length) state.grid.querySelector(".h3-removal-empty")?.remove();
    state.node.setDirtyCanvas?.(true, true);
}

async function refresh(state) {
    if (!mounts.has(state)) return;
    const revision = state.revision;
    const requestedRun = state.node.properties.h3_removal_preview_run;
    const query = new URLSearchParams({workflow_id: workflowId(), node_id: String(state.node.id)});
    if (requestedRun) query.set("run_id", requestedRun);
    try {
        const response = await api.fetchApi(`/h3_relay/removal/previews?${query}`);
        if (response.ok) {
            const run = await response.json();
            if (state.revision === revision && state.node.properties.h3_removal_preview_run === requestedRun) render(state, run);
        }
    } catch (error) { console.debug("H3 removal previews will refresh on reconnect", error); }
}

api.addEventListener("h3_relay.removal.preview", e => {
    for (const state of mounts) render(state, e.detail);
});
for (const event of ["reconnected", "execution_success"]) {
    api.addEventListener(event, () => { for (const state of mounts) void refresh(state); });
}
for (const event of ["execution_interrupted", "execution_error"]) {
    api.addEventListener(event, e => {
        for (const state of mounts) {
            if (state.run?.prompt_id === e.detail?.prompt_id) {
                render(state, {...state.run, status: "stopped"});
                window.setTimeout(() => refresh(state), 750);
            }
        }
    });
}
app.registerExtension({
    name: "h3_relay.person_remover",
    nodeCreated(node) { if (node.comfyClass === NODE) mount(node); },
    loadedGraphNode(node) { if (node.comfyClass === NODE) { mount(node); void refresh(node._h3RemovalPreview); } },
});
