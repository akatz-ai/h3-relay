import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODES = new Set(["H3RelayPersonRemover", "H3RelayWindowedEdit"]);
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

function controls(state) {
    try { return JSON.parse(state.controlWidget?.value || "{}"); }
    catch { return {}; }
}

function setControls(state, data) {
    state.controlWidget.value = JSON.stringify(data);
    state.node.graph?.change();
    updateButtons(state);
}

function updateButtons(state) {
    const windows = controls(state).windows || {};
    const busy = state.pending || state.run?.status === "rendering";
    state.regenerate.disabled = busy || !state.run?.segments.length || !state.controlWidget;
    for (const card of state.cards.values()) {
        const {segment, seed, reroll, draftSeed} = card._h3Controls;
        seed.disabled = busy;
        if (document.activeElement !== seed) seed.value = draftSeed ?? windows[segment.index]?.seed ?? segment.seed ?? state.run.seed;
        reroll.disabled = busy || !segment.record_id || !state.controlWidget;
    }
}

function keepPrefix(state, data, end) {
    data.windows ??= {};
    for (const item of Object.values(data.windows)) { item.locked = false; delete item.record_id; }
    for (const segment of state.run.segments.filter(s => s.index < end)) {
        if (!segment.record_id) throw new Error("Earlier windows need a saved render before rerolling.");
        data.windows[segment.index] = {seed: segment.seed, locked: true, record_id: segment.record_id};
    }
    if (state.run.segments.filter(s => s.index < end).length !== end) throw new Error("Earlier windows are incomplete.");
}

async function queueReroll(state, index, all) {
    if (state.pending || state.run?.status === "rendering") return;
    const original = state.controlWidget.value;
    const originalRun = state.run.run_id;
    let queued = false;
    try {
        state.pending = true; updateButtons(state);
        const queue = await (await api.fetchApi("/queue")).json();
        if (queue.queue_running.length || queue.queue_pending.length) throw new Error("Finish the current queue before rerolling.");
        const data = controls(state);
        keepPrefix(state, data, index);
        data.requested_from_run = state.run.run_id;
        if (all) {
            data.windows = {};
            for (let i = 0; i < state.run.total; i++) data.windows[i] = {seed: randomSeed(), locked: false};
        } else {
            const card = state.cards.get(index)._h3Controls;
            const chosen = validatedSeed(card.seed.value);
            data.windows[index] = {seed: chosen === (card.segment.seed ?? state.run.seed) ? randomSeed() : chosen, locked: false};
        }
        setControls(state, data);
        state.progress.textContent = all ? "Queued fresh windows…" : `Queued reroll from window ${index + 1}…`;
        const accepted = await app.queuePrompt(0, 1);
        if (accepted === false) throw new Error("Could not queue the render. Check the workflow errors and try again.");
        queued = true;
    } catch (error) {
        state.controlWidget.value = original;
        state.progress.textContent = error.message;
    } finally {
        state.pending = false;
        if (queued && state.run.run_id !== originalRun) render(state, state.run);
        updateButtons(state);
    }
}

function randomSeed() {
    const values = crypto.getRandomValues(new Uint32Array(2));
    return ((BigInt(values[0]) << 32n) | BigInt(values[1])).toString();
}

function validatedSeed(value) {
    if (!/^\d+$/.test(value) || BigInt(value) > 18446744073709551615n) throw new Error("Enter a seed from 0 to 18446744073709551615.");
    return BigInt(value).toString();
}

function mount(node) {
    if (node._h3RemovalPreview) return;
    const masterSeed = node.widgets?.find(w => w.name === "seed");
    if (masterSeed) masterSeed.label = "master seed";
    const root = element("div", "h3-removal-preview");
    const style = document.createElement("style");
    style.textContent = `
      .h3-removal-preview{font:12px/1.45 system-ui,sans-serif;color:#e3ebe7;background:#18201e;
        border:1px solid #40554b;border-radius:8px;padding:10px;box-sizing:border-box;
        width:100%;height:100%;overflow:auto;pointer-events:auto}
      .h3-removal-preview *{box-sizing:border-box}
      .h3-removal-status{display:flex;align-items:center;justify-content:space-between;gap:8px;margin:8px 0}
      .h3-removal-status button,.h3-removal-actions button{border:1px solid #587465;background:#23372c;
        color:#e3ebe7;border-radius:4px;font:inherit;padding:4px 7px;cursor:pointer;white-space:nowrap}
      .h3-removal-status button:disabled,.h3-removal-actions button:disabled{opacity:.4;cursor:default}
      .h3-removal-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
      .h3-removal-card{border:1px solid #41564c;border-radius:6px;overflow:hidden;background:#101613}
      .h3-removal-play{display:block;border:0;padding:0;width:100%;background:#101613;cursor:pointer}
      .h3-removal-actions{display:flex;gap:5px;align-items:center;padding:0 7px 8px}
      .h3-removal-actions input{width:100%;min-width:0;border:1px solid #40554b;border-radius:4px;
        background:#18201e;color:#e3ebe7;padding:5px;font:11px system-ui}
      .h3-removal-actions input:disabled{opacity:.6}
      .h3-removal-card video{display:block;width:100%;aspect-ratio:2/1;object-fit:contain;pointer-events:none}
      .h3-removal-caption{padding:6px 8px;font-size:11px;color:#b8c9bf}
      .h3-removal-caption strong{display:block;color:#e3ebe7;font-weight:600}
      .h3-removal-empty{grid-column:1/-1;padding:25px 12px;text-align:center;color:#a1b5a9;
        border:1px dashed #40554b;border-radius:6px}
      .h3-removal-hint{color:#9caea4;margin-top:8px;font-size:11px}
    `;
    const status = element("div", "h3-removal-status");
    const progress = element("span", "", "Window previews");
    const regenerate = element("button", "", "Regenerate all");
    regenerate.type = "button"; regenerate.disabled = true;
    status.append(progress, regenerate);
    const grid = element("div", "h3-removal-grid");
    grid.append(element("div", "h3-removal-empty", "Each completed window appears here before the next one starts."));
    root.append(style, status, grid, element("div", "h3-removal-hint",
        "Window 1 uses the master seed; each following window adds 1. Change the master to rebuild all windows; keep it fixed to reroll this window onward. Regenerate all creates fresh per-window overrides."));
    const controlWidget = node.widgets?.find(w => w.name === "window_controls");
    if (controlWidget) {
        controlWidget.type = "h3_hidden_controls";
        controlWidget.computeSize = () => [0, -4];
        controlWidget.draw = () => {};
    }
    const state = { node, root, progress, grid, regenerate, controlWidget, pending: false,
                    run: null, revision: 0, cards: new Map() };
    regenerate.addEventListener("click", () => queueReroll(state, 0, true));
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
    // The server hashes actual media/settings, including replacements under the
    // same filename. Bind controls to that identity, once per configuration.
    // Repeated progress/reconnect events must not clear newly chosen locks.
    if (run.config_key && state.controlWidget) {
        let data = controls(state);
        if (data.config_key !== run.config_key) {
            if (data.config_key || run.controls_reset) data = {windows: {}};
            data.config_key = run.config_key;
            setControls(state, data);
        }
    }
    if (state.run?.run_id !== run.run_id) {
        for (const video of state.grid.querySelectorAll("video")) video.pause();
        state.grid.replaceChildren();
        state.cards.clear();
    }
    state.run = run;
    state.revision++;
    state.node.properties.h3_removal_preview_run = run.run_id;
    state.progress.textContent = `${run.segments.length}/${run.total} windows · ${run.window_frames}f${run.window_plan ? " max · adaptive" : ""}` +
        ` · master seed ${run.seed}` +
        (run.status === "complete" ? " · complete" : run.status === "stopped" ? " · stopped" : "") +
        (run.controls_reset ? " · inputs/settings changed — fresh windows" : "");
    for (const segment of run.segments) {
        if (state.cards.has(segment.index)) continue;
        const card = element("div", "h3-removal-card");
        card.dataset.windowIndex = segment.index;
        if (segment.video) {
            const play = element("button", "h3-removal-play");
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
            `Source ${segment.start}–${segment.start + segment.frames - 1}${segment.reused ? " · reused" : ""}`));
        caption.append(element("div", "", `Rendered seed ${segment.seed ?? run.seed}`));
        card.append(caption);
        const actions = element("div", "h3-removal-actions");
        const seed = document.createElement("input"); seed.type = "text"; seed.inputMode = "numeric";
        seed.setAttribute("aria-label", `Seed for window ${segment.index + 1}`);
        seed.title = `Seed for window ${segment.index + 1}`;
        const reroll = element("button", "", "Reroll"); reroll.type = "button";
        reroll.setAttribute("aria-label", `Reroll window ${segment.index + 1}`);
        reroll.title = "Regenerate this window and all later windows; keep earlier results.";
        card._h3Controls = {segment, seed, reroll};
        seed.addEventListener("change", () => {
            // Editing a seed is a draft until Reroll; ordinary Run keeps results.
            card._h3Controls.draftSeed = seed.value;
        });
        reroll.addEventListener("click", () => queueReroll(state, segment.index, false));
        actions.append(seed, reroll); card.append(actions);
        state.grid.append(card);
        state.cards.set(segment.index, card);
    }
    if (!run.segments.length && !state.grid.children.length) {
        state.grid.append(element("div", "h3-removal-empty", run.status === "stopped"
            ? "Stopped before the first window finished." : "Rendering the first window…"));
    } else if (run.segments.length) state.grid.querySelector(".h3-removal-empty")?.remove();
    // Preserve completed windows automatically, including after reload. An old
    // progress/refresh response must not overwrite an explicit queued reroll.
    if (state.controlWidget && !state.pending) {
        const data = controls(state);
        if (data.requested_from_run !== run.run_id) {
            delete data.requested_from_run;
            let end = 0;
            while (run.segments.some(s => s.index === end && s.record_id)) end++;
            keepPrefix(state, data, end);
            setControls(state, data);
        }
    }
    updateButtons(state);
    state.node.setDirtyCanvas?.(true, true);
}

async function refresh(state) {
    if (!mounts.has(state)) return;
    const revision = state.revision;
    const requestedRun = state.node.properties.h3_removal_preview_run;
    const query = new URLSearchParams({workflow_id: workflowId(), node_id: String(state.node.id)});
    // A saved pending reroll may finish while the browser is closed. Follow
    // this node's latest run until that request has adopted its new results.
    if (requestedRun && controls(state).requested_from_run !== requestedRun) query.set("run_id", requestedRun);
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
    nodeCreated(node) { if (NODES.has(node.comfyClass)) mount(node); },
    loadedGraphNode(node) { if (NODES.has(node.comfyClass)) { mount(node); void refresh(node._h3RemovalPreview); } },
});
