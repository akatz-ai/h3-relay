import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "H3RelayAssemble";

const sleep = (milliseconds) => new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds);
});

function updateButton(node, text, disabled = false) {
    const widget = node._h3RelayStagedWidget;
    if (!widget) return;
    widget.name = text;
    widget.disabled = disabled;
    node.setDirtyCanvas?.(true, true);
}

async function runStaged(node) {
    if (node._h3RelayStagedBusy) return;
    node._h3RelayStagedBusy = true;
    try {
        updateButton(node, "Preparing staged run…", true);
        const serialized = await app.graphToPrompt();
        const response = await api.fetchApi("/h3_relay/staged", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                prompt: serialized.output,
                workflow: serialized.workflow,
                assemble_node_id: String(node.id),
                client_id: api.clientId,
                memory_mode: "minimum_ram",
            }),
        });
        const submitted = await response.json();
        if (!response.ok) throw new Error(submitted.error || `HTTP ${response.status}`);
        const runId = submitted.run_id;
        while (true) {
            const statusResponse = await api.fetchApi(
                `/h3_relay/staged/${encodeURIComponent(runId)}`,
            );
            const status = await statusResponse.json();
            if (!statusResponse.ok) throw new Error(status.error || "Staged status failed.");
            if (status.status === "success") {
                updateButton(node, "✓ Staged sequence complete", false);
                window.setTimeout(() => updateButton(
                    node, "▶ Run staged · bounded RAM", false), 5000);
                break;
            }
            if (["error", "cancelled"].includes(status.status)) {
                throw new Error(status.error || `Staged run ${status.status}.`);
            }
            const current = status.current_stage || 0;
            const total = status.stages?.length || 0;
            const title = status.stage?.title || "waiting";
            updateButton(node, `${current}/${total} · ${title}`, true);
            await sleep(1000);
        }
    } catch (error) {
        console.error("H3 Relay staged execution failed", error);
        updateButton(node, `⚠ ${error?.message || error}`, false);
    } finally {
        node._h3RelayStagedBusy = false;
    }
}

function mount(node) {
    if (node._h3RelayStagedWidget) return;
    const widget = node.addWidget(
        "button",
        "▶ Run staged · bounded RAM",
        null,
        () => void runStaged(node),
    );
    widget.serialize = false;
    node._h3RelayStagedWidget = widget;
}

app.registerExtension({
    name: "h3_relay.staged_assemble",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            window.setTimeout(() => mount(this), 0);
            return result;
        };
    },
    async nodeCreated(node) {
        if (node.comfyClass === NODE_NAME) mount(node);
    },
});
