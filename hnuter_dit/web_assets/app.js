"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const colors = ["#4cdde5", "#68e6a5", "#bc8cff", "#f4b54d", "#5fa8ff", "#ff7e8d", "#93e36f", "#e889f1", "#64d4a7", "#f3d05e", "#80a7ff", "#ff9f64", "#55ead1", "#dca2ff", "#9be2ff", "#ff879c"];

const state = {
  catalog: null,
  scene: null,
  startPose: null,
  goalPose: null,
  predictions: null,
  selectedObstacle: null,
  selectedTrajectory: 0,
  validationTimer: null,
  busy: false,
  hitBoxes: [],
  endpointHits: [],
  camera: { yaw: -0.72, pitch: 0.58, zoom: 72, panX: 0, panY: 20, view: "iso" },
  drag: null,
  animationStart: performance.now(),
  revision: 0,
  validationRequest: 0,
  execution: { state: "idle", executed_poses: [] },
  executionPending: false,
  executionTimer: null,
  executionRequest: 0,
  executionDisplayRevision: null,
  invalidatedPlanId: null,
};

async function post(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}

function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast show${error ? " error" : ""}`;
  clearTimeout(node.timer);
  node.timer = setTimeout(() => { node.className = "toast"; }, 2600);
}

function randomSeed() {
  return Math.floor(Math.random() * 2_000_000_000) + 1;
}

function format(value, digits = 2) {
  return value != null && value !== "" && Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}

function setBusy(busy, message = "") {
  state.busy = busy;
  refreshControls();
  if (message) {
    $("#inferenceStatus").textContent = message;
    $("#inferenceStatus").className = `inference-status${busy ? " busy" : ""}`;
  }
}

function executionEnabled() { return Boolean(state.catalog?.execution?.enabled); }
function executionActive() { return ["starting", "running", "stopping", "initializing"].includes(state.execution?.state); }
function editsLocked() { return state.busy || state.executionPending || executionActive(); }
function candidateId(trajectory) { return trajectory?.id ?? trajectory?.index; }
function sameId(left, right) { return left != null && right != null && String(left) === String(right); }
function isFinal(trajectory) { return sameId(candidateId(trajectory), state.predictions?.final_candidate_id); }
function selectionFor(trajectory) { return state.predictions?.selection?.candidates?.find((item) => sameId(item.id, candidateId(trajectory))); }

function refreshControls() {
  const locked = editsLocked();
  $("#randomScene").disabled = locked;
  $("#randomEndpoints").disabled = locked || !state.scene;
  $("#runModel").disabled = locked || !state.scene;
  $("#addObstacle").disabled = locked || !state.scene;
  $("#deleteObstacle").disabled = locked;
  $$(".left-panel input, .left-panel select, [data-obstacle]").forEach((input) => { input.disabled = locked; });
  if (executionEnabled()) {
    $("#sampleCount").disabled = true;
    $("#guidanceSteps").disabled = true;
    $("#previewMujoco").disabled = locked || !state.scene;
    $("#executeTrajectory").disabled = locked || !state.predictions?.plan_id || state.predictions?.final_candidate_id == null;
    $("#stopExecution").disabled = state.executionPending || !(executionActive() || state.execution?.state === "preview");
  }
  $$(".choose-final").forEach((button) => { button.disabled = locked; });
}

function quaternionToRpy(q) {
  const [w, x, y, z] = q;
  const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  const sinp = Math.max(-1, Math.min(1, 2 * (w * y - z * x)));
  const pitch = Math.asin(sinp);
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  return [roll, pitch, yaw].map((value) => value * 180 / Math.PI);
}

function rpyToQuaternion(rollDeg, pitchDeg, yawDeg) {
  const [r, p, y] = [rollDeg, pitchDeg, yawDeg].map((v) => Number(v) * Math.PI / 180 / 2);
  const [cr, sr, cp, sp, cy, sy] = [Math.cos(r), Math.sin(r), Math.cos(p), Math.sin(p), Math.cos(y), Math.sin(y)];
  return [
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
  ];
}

function quaternionMatrix(q) {
  let [w, x, y, z] = q.map(Number);
  const norm = Math.hypot(w, x, y, z) || 1;
  [w, x, y, z] = [w / norm, x / norm, y / norm, z / norm];
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}

function fillPoseEditor(kind, pose) {
  if (!pose) return;
  const editor = $(`[data-pose-editor="${kind}"]`);
  const rpy = quaternionToRpy(pose.slice(3));
  const values = { x: pose[0], y: pose[1], z: pose[2], roll: rpy[0], pitch: rpy[1], yaw: rpy[2] };
  $$(`[data-axis]`, editor).forEach((input) => { input.value = format(values[input.dataset.axis], 3); });
}

function readPoseEditor(kind) {
  const editor = $(`[data-pose-editor="${kind}"]`);
  const value = Object.fromEntries($$(`[data-axis]`, editor).map((input) => [input.dataset.axis, Number(input.value)]));
  return [value.x, value.y, value.z, ...rpyToQuaternion(value.roll, value.pitch, value.yaw)];
}

function endpointPayload() {
  state.startPose = readPoseEditor("start");
  state.goalPose = readPoseEditor("goal");
  return { scene: state.scene, start_pose: state.startPose, goal_pose: state.goalPose };
}

function activeObstacles() {
  return state.scene ? state.scene.obstacles.filter((item) => item.role !== "floor") : [];
}

function invalidatePredictions() {
  if (state.predictions?.plan_id) state.invalidatedPlanId = state.predictions.plan_id;
  state.revision += 1;
  state.predictions = null;
  state.selectedTrajectory = 0;
  $("#downloadPredictions").disabled = true;
  renderTrajectoryRail();
  renderMetrics();
  renderFinalSelection();
  refreshControls();
  if (executionEnabled() && state.executionDisplayRevision != null) {
    $("#executionStatus").textContent = "场景或端点已修改，请重新预览 MuJoCo 场景或规划后执行";
    $("#executionElapsed").textContent = $("#executionError").textContent = "—";
  }
}

function applyScene(payload) {
  state.scene = payload.scene;
  state.startPose = payload.start_pose || payload.scene?.precheck_pairs?.[0]?.start_pose;
  state.goalPose = payload.goal_pose || payload.scene?.precheck_pairs?.[0]?.goal_pose;
  state.selectedObstacle = null;
  invalidatePredictions();
  fillPoseEditor("start", state.startPose);
  fillPoseEditor("goal", state.goalPose);
  renderObstacles();
  renderObstacleEditor();
  renderValidation(payload.validation);
  $("#sceneLabel").textContent = `${state.scene.environment_id} · ${activeObstacles().length} BOX TOKENS`;
  $("#emptyState").hidden = true;
  $("#inferenceStatus").textContent = "场景就绪，可以编辑或运行 DiT";
  $("#inferenceStatus").className = "inference-status";
  draw();
}

async function generateScene() {
  if (editsLocked()) return;
  try {
    setBusy(true, "正在生成 URDF 可行场景与随机端点…");
    const payload = await post("/api/scene/random", {
      family: $("#family").value,
      seed: Number($("#sceneSeed").value),
      obstacle_count_min: Number($("#countMin").value),
      obstacle_count_max: Number($("#countMax").value),
    });
    applyScene(payload);
    toast(`已生成 ${payload.scene.generation_family}`);
  } catch (error) {
    $("#inferenceStatus").textContent = error.message;
    $("#inferenceStatus").className = "inference-status error";
    toast(error.message, true);
  } finally { setBusy(false); }
}

async function randomizeEndpoints() {
  if (!state.scene || editsLocked()) return;
  try {
    setBusy(true, "正在当前场景区域内拒绝采样新端点…");
    const payload = await post("/api/endpoints/random", {
      scene: state.scene,
      seed: randomSeed(),
      condition_index: Math.floor(Math.random() * 10000),
    });
    applyScene(payload);
    toast("起点与终点已随机更新");
  } catch (error) {
    $("#inferenceStatus").textContent = error.message;
    $("#inferenceStatus").className = "inference-status error";
    toast(error.message, true);
  } finally { setBusy(false); }
}

function scheduleValidation() {
  if (editsLocked()) return;
  if (state.scene) {
    state.startPose = readPoseEditor("start");
    state.goalPose = readPoseEditor("goal");
  }
  invalidatePredictions();
  clearTimeout(state.validationTimer);
  state.validationTimer = setTimeout(validateCurrent, 220);
  draw();
}

async function validateCurrent() {
  if (!state.scene) return;
  const revision = state.revision, request = ++state.validationRequest;
  try {
    const validation = await post("/api/validate", { ...endpointPayload(), invalidate_plan_id: state.invalidatedPlanId });
    if (revision === state.revision && request === state.validationRequest) renderValidation(validation);
  } catch (error) {
    if (revision === state.revision && request === state.validationRequest) renderValidation({ valid: false, issues: [error.message], warnings: [] });
  }
}

function renderValidation(validation) {
  if (!validation) return;
  const card = $("#validationCard");
  const issues = validation.issues || [];
  const warnings = validation.warnings || [];
  if (issues.length) {
    card.className = "validation-card invalid";
    card.innerHTML = `<b>${issues.length} 个阻断问题</b><span>${escapeHtml(issues.slice(0, 3).join(" · "))}</span>`;
  } else if (warnings.length) {
    card.className = "validation-card warning";
    card.innerHTML = `<b>几何有效，${warnings.length} 个证书提示</b><span>手工编辑后原路线证书可失效，但仍可测试 DiT：${escapeHtml(warnings.slice(0, 2).join(" · "))}</span>`;
  } else {
    card.className = "validation-card valid";
    card.innerHTML = `<b>场景与端点有效</b><span>${validation.active_tokens} / 32 tokens · ${escapeHtml(validation.collision_backend || "checker ready")}</span>`;
  }
  $("#backendTag").textContent = validation.collision_backend?.includes("coal") ? "FULL URDF · COAL" : "CONSERVATIVE OBB";
}

async function runModel() {
  if (!state.scene || editsLocked()) return;
  const revision = state.revision;
  clearTimeout(state.validationTimer);
  state.executionDisplayRevision = null;
  try {
    setBusy(true, "DiT 正在从高斯噪声生成 SE(3) 轨迹…");
    const result = await post("/api/sample", {
      ...endpointPayload(),
      count: executionEnabled() ? 8 : Number($("#sampleCount").value),
      sampling_steps: Number($("#ddimSteps").value),
      seed: Number($("#sampleSeed").value),
      guidance_steps: Number($("#guidanceSteps").value),
      guidance_step_size_m: Number($("#guidanceStepSize").value),
    });
    if (revision !== state.revision) return;
    state.predictions = result;
    state.invalidatedPlanId = null;
    state.selectedTrajectory = Math.max(0, result.trajectories.findIndex((item) => executionEnabled() ? isFinal(item) : item.metrics.valid));
    $("#downloadPredictions").disabled = false;
    const guidance = result.summary.collision_guidance;
    const guidanceLabel = guidance?.enabled ? ` · ${guidance.mode} guidance ${guidance.late_ddim_steps} steps` : "";
    $("#inferenceStatus").textContent = `${result.summary.valid_count}/${result.summary.requested_count} 条通过 · 神经采样 ${format(result.summary.neural_sampling_time_s, 3)} s${guidanceLabel}`;
    $("#inferenceStatus").className = "inference-status";
    renderTrajectoryRail();
    renderMetrics();
    renderFinalSelection();
    state.animationStart = performance.now();
    draw();
    toast(`生成完成：${result.summary.valid_count}/${result.summary.requested_count} 有效`);
  } catch (error) {
    $("#inferenceStatus").textContent = error.message;
    $("#inferenceStatus").className = "inference-status error";
    toast(error.message, true);
  } finally { setBusy(false); }
}

function renderFinalSelection() {
  if (!executionEnabled()) return;
  const predictions = state.predictions;
  const final = predictions?.trajectories?.find(isFinal);
  const label = final ? `最终轨迹 PATH ${String(predictions.trajectories.indexOf(final) + 1).padStart(2, "0")}` : predictions ? "没有可执行的最终轨迹" : "等待生成 8 条候选轨迹";
  const message = predictions?.selection?.message || (final ? "可以直接执行，也可以在候选卡片中更换最终轨迹。" : "修改起终点、场景或推理 seed 后重新生成。");
  $("#finalSelection").className = `final-selection${final ? " ready" : ""}`;
  $("#finalSelection").innerHTML = `<b>${escapeHtml(label)}</b><span>${escapeHtml(message)}</span>`;
  refreshControls();
}

async function chooseFinal(trajectory) {
  if (editsLocked() || !state.predictions?.plan_id) return;
  const revision = state.revision;
  const planId = state.predictions.plan_id;
  try {
    setBusy(true, "正在检查并更换最终轨迹…");
    const result = await post("/api/plan/select", { plan_id: planId, candidate_id: candidateId(trajectory) });
    if (revision !== state.revision || planId !== state.predictions?.plan_id) return;
    state.predictions = result;
    state.selectedTrajectory = Math.max(0, result.trajectories.findIndex(isFinal));
    state.animationStart = performance.now();
    state.executionDisplayRevision = null;
    renderTrajectoryRail(); renderMetrics(); renderFinalSelection(); draw();
    setBusy(false, "已更新最终轨迹，可开始 MPPI 执行");
  } catch (error) { toast(error.message, true); }
  finally { setBusy(false); }
}

function renderExecutionStatus() {
  if (!executionEnabled()) return;
  const status = state.execution || {};
  const labels = { idle: "就绪", preview: "场景预览", starting: "正在启动", running: "正在执行", stopping: "正在停止", completed: "执行完成", stopped: "已停止", failed: "执行失败" };
  const stale = state.executionDisplayRevision != null && state.executionDisplayRevision !== state.revision;
  const node = $("#executionStatus");
  node.className = `execution-status${status.state === "failed" || status.error ? " error" : ""}${executionActive() ? " running" : ""}`;
  node.textContent = stale ? "场景或端点已修改，请重新预览 MuJoCo 场景或规划后执行" : `${labels[status.state] || status.state || "就绪"} · ${status.message || "先预览场景，或生成候选后执行最终轨迹"}${status.error ? ` · ${status.error}` : ""}`;
  $("#executionElapsed").textContent = stale ? "—" : `${format(status.simulation_time_s ?? status.elapsed_s, 1)} s`;
  $("#executionError").textContent = stale ? "—" : `${format(status.tracking_error_m, 3)} m`;
  if (status.retiming && !stale) {
    $("#executionReference").textContent = `TOPP-RA 连续参考 ${format(status.retiming.duration_s, 1)} s · 速度上限 ${format(status.retiming.speed_mps, 1)} m/s · ${status.reference_audit?.checked ? "平滑曲线已通过净空复检" : "连续通过中间 waypoint"}`;
  } else {
    $("#executionReference").textContent = "TOPP-RA 连续参考；起终点静止，中间持续飞行。";
  }
  refreshControls();
  draw();
}

async function pollExecution() {
  if (!executionEnabled() || state.executionPending) return;
  const request = state.executionRequest;
  try {
    const response = await fetch("/api/execution/status", { cache: "no-store" });
    const status = await response.json();
    if (!response.ok) throw new Error(status.error || `HTTP ${response.status}`);
    if (request !== state.executionRequest || state.executionPending) return;
    state.execution = status;
    renderExecutionStatus();
  } catch (error) {
    if (request !== state.executionRequest) return;
    $("#executionStatus").textContent = `无法读取 MuJoCo 状态 · ${error.message}`;
    $("#executionStatus").className = "execution-status error";
  } finally {
    clearTimeout(state.executionTimer);
    state.executionTimer = setTimeout(pollExecution, 500);
  }
}

async function executionAction(action) {
  if (!executionEnabled() || state.executionPending) return;
  if (action !== "stop" && editsLocked()) return;
  if (action === "start" && (!state.predictions?.plan_id || state.predictions.final_candidate_id == null)) return;
  if (action === "preview" && !state.scene) return;
  const revision = state.revision;
  const request = ++state.executionRequest;
  state.executionPending = true;
  state.executionDisplayRevision = revision;
  state.execution = { ...state.execution, state: action === "stop" ? "stopping" : "starting", message: action === "preview" ? "同步当前场景与起终点到 MuJoCo…" : action === "start" ? "加载最终轨迹并启动 MPPI…" : "等待仿真退出…" };
  renderExecutionStatus();
  try {
    const payload = action === "preview" ? endpointPayload() : action === "start" ? { plan_id: state.predictions.plan_id, candidate_id: state.predictions.final_candidate_id } : {};
    const status = await post(`/api/execution/${action}`, payload);
    if (request !== state.executionRequest || revision !== state.revision) return;
    state.execution = status;
    if (status.state === "failed") toast(status.error || status.message || "MuJoCo 执行失败", true);
  } catch (error) {
    state.execution = { ...state.execution, error: error.message, message: error.message };
    toast(error.message, true);
  } finally {
    state.executionPending = false;
    renderExecutionStatus();
    clearTimeout(state.executionTimer);
    state.executionTimer = setTimeout(pollExecution, 500);
  }
}

function renderObstacles() {
  const list = $("#obstacleList");
  list.innerHTML = "";
  activeObstacles().forEach((obstacle, index) => {
    const row = document.createElement("div");
    row.className = `obstacle-row${index === state.selectedObstacle ? " selected" : ""}`;
    const p = obstacle.pose.position, s = obstacle.size_xyz;
    row.innerHTML = `<i></i><div><b>${escapeHtml(obstacle.id || `box_${index}`)}</b><small>${p.map((v) => format(v)).join(" · ")}</small></div><em>${s.map((v) => format(v)).join("×")}</em>`;
    row.addEventListener("click", () => {
      state.selectedObstacle = index;
      renderObstacles(); renderObstacleEditor(); draw();
    });
    list.appendChild(row);
  });
  $("#obstacleCount").textContent = `${activeObstacles().length} / 32 TOKENS`;
}

function renderObstacleEditor() {
  const editor = $("#obstacleEditor");
  const obstacle = activeObstacles()[state.selectedObstacle];
  editor.hidden = !obstacle;
  if (!obstacle) return;
  $("#obstacleName").textContent = obstacle.id;
  $$(`[data-obstacle]`, editor).forEach((input) => {
    const field = input.dataset.obstacle;
    if (field.startsWith("position")) input.value = obstacle.pose.position[Number(field.split(".")[1])];
    else if (field.startsWith("size")) input.value = obstacle.size_xyz[Number(field.split(".")[1])];
    else input.value = format(quaternionToRpy(obstacle.pose.quaternion_wxyz)[2], 2);
  });
}

function updateObstacle(input) {
  if (editsLocked()) return;
  const obstacle = activeObstacles()[state.selectedObstacle];
  if (!obstacle || !Number.isFinite(Number(input.value))) return;
  const field = input.dataset.obstacle;
  if (field.startsWith("position")) obstacle.pose.position[Number(field.split(".")[1])] = Number(input.value);
  else if (field.startsWith("size")) obstacle.size_xyz[Number(field.split(".")[1])] = Math.max(.05, Number(input.value));
  else obstacle.pose.quaternion_wxyz = rpyToQuaternion(0, 0, Number(input.value));
  renderObstacles(); scheduleValidation();
}

function addObstacle(position = [0, 0, 0.5]) {
  if (!state.scene || editsLocked() || activeObstacles().length >= 32) return;
  const id = `manual_box_${Date.now().toString(36)}`;
  state.scene.obstacles.push({
    id, type: "box", collision: true, role: "obstacle",
    family: "manual", functional_role: "distractor", physical_support: "ground",
    pose: { position: position.map(Number), quaternion_wxyz: [1, 0, 0, 0] },
    size_xyz: [1, 1, 1],
  });
  state.selectedObstacle = activeObstacles().length - 1;
  renderObstacles(); renderObstacleEditor(); scheduleValidation();
}

function deleteObstacle() {
  if (editsLocked()) return;
  const target = activeObstacles()[state.selectedObstacle];
  if (!target) return;
  const actualIndex = state.scene.obstacles.indexOf(target);
  if (actualIndex >= 0) state.scene.obstacles.splice(actualIndex, 1);
  state.selectedObstacle = null;
  renderObstacles(); renderObstacleEditor(); scheduleValidation();
}

function renderTrajectoryRail() {
  const rail = $("#trajectoryRail");
  rail.innerHTML = "";
  const trajectories = state.predictions?.trajectories || [];
  if (!trajectories.length) {
    rail.innerHTML = `<div class="rail-empty">生成轨迹后可在这里切换候选并查看 COAL 指标</div>`;
    return;
  }
  trajectories.forEach((trajectory, index) => {
    const metric = trajectory.metrics;
    const final = executionEnabled() && isFinal(trajectory);
    const selection = selectionFor(trajectory);
    const card = document.createElement("div");
    card.className = `trajectory-card${metric.valid ? "" : " invalid"}${index === state.selectedTrajectory ? " selected" : ""}${final ? " final" : ""}`;
    card.title = selection?.reason || (metric.valid ? "几何检查通过" : "几何检查未通过");
    card.innerHTML = `<header><b>PATH ${String(index + 1).padStart(2, "0")}</b><span>${final ? "FINAL" : metric.valid ? "VALID" : "REJECT"}</span></header><dl><dt>CLEARANCE</dt><dd>${format(metric.minimum_clearance_m, 3)} m</dd><dt>LENGTH</dt><dd>${format(metric.path_length_m, 2)} m</dd><dt>COLLISIONS</dt><dd>${metric.collision_sample_count}</dd></dl>`;
    if (executionEnabled()) {
      const note = document.createElement("div");
      note.className = "candidate-selection-note";
      note.textContent = selection?.eligible ? `SCORE ${format(selection.score, 3)}` : selection?.reason || "未通过执行筛选";
      card.appendChild(note);
      if (selection?.eligible && !final) {
        const button = document.createElement("button");
        button.className = "choose-final"; button.textContent = "设为最终轨迹";
        button.disabled = editsLocked();
        button.addEventListener("click", (event) => { event.stopPropagation(); chooseFinal(trajectory); });
        card.appendChild(button);
      }
    }
    card.addEventListener("click", () => {
      state.selectedTrajectory = index;
      state.animationStart = performance.now();
      renderTrajectoryRail(); renderMetrics(); draw();
    });
    rail.appendChild(card);
  });
}

function renderMetrics() {
  const summary = state.predictions?.summary;
  const selected = state.predictions?.trajectories?.[state.selectedTrajectory];
  if (!summary || !selected) {
    $("#validMetric").textContent = $("#latencyMetric").textContent = $("#clearanceMetric").textContent = $("#lengthMetric").textContent = "—";
    $("#selectedDetails").textContent = "选择一条生成轨迹查看详细指标";
    return;
  }
  const metric = selected.metrics;
  const selection = selectionFor(selected);
  $("#validMetric").textContent = `${summary.valid_count}/${summary.requested_count}`;
  $("#latencyMetric").textContent = format(summary.milliseconds_per_path, 1);
  $("#clearanceMetric").textContent = `${format(metric.minimum_clearance_m, 3)}m`;
  $("#lengthMetric").textContent = `${format(metric.path_length_m, 2)}m`;
  $("#selectedDetails").innerHTML = [
    `PATH ${state.selectedTrajectory + 1} · ${isFinal(selected) ? "FINAL · " : ""}${metric.valid ? "VALID" : "REJECTED"}`,
    ...(selection ? [`执行筛选：${escapeHtml(selection.reason || (selection.eligible ? "可执行" : "不可执行"))}${selection.score == null ? "" : ` · score ${format(selection.score, 3)}`}`] : []),
    `collision samples = ${metric.collision_sample_count}`,
    `bounds violations = ${metric.out_of_bounds_sample_count}`,
    `attitude violations = ${metric.attitude_violation_sample_count}`,
    `p95 / max turn = ${format(metric.p95_turn_angle_deg, 1)}° / ${format(metric.maximum_turn_angle_deg, 1)}°`,
    `accel RMS / jerk RMS = ${format(metric.acceleration_rms_m_per_index2, 5)} / ${format(metric.jerk_rms_m_per_index3, 5)}`,
    `backtracking = ${format(100 * metric.backtracking_fraction, 2)}%`,
    `max |roll| / |pitch| = ${format(metric.maximum_abs_roll_deg, 1)}° / ${format(metric.maximum_abs_pitch_deg, 1)}°`,
  ].join("<br>");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[character]);
}

function download(name, value) {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

// Canvas renderer -----------------------------------------------------------
const canvas = $("#sceneCanvas");
const context = canvas.getContext("2d");
const boxFaces = [[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]];
const boxEdges = [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return rect;
}

function project(point, rect) {
  const [x, y, z0] = [point[0], point[1], point[2] - 1.8];
  const camera = state.camera;
  const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
  const rx = cy * x - sy * y;
  const ry = sy * x + cy * y;
  const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
  const vertical = cp * z0 - sp * ry;
  const depth = sp * z0 + cp * ry;
  return [rect.width / 2 + camera.panX + rx * camera.zoom, rect.height / 2 + camera.panY - vertical * camera.zoom, depth];
}

function boxVertices(obstacle) {
  const center = obstacle.pose.position.map(Number);
  const half = obstacle.size_xyz.map((value) => Number(value) / 2);
  const rotation = quaternionMatrix(obstacle.pose.quaternion_wxyz);
  const result = [];
  for (let bits = 0; bits < 8; bits += 1) {
    const local = [bits & 1 ? half[0] : -half[0], bits & 2 ? half[1] : -half[1], bits & 4 ? half[2] : -half[2]];
    result.push(center.map((value, row) => value + rotation[row][0] * local[0] + rotation[row][1] * local[1] + rotation[row][2] * local[2]));
  }
  return result;
}

function drawLine(points, rect, color, width = 1, dash = []) {
  if (points.length < 2) return;
  context.beginPath(); context.setLineDash(dash);
  points.forEach((point, index) => {
    const p = project(point, rect);
    if (index) context.lineTo(p[0], p[1]); else context.moveTo(p[0], p[1]);
  });
  context.strokeStyle = color; context.lineWidth = width; context.stroke(); context.setLineDash([]);
}

function drawBounds(rect) {
  const low = [-4, -4, 0], high = [4, 4, 4];
  const vertices = [];
  for (let bits = 0; bits < 8; bits += 1) vertices.push([bits & 1 ? high[0] : low[0], bits & 2 ? high[1] : low[1], bits & 4 ? high[2] : low[2]]);
  context.strokeStyle = "#26343e"; context.lineWidth = .75; context.setLineDash([4, 5]);
  boxEdges.forEach(([a, b]) => {
    const p = project(vertices[a], rect), q = project(vertices[b], rect);
    context.beginPath(); context.moveTo(p[0], p[1]); context.lineTo(q[0], q[1]); context.stroke();
  });
  context.setLineDash([]);
}

function drawObstacles(rect) {
  const faces = [];
  state.hitBoxes = [];
  activeObstacles().forEach((obstacle, obstacleIndex) => {
    const projected = boxVertices(obstacle).map((point) => project(point, rect));
    const xs = projected.map((p) => p[0]), ys = projected.map((p) => p[1]);
    state.hitBoxes.push({ obstacleIndex, minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) });
    boxFaces.forEach((face, faceIndex) => faces.push({ obstacleIndex, faceIndex, points: face.map((index) => projected[index]), depth: face.reduce((sum, index) => sum + projected[index][2], 0) / face.length }));
  });
  faces.sort((a, b) => a.depth - b.depth);
  faces.forEach((face) => {
    const selected = face.obstacleIndex === state.selectedObstacle;
    const light = .12 + face.faceIndex * .018;
    context.beginPath();
    face.points.forEach((p, index) => index ? context.lineTo(p[0], p[1]) : context.moveTo(p[0], p[1]));
    context.closePath();
    context.fillStyle = selected ? `rgba(76,221,229,${.18 + light})` : `rgba(224,143,55,${light})`;
    context.strokeStyle = selected ? "#69e9ef" : "#a66f35";
    context.lineWidth = selected ? 1.4 : .65;
    context.fill(); context.stroke();
  });
}

function drawEndpoint(pose, label, color, rect) {
  if (!pose) return;
  const point = project(pose, rect);
  context.beginPath(); context.arc(point[0], point[1], 6, 0, Math.PI * 2);
  context.fillStyle = color; context.shadowColor = color; context.shadowBlur = 12; context.fill(); context.shadowBlur = 0;
  context.fillStyle = "#e8eef3"; context.font = "700 9px ui-monospace, monospace"; context.fillText(label, point[0] + 9, point[1] - 8);
  state.endpointHits.push({ kind: label === "START" ? "start" : "goal", x: point[0], y: point[1], radius: 13 });
  const matrix = quaternionMatrix(pose.slice(3));
  [[0,"#ff6678"],[1,"#68e6a5"],[2,"#62a9ff"]].forEach(([axis, axisColor]) => {
    const end = pose.slice(0, 3).map((value, row) => value + .38 * matrix[row][axis]);
    const projected = project(end, rect);
    context.beginPath(); context.moveTo(point[0], point[1]); context.lineTo(projected[0], projected[1]); context.strokeStyle = axisColor; context.lineWidth = 1.2; context.stroke();
  });
}

function drawTrajectories(rect) {
  const trajectories = state.predictions?.trajectories || [];
  if (!trajectories.length) return;
  const showAll = $("#showAll").checked;
  trajectories.forEach((trajectory, index) => {
    if (!showAll && index !== state.selectedTrajectory && !isFinal(trajectory)) return;
    const selected = index === state.selectedTrajectory;
    const path = trajectory.poses_wxyz;
    const clearance = trajectory.clearance_m;
    for (let pointIndex = 1; pointIndex < path.length; pointIndex += 1) {
      const collision = clearance[pointIndex] <= 0 || clearance[pointIndex - 1] <= 0;
      drawLine([path[pointIndex - 1], path[pointIndex]], rect, collision ? "#ff5268" : colors[index % colors.length], selected ? 2.8 : 1.15);
    }
  });
  const final = executionEnabled() ? trajectories.find(isFinal) : null;
  if (final) {
    drawLine(final.poses_wxyz, rect, "#060a0f", 6);
    drawLine(final.poses_wxyz, rect, "#ffffff", 3.2);
    const middle = project(final.poses_wxyz[Math.floor(final.poses_wxyz.length / 2)], rect);
    context.font = "700 9px ui-monospace, monospace"; context.fillStyle = "#ffffff";
    context.fillText("FINAL", middle[0] + 8, middle[1] - 8);
  }
  if ($("#animateVehicle").checked) {
    const selected = trajectories[state.selectedTrajectory];
    if (selected) {
      const index = Math.floor(((performance.now() - state.animationStart) / 35) % selected.poses_wxyz.length);
      const pose = selected.poses_wxyz[index], p = project(pose, rect);
      context.beginPath(); context.arc(p[0], p[1], 5, 0, Math.PI * 2); context.fillStyle = "#ffffff"; context.shadowColor = colors[state.selectedTrajectory % colors.length]; context.shadowBlur = 14; context.fill(); context.shadowBlur = 0;
    }
  }
}

function drawExecution(rect) {
  if (!executionEnabled() || state.executionDisplayRevision !== state.revision) return;
  const status = state.execution;
  if (status.plan_id && status.plan_id !== state.predictions?.plan_id) return;
  const poses = status.executed_poses || [];
  if (!poses.length) return;
  drawLine(poses, rect, "#ffab52", 3);
  const pose = poses[poses.length - 1];
  if (pose.length < 7) return;
  const rotation = quaternionMatrix(pose.slice(3, 7));
  const worldPoint = (local) => pose.slice(0, 3).map((value, row) => value + rotation[row][0] * local[0] + rotation[row][1] * local[1] + rotation[row][2] * local[2]);
  const arms = [[-.20, -.20, 0], [.20, .20, 0], [-.20, .20, 0], [.20, -.20, 0]].map(worldPoint);
  drawLine(arms.slice(0, 2), rect, "#ffab52", 2.8);
  drawLine(arms.slice(2), rect, "#ffab52", 2.8);
  arms.forEach((arm) => {
    const point = project(arm, rect);
    context.beginPath(); context.arc(point[0], point[1], 4, 0, Math.PI * 2);
    context.strokeStyle = "#ffab52"; context.lineWidth = 1.5; context.stroke();
  });
  const center = project(pose, rect);
  context.beginPath(); context.arc(center[0], center[1], 4, 0, Math.PI * 2);
  context.fillStyle = "#ffab52"; context.fill();
  context.font = "700 9px ui-monospace, monospace"; context.fillText("MUJOCO · ACTUAL", center[0] + 10, center[1] + 16);
}

function draw() {
  const rect = resizeCanvas();
  context.clearRect(0, 0, rect.width, rect.height);
  state.endpointHits = [];
  drawBounds(rect);
  if (state.scene) drawObstacles(rect);
  drawTrajectories(rect);
  drawEndpoint(state.startPose, "START", "#68e6a5", rect);
  drawEndpoint(state.goalPose, "GOAL", "#bc8cff", rect);
  drawExecution(rect);
  $("#cameraLabel").textContent = `${state.camera.view.toUpperCase()} · ${Math.round(state.camera.zoom / 72 * 100)}%`;
}

function resetCamera(view = "iso") {
  state.camera.view = view;
  if (view === "top") Object.assign(state.camera, { yaw: 0, pitch: Math.PI / 2 - .01, zoom: 64, panX: 0, panY: 0 });
  else if (view === "front") Object.assign(state.camera, { yaw: 0, pitch: 0, zoom: 72, panX: 0, panY: 50 });
  else Object.assign(state.camera, { yaw: -.72, pitch: .58, zoom: 72, panX: 0, panY: 20 });
  $$(`[data-view]`).forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  draw();
}

function canvasCoordinates(event) {
  const rect = canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top, rect };
}

function hitTarget(x, y) {
  const endpoint = state.endpointHits.find((item) => Math.hypot(x - item.x, y - item.y) <= item.radius);
  if (endpoint) return endpoint;
  const obstacle = [...state.hitBoxes].reverse().find((box) => x >= box.minX && x <= box.maxX && y >= box.minY && y <= box.maxY);
  return obstacle ? { kind: "obstacle", index: obstacle.obstacleIndex } : null;
}

function screenToWorld(x, y, fixedCoordinate, rect) {
  const camera = state.camera;
  const screenX = (x - rect.width / 2 - camera.panX) / camera.zoom;
  const vertical = -(y - rect.height / 2 - camera.panY) / camera.zoom;
  const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
  const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
  if (camera.view === "front" || Math.abs(sp) < .08) {
    const worldY = fixedCoordinate;
    const worldX = Math.abs(cy) > .08 ? (screenX + sy * worldY) / cy : 0;
    const rotatedY = sy * worldX + cy * worldY;
    const worldZ = Math.abs(cp) > .08 ? (vertical + sp * rotatedY) / cp + 1.8 : 1.8;
    return [worldX, worldY, worldZ];
  }
  const worldZ = fixedCoordinate;
  const rotatedY = (cp * (worldZ - 1.8) - vertical) / sp;
  return [cy * screenX + sy * rotatedY, -sy * screenX + cy * rotatedY, worldZ];
}

function targetPosition(target) {
  if (target.kind === "start") return state.startPose.slice(0, 3);
  if (target.kind === "goal") return state.goalPose.slice(0, 3);
  return activeObstacles()[target.index].pose.position.slice();
}

function setTargetPosition(target, position) {
  if (target.kind === "start" || target.kind === "goal") {
    const pose = target.kind === "start" ? state.startPose : state.goalPose;
    pose.splice(0, 3, ...position);
    fillPoseEditor(target.kind, pose);
  } else {
    const obstacle = activeObstacles()[target.index];
    obstacle.pose.position.splice(0, 3, ...position);
    renderObstacleEditor();
    renderObstacles();
  }
}

function clampTargetPosition(target, position) {
  const result = position.slice();
  if (target.kind === "obstacle") {
    const obstacle = activeObstacles()[target.index];
    const half = obstacle.size_xyz.map((value) => Math.max(.025, Number(value) / 2));
    result[0] = Math.max(-4 + half[0], Math.min(4 - half[0], result[0]));
    result[1] = Math.max(-4 + half[1], Math.min(4 - half[1], result[1]));
    result[2] = Math.max(half[2], Math.min(4 - half[2], result[2]));
  } else {
    result[0] = Math.max(-3.3, Math.min(3.3, result[0]));
    result[1] = Math.max(-3.3, Math.min(3.3, result[1]));
    result[2] = Math.max(.65, Math.min(3.5, result[2]));
  }
  return result;
}

canvas.addEventListener("mousedown", (event) => {
  const point = canvasCoordinates(event);
  const target = event.button === 0 && !editsLocked() ? hitTarget(point.x, point.y) : null;
  if (target) {
    if (target.kind === "obstacle") {
      state.selectedObstacle = target.index;
      renderObstacles(); renderObstacleEditor();
    }
    state.drag = {
      x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY,
      target, original: targetPosition(target), altitude: event.altKey, moved: false,
    };
    canvas.classList.add("drag-object");
    return;
  }
  state.drag = { x: event.clientX, y: event.clientY, pan: event.button === 2 || event.shiftKey, moved: false };
});
window.addEventListener("mousemove", (event) => {
  if (!state.drag) return;
  const dx = event.clientX - state.drag.x, dy = event.clientY - state.drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 2) state.drag.moved = true;
  if (state.drag.target) {
    if (editsLocked()) return;
    if (!state.drag.moved) return;
    if (!state.drag.invalidated) { invalidatePredictions(); state.drag.invalidated = true; }
    const point = canvasCoordinates(event);
    let position;
    if (state.drag.altitude || event.altKey) {
      position = state.drag.original.slice();
      position[2] = state.drag.original[2] - (event.clientY - state.drag.startY) / state.camera.zoom;
    } else {
      const fixed = state.camera.view === "front" ? state.drag.original[1] : state.drag.original[2];
      position = screenToWorld(point.x, point.y, fixed, point.rect);
    }
    setTargetPosition(state.drag.target, clampTargetPosition(state.drag.target, position));
  } else if (state.drag.pan) { state.camera.panX += dx; state.camera.panY += dy; }
  else { state.camera.yaw += dx * .008; state.camera.pitch = Math.max(.02, Math.min(1.54, state.camera.pitch - dy * .006)); state.camera.view = "iso"; }
  state.drag.x = event.clientX; state.drag.y = event.clientY; draw();
});
window.addEventListener("mouseup", (event) => {
  if (!state.drag) return;
  const movedTarget = state.drag.target && state.drag.moved;
  if (!state.drag.target && !state.drag.moved && event.target === canvas) {
    const rect = canvas.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
    const hits = state.hitBoxes.filter((box) => x >= box.minX && x <= box.maxX && y >= box.minY && y <= box.maxY);
    if (hits.length) { state.selectedObstacle = hits[hits.length - 1].obstacleIndex; renderObstacles(); renderObstacleEditor(); draw(); }
  }
  canvas.classList.remove("drag-object");
  state.drag = null;
  if (movedTarget) scheduleValidation();
});
canvas.addEventListener("mousemove", (event) => {
  if (state.drag) return;
  const point = canvasCoordinates(event);
  canvas.classList.toggle("drag-object", !editsLocked() && Boolean(hitTarget(point.x, point.y)));
});
canvas.addEventListener("mouseleave", () => { if (!state.drag) canvas.classList.remove("drag-object"); });
canvas.addEventListener("dblclick", (event) => {
  if (!state.scene || editsLocked()) return;
  const point = canvasCoordinates(event);
  if (hitTarget(point.x, point.y)) return;
  const world = screenToWorld(point.x, point.y, state.camera.view === "front" ? 0 : .5, point.rect);
  const position = state.camera.view === "front" ? [world[0], 0, Math.max(.5, world[2])] : [world[0], world[1], .5];
  addObstacle([
    Math.max(-3.5, Math.min(3.5, position[0])),
    Math.max(-3.5, Math.min(3.5, position[1])),
    Math.max(0.5, Math.min(3.5, position[2])),
  ]);
});
canvas.addEventListener("wheel", (event) => { event.preventDefault(); state.camera.zoom = Math.max(28, Math.min(180, state.camera.zoom * Math.exp(-event.deltaY * .001))); draw(); }, { passive: false });
canvas.addEventListener("contextmenu", (event) => event.preventDefault());

// Wiring --------------------------------------------------------------------
$("#randomScene").addEventListener("click", generateScene);
$("#randomEndpoints").addEventListener("click", randomizeEndpoints);
$("#runModel").addEventListener("click", runModel);
$("#previewMujoco").addEventListener("click", () => executionAction("preview"));
$("#executeTrajectory").addEventListener("click", () => executionAction("start"));
$("#stopExecution").addEventListener("click", () => executionAction("stop"));
$("#addObstacle").addEventListener("click", () => addObstacle());
$("#deleteObstacle").addEventListener("click", deleteObstacle);
$$(`[data-obstacle]`).forEach((input) => input.addEventListener("input", () => updateObstacle(input)));
$$(`[data-pose-editor] input`).forEach((input) => input.addEventListener("input", scheduleValidation));
$$("#sampleCount, #ddimSteps, #guidanceSteps, #guidanceStepSize, #sampleSeed").forEach((input) => input.addEventListener("change", () => { if (!editsLocked()) { invalidatePredictions(); draw(); } }));
$$(`[data-view]`).forEach((button) => button.addEventListener("click", () => resetCamera(button.dataset.view)));
$("#resetCamera").addEventListener("click", () => resetCamera(state.camera.view));
$("#showAll").addEventListener("change", draw);
$("#animateVehicle").addEventListener("change", () => { state.animationStart = performance.now(); draw(); });
$("#downloadScene").addEventListener("click", () => state.scene && download(`${state.scene.environment_id || "edited_scene"}.json`, { scene: state.scene, start_pose: state.startPose, goal_pose: state.goalPose }));
$("#downloadPredictions").addEventListener("click", () => state.predictions && download(`dit_predictions_${state.predictions.scene_id}.json`, state.predictions));
window.addEventListener("resize", draw);

let lastAnimationFrame = 0;
function animationLoop(timestamp) {
  if ($("#animateVehicle").checked && timestamp - lastAnimationFrame > 32) { draw(); lastAnimationFrame = timestamp; }
  requestAnimationFrame(animationLoop);
}
requestAnimationFrame(animationLoop);

async function initialize() {
  try {
    const response = await fetch("/api/catalog");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.catalog = await response.json();
    state.catalog.families.forEach((family) => {
      const option = document.createElement("option"); option.value = family; option.textContent = family; $("#family").appendChild(option);
    });
    const model = state.catalog.model;
    $("#modelDevice").textContent = `${model.device.toUpperCase()} READY`;
    $("#modelDevice").classList.add("ready");
    $("#modelName").textContent = `${model.preset} · ${model.parameter_count.toLocaleString()} params`;
    $("#modelMeta").textContent = `step ${model.training_step.toLocaleString()} · val ${format(model.validation_loss, 5)} · ${model.checkpoint_sha256.slice(0, 12)}`;
    $("#sceneSeed").value = executionEnabled() ? state.catalog.initial_scene_seed ?? 42 : randomSeed();
    if (executionEnabled()) {
      document.body.classList.add("execution-enabled");
      document.title = "Base DiT + MPPI · MuJoCo";
      $(".brand h1").textContent = "Base DiT + MPPI";
      $("#executionPanel").hidden = false;
      $$(".execution-only").forEach((node) => { node.hidden = false; });
      $("#sampleCount").value = state.catalog.execution.batch_size || 8;
      $("#guidanceSteps").value = String(state.catalog.execution.guidance_steps || 8);
      $("#runModel span").textContent = "生成 8 条候选并选择最终轨迹";
      $(".safety-note").textContent = "先检查最终轨迹，再执行 MPPI。界面显示实际飞行轨迹；MuJoCo 窗口呈现完整动力学运动。";
      renderFinalSelection();
    }
    setBusy(false, "模型已加载，正在准备首个场景…");
    await generateScene();
    if (executionEnabled()) pollExecution();
  } catch (error) {
    $("#modelDevice").textContent = "MODEL ERROR";
    $("#inferenceStatus").textContent = error.message;
    $("#inferenceStatus").className = "inference-status error";
    toast(error.message, true);
  }
  draw();
}

initialize();
