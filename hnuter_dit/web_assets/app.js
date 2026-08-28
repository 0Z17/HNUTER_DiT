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
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}

function setBusy(busy, message = "") {
  state.busy = busy;
  $("#randomScene").disabled = busy;
  $("#randomEndpoints").disabled = busy || !state.scene;
  $("#runModel").disabled = busy || !state.scene;
  if (message) {
    $("#inferenceStatus").textContent = message;
    $("#inferenceStatus").className = `inference-status${busy ? " busy" : ""}`;
  }
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
  state.predictions = null;
  state.selectedTrajectory = 0;
  $("#downloadPredictions").disabled = true;
  renderTrajectoryRail();
  renderMetrics();
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
  if (!state.scene) return;
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
  try { renderValidation(await post("/api/validate", endpointPayload())); }
  catch (error) { renderValidation({ valid: false, issues: [error.message], warnings: [] }); }
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
  if (!state.scene) return;
  try {
    setBusy(true, "DiT 正在从高斯噪声生成 SE(3) 轨迹…");
    const result = await post("/api/sample", {
      ...endpointPayload(),
      count: Number($("#sampleCount").value),
      sampling_steps: Number($("#ddimSteps").value),
      seed: Number($("#sampleSeed").value),
    });
    state.predictions = result;
    state.selectedTrajectory = Math.max(0, result.trajectories.findIndex((item) => item.metrics.valid));
    $("#downloadPredictions").disabled = false;
    $("#inferenceStatus").textContent = `${result.summary.valid_count}/${result.summary.requested_count} 条通过 · 神经采样 ${format(result.summary.neural_sampling_time_s, 3)} s`;
    $("#inferenceStatus").className = "inference-status";
    renderTrajectoryRail();
    renderMetrics();
    state.animationStart = performance.now();
    draw();
    toast(`生成完成：${result.summary.valid_count}/${result.summary.requested_count} 有效`);
  } catch (error) {
    $("#inferenceStatus").textContent = error.message;
    $("#inferenceStatus").className = "inference-status error";
    toast(error.message, true);
  } finally { setBusy(false); }
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
  const obstacle = activeObstacles()[state.selectedObstacle];
  if (!obstacle || !Number.isFinite(Number(input.value))) return;
  const field = input.dataset.obstacle;
  if (field.startsWith("position")) obstacle.pose.position[Number(field.split(".")[1])] = Number(input.value);
  else if (field.startsWith("size")) obstacle.size_xyz[Number(field.split(".")[1])] = Math.max(.05, Number(input.value));
  else obstacle.pose.quaternion_wxyz = rpyToQuaternion(0, 0, Number(input.value));
  renderObstacles(); scheduleValidation();
}

function addObstacle(position = [0, 0, 0.5]) {
  if (!state.scene || activeObstacles().length >= 32) return;
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
    const card = document.createElement("div");
    card.className = `trajectory-card${metric.valid ? "" : " invalid"}${index === state.selectedTrajectory ? " selected" : ""}`;
    card.innerHTML = `<header><b>PATH ${String(index + 1).padStart(2, "0")}</b><span>${metric.valid ? "VALID" : "REJECT"}</span></header><dl><dt>CLEARANCE</dt><dd>${format(metric.minimum_clearance_m, 3)} m</dd><dt>LENGTH</dt><dd>${format(metric.path_length_m, 2)} m</dd><dt>COLLISIONS</dt><dd>${metric.collision_sample_count}</dd></dl>`;
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
  $("#validMetric").textContent = `${summary.valid_count}/${summary.requested_count}`;
  $("#latencyMetric").textContent = format(summary.milliseconds_per_path, 1);
  $("#clearanceMetric").textContent = `${format(metric.minimum_clearance_m, 3)}m`;
  $("#lengthMetric").textContent = `${format(metric.path_length_m, 2)}m`;
  $("#selectedDetails").innerHTML = [
    `PATH ${state.selectedTrajectory + 1} · ${metric.valid ? "VALID" : "REJECTED"}`,
    `collision samples = ${metric.collision_sample_count}`,
    `bounds violations = ${metric.out_of_bounds_sample_count}`,
    `attitude violations = ${metric.attitude_violation_sample_count}`,
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
    if (!showAll && index !== state.selectedTrajectory) return;
    const selected = index === state.selectedTrajectory;
    const path = trajectory.poses_wxyz;
    const clearance = trajectory.clearance_m;
    for (let pointIndex = 1; pointIndex < path.length; pointIndex += 1) {
      const collision = clearance[pointIndex] <= 0 || clearance[pointIndex - 1] <= 0;
      drawLine([path[pointIndex - 1], path[pointIndex]], rect, collision ? "#ff5268" : colors[index % colors.length], selected ? 2.8 : 1.15);
    }
  });
  if ($("#animateVehicle").checked) {
    const selected = trajectories[state.selectedTrajectory];
    if (selected) {
      const index = Math.floor(((performance.now() - state.animationStart) / 35) % selected.poses_wxyz.length);
      const pose = selected.poses_wxyz[index], p = project(pose, rect);
      context.beginPath(); context.arc(p[0], p[1], 5, 0, Math.PI * 2); context.fillStyle = "#ffffff"; context.shadowColor = colors[state.selectedTrajectory % colors.length]; context.shadowBlur = 14; context.fill(); context.shadowBlur = 0;
    }
  }
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
  const target = event.button === 0 ? hitTarget(point.x, point.y) : null;
  if (target) {
    if (target.kind === "obstacle") {
      state.selectedObstacle = target.index;
      renderObstacles(); renderObstacleEditor();
    }
    invalidatePredictions();
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
  canvas.classList.toggle("drag-object", Boolean(hitTarget(point.x, point.y)));
});
canvas.addEventListener("mouseleave", () => { if (!state.drag) canvas.classList.remove("drag-object"); });
canvas.addEventListener("dblclick", (event) => {
  if (!state.scene) return;
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
$("#addObstacle").addEventListener("click", () => addObstacle());
$("#deleteObstacle").addEventListener("click", deleteObstacle);
$$(`[data-obstacle]`).forEach((input) => input.addEventListener("input", () => updateObstacle(input)));
$$(`[data-pose-editor] input`).forEach((input) => input.addEventListener("input", scheduleValidation));
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
    $("#sceneSeed").value = randomSeed();
    setBusy(false, "模型已加载，正在准备首个场景…");
    await generateScene();
  } catch (error) {
    $("#modelDevice").textContent = "MODEL ERROR";
    $("#inferenceStatus").textContent = error.message;
    $("#inferenceStatus").className = "inference-status error";
    toast(error.message, true);
  }
  draw();
}

initialize();
