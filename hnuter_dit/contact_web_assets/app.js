"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const DECODERS = ["off", "on", "on_smooth"];
const COLORS = { off: "#b397f7", on: "#5adddf", on_smooth: "#83e6af", actual: "#ffb061" };
const ACTIVE_STATES = ["starting", "running", "stopping", "initializing"];
const FAILURE_LABELS = {
  physical_collision: "机体碰撞", contact: "接触误差", normal: "法向误差", attitude: "姿态范围",
  workspace: "工作空间", tool_point_collision: "工具点碰撞", chart: "接触图域", forbidden_edge: "禁入区",
  psi_quality: "twist 质量", psi_continuity: "twist 连续性", cylinder_chart_continuity: "圆柱图域连续性",
  endpoint: "端点误差", shape_quality: "轨迹形状", projection_binding: "投影绑定",
  curvature: "曲率超限", source_curvature: "源轨迹曲率超限", shape: "轨迹形状", decoder_error: "解码错误",
  full_urdf_collision: "完整 URDF 碰撞", endpoints: "端点误差", forbidden: "禁入区", tool_point: "工具点碰撞",
  shape_backtracking: "轨迹回退超限", shape_chord_arc_ratio: "局部轨迹形状超限",
  physical_collision_unverified: "机体碰撞审核未完成", execution_gates_unverified: "执行审核未完成",
  raw_contact_reference_unsupported: "off 接触执行暂不支持", source_curvature_exceeds_8_inv_m: "源轨迹曲率超过 8 m⁻¹",
  dense_physical_collision: "密集段机体碰撞", dense_contact: "密集段接触", dense_normal: "密集段法向",
  dense_attitude: "密集段姿态", dense_workspace: "密集段空间", dense_tool_point_collision: "密集段工具碰撞",
  dense_chart: "密集段图域", dense_forbidden_edge: "密集段禁入区", invalid_input_or_backend_error: "输入 / 审核错误",
};
const state = {
  catalog: null, result: null, busy: false, revision: 0, candidateIndex: null,
  conditionOffset: 0, conditionTotal: 0, conditionRequest: 0, conditionLoading: false,
  progress: 0, playing: false, previousFrame: 0, dirty: true,
  camera: { yaw: -.72, pitch: .56, scale: 62, panX: 0, panY: 10, center: [0, 0, 2], view: "iso" },
  drag: null, execution: { state: "idle", executed_poses: [] }, executionPending: false,
  executionForPlan: null, executionTimer: null, pollRequest: 0,
  invalidation: Promise.resolve(), invalidationFailed: false,
};
const canvas = $("#sceneCanvas");
const ctx = canvas.getContext("2d");

const finite = (value) => value != null && value !== "" && Number.isFinite(Number(value));
const format = (value, digits = 2) => finite(value) ? Number(value).toFixed(digits) : "—";
const metric = (value, multiplier, digits, unit) => finite(value) ? `${format(Number(value) * multiplier, digits)} ${unit}` : "—";
const candidate = () => state.result?.candidates?.find((item) => item.index === state.candidateIndex);
const variant = (key = $("#decoder").value) => candidate()?.variants?.[key];
const activeExecution = () => ACTIVE_STATES.includes(state.execution?.state);
const locked = () => state.busy || state.conditionLoading || state.executionPending || activeExecution();
const executionAvailable = () => {
  const value = state.catalog?.execution;
  return Boolean(value?.available ?? value?.enabled) && !(value?.viewer_enabled && value?.viewer_available === false);
};
const executionMatchesPlan = (value = state.execution) => Boolean(state.result?.plan_id && value.plan_id === state.result.plan_id);
const showActualVehicle = () => $("#showActual").checked && executionMatchesPlan() && Boolean(state.execution.current_pose || state.execution.executed_poses?.length);
const array = (value) => Array.isArray(value) ? value : [];
const executionAudit = (value = variant()) => value?.execution_audit || {};
const executionEligible = (value = variant()) => Boolean(executionAudit(value).eligible ?? value?.execution_eligible);
const curvatureLimit = 8;
function curvatureInfo(value = variant()) {
  const values = array(value?.diagnostics?.curvature_inv_m).filter(finite).map(Number);
  const maximum = value?.metrics?.maximum_curvature_inv_m ?? (values.length ? Math.max(...values) : null);
  const count = values.filter((v) => v > curvatureLimit + 1e-8).length;
  return { maximum, count, warning: Boolean(executionAudit(value).curvature_warning || (finite(maximum) && Number(maximum) > curvatureLimit + 1e-8)) };
}

function status(message, kind = "") {
  $("#requestStatus").textContent = message;
  $("#requestStatus").className = `request-status${kind ? ` ${kind}` : ""}`;
}
function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast show${error ? " error" : ""}`;
  clearTimeout(node.hideTimer);
  node.hideTimer = setTimeout(() => { node.className = "toast"; }, 4500);
}
async function request(url, payload) {
  const response = await fetch(url, payload === undefined ? { cache: "no-store" } : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  });
  let data;
  try { data = await response.json(); } catch (_) { throw new Error(`服务返回非 JSON 响应（HTTP ${response.status}）`); }
  if (!response.ok) throw new Error(data.error || data.message || `HTTP ${response.status}`);
  return data;
}
function setBusy(value, message) {
  state.busy = value;
  if (message) status(message, value ? "busy" : "");
  refreshControls();
}
function option(value, label) {
  const node = document.createElement("option");
  node.value = value; node.textContent = label;
  return node;
}
function refreshControls() {
  const isLocked = locked(), ready = Boolean(state.catalog);
  $$(".controls input, .controls select").forEach((node) => { node.disabled = isLocked || !ready; });
  $("#conditions").disabled = isLocked || state.conditionLoading || !ready;
  $("#previousPage").disabled = isLocked || state.conditionLoading || state.conditionOffset === 0;
  $("#nextPage").disabled = isLocked || state.conditionLoading || state.conditionOffset + 100 >= state.conditionTotal;
  $("#loadReplay").disabled = isLocked || !ready || !$("#conditionId").value.trim();
  $("#sampleModel").disabled = isLocked || !ready || !$("#conditionId").value.trim();
  $("#candidate").disabled = isLocked || !state.result?.candidates?.length;
  $("#decoder").disabled = state.executionPending || activeExecution();
  $("#allowCurvatureFailure").disabled = isLocked;
  ["#executionSpeed", "#executionAcceleration"].forEach((id) => { $(id).disabled = isLocked || !ready; });
  $$(".variant-card").forEach((node) => { node.disabled = state.executionPending || activeExecution(); });
  $("#playGeometry").disabled = !variant()?.display_poses_wxyz?.length;
  $("#progress").disabled = !variant()?.display_poses_wxyz?.length;
  $("#exportResult").disabled = !state.result;
  $("#previewExecution").disabled = isLocked || !executionAvailable() || !state.result?.plan_id;
  const selected = variant(), audit = executionAudit(selected), warning = curvatureInfo(selected).warning;
  const decoder = $("#decoder").value, allowCurvature = $("#allowCurvatureFailure").checked;
  const eligible = executionEligible(selected) && (!warning || allowCurvature);
  $("#startExecution").disabled = isLocked || !executionAvailable() || !state.result?.plan_id || !eligible;
  $("#startExecution").textContent = warning ? "曲率诊断飞行" : `执行 ${decoder}`;
  $("#startExecution").classList.toggle("diagnostic", warning);
  const canCloseViewer = Boolean(state.execution?.viewer_open);
  $("#stopExecution").disabled = state.executionPending || !(activeExecution() || state.execution?.state === "preview" || canCloseViewer);
  $("#stopExecution").textContent = !activeExecution() && canCloseViewer ? "关闭窗口" : "停止";
  const blockers = array(audit.blocking_failures).map((name) => FAILURE_LABELS[name] || name);
  $("#eligibility").textContent = !state.result ? "载入轨迹后，执行当前选择的 decoder。" :
    selected?.error ? `${decoder} 没有可执行轨迹：${selected.error}` :
      warning && !allowCurvature ? `${decoder} 曲率超限，当前已关闭诊断飞行。可勾选“曲率超标允许诊断飞行”。` :
      eligible ? warning ? `${decoder} 曲率超限；可启动诊断飞行。原始失败判定保留。` : `${decoder} 已通过源轨迹审核；启动时将再次审核实际执行曲线。` :
        `${decoder} 禁止执行：${blockers.length ? blockers.join("、") : "未通过执行审核"}。可预览场景或选择其他 decoder。`;
  $("#eligibility").classList.toggle("warning", warning);
  $("#executionGeometry").textContent = !selected ? "" : decoder === "on_smooth" ?
    "on_smooth 执行 intrinsic cubic 曲线，并独立审核实际执行曲线；128 节点评分仍按原始方案保留。" :
    decoder === "on" ? "on 执行原始 intrinsic 分段接触边，每个节点停车，不做几何平滑。" :
      audit.geometry === "unsupported_raw_contact_reference" ? "off 可查看原始输出；当前接触闭环支持 on 与 on_smooth。" :
        "off 执行原始机体折线与姿态插值，每个节点停车；接触及其他安全门限仍需通过。";
}
function invalidateResult() {
  if (activeExecution() || state.executionPending) return;
  const oldPlan = state.result?.plan_id;
  state.revision += 1; state.result = null; state.candidateIndex = null;
  state.progress = 0; state.playing = false; state.executionForPlan = null;
  $("#playGeometry").textContent = "▶";
  $("#candidate").replaceChildren(option("", "未载入"));
  $("#candidateProtocol").textContent = "";
  $("#provenance").textContent = "条件已更改";
  $("#sceneTitle").textContent = "任务接触轨迹";
  $("#emptyScene").hidden = false;
  status("条件已更改，请读取已保存轨迹或重新在线推理。");
  renderMetrics(); executionState(state.execution); updateProgress(); refreshControls(); state.dirty = true;
  if (oldPlan) {
    state.invalidation = state.invalidation.then(() => request("/api/plan/invalidate", { plan_id: oldPlan }))
      .then(() => { state.invalidationFailed = false; })
      .catch((error) => { state.invalidationFailed = true; status(`旧计划失效请求失败：${error.message}`, "error"); toast(error.message, true); });
  }
}

async function loadConditions(offset = 0, preserveId = null) {
  const revision = ++state.conditionRequest;
  state.conditionLoading = true; refreshControls();
  try {
    const result = await request("/api/conditions", {
      family: $("#family").value, kind: $("#kind").value, difficulty: $("#difficulty").value, offset, limit: 100,
    });
    if (revision !== state.conditionRequest) return;
    state.conditionOffset = result.offset ?? offset; state.conditionTotal = result.total;
    $("#conditions").replaceChildren(...array(result.items).map((item) => {
      const shortId = String(item.condition_id).split("/").slice(-2).join("/");
      const node = option(item.condition_id, `${shortId} · ${item.kind === "plane" ? "平面" : "圆柱"} · ${item.difficulty}`);
      node.title = item.condition_id;
      return node;
    }));
    $("#conditionTotal").textContent = `${result.total.toLocaleString()} 条`;
    $("#pageLabel").textContent = result.total ? `${Math.floor(state.conditionOffset / 100) + 1} / ${Math.ceil(result.total / 100)}` : "0 / 0";
    const current = preserveId || result.items?.[0]?.condition_id || "";
    $("#conditionId").value = current;
    $("#conditions").value = current;
    if (!result.items?.length) status("当前筛选没有匹配条件；可以直接输入完整 Condition ID。");
  } catch (error) {
    if (revision === state.conditionRequest) { status(error.message, "error"); toast(error.message, true); }
  } finally {
    if (revision === state.conditionRequest) { state.conditionLoading = false; refreshControls(); }
  }
}
async function loadPlan(source) {
  if (locked()) return;
  const conditionId = $("#conditionId").value.trim();
  if (!conditionId) { toast("请选择一个测试条件。", true); return; }
  const count = Number($("#sampleCount").value), samplingSteps = Number($("#samplingSteps").value);
  if (source === "sample" && (!Number.isInteger(samplingSteps) || samplingSteps < 1 || samplingSteps > 100)) {
    toast("DDIM 步数应为 1 至 100 的整数。", true); return;
  }
  if (state.result) invalidateResult();
  const revision = state.revision;
  setBusy(true, source === "replay" ? "正在读取保存的预测与评分…" : "正在执行模型推理、接触解码与完整 URDF 审核…");
  try {
    await state.invalidation;
    if (state.invalidationFailed) throw new Error("旧计划失效失败，请刷新页面后重试。");
    const payload = { model_id: $("#model").value, condition_id: conditionId, scenario: $("#scenario").value };
    if (source === "sample") Object.assign(payload, { count, sampling_steps: samplingSteps });
    const result = await request(`/api/${source}`, payload);
    if (revision !== state.revision) return;
    if (!result.condition || !Array.isArray(result.candidates) || !result.candidates.length) throw new Error("服务未返回有效任务和候选轨迹。");
    applyResult(result);
    const failed = result.candidates.reduce((sum, item) => sum + DECODERS.filter((key) => item.variants?.[key]?.error).length, 0);
    status(`已载入 ${result.candidates.length} 个候选${failed ? `；${failed} 个 decoder 解码 / 审核失败，其余轨迹仍可查看` : ""}。选择 decoder 查看轨迹与审核详情。`, failed ? "warning" : "");
  } catch (error) {
    const message = `${source === "replay" ? "保存轨迹读取失败" : "在线推理失败"}：${error.message}`;
    status(message, "error"); toast(message, true);
  }
  finally { setBusy(false); }
}
function applyResult(result) {
  state.result = result;
  state.candidateIndex = result.selected_candidate_index ?? result.candidates[0].index;
  if (!result.candidates.some((item) => item.index === state.candidateIndex)) state.candidateIndex = result.candidates[0].index;
  state.executionForPlan = null; state.playing = false; state.progress = 0;
  $("#playGeometry").textContent = "▶";
  $("#candidate").replaceChildren(...result.candidates.map((item) => {
    const eligible = DECODERS.filter((key) => executionEligible(item.variants?.[key]));
    return option(item.index, `候选 ${item.index}${eligible.length ? ` · ${eligible.join(" / ")} 可执行` : ""}`);
  }));
  $("#candidate").value = String(state.candidateIndex);
  const g = result.condition.surface_geometry;
  $("#sceneTitle").textContent = `${g.type === "plane" ? "平面接触" : "圆柱接触"} · ${String(result.condition_id).split("/").slice(-2).join("/")}`;
  $("#provenance").textContent = String(result.provenance?.source || "").startsWith("online") ? "在线推理" : "保存预测";
  $("#provenance").title = JSON.stringify(result.provenance || {});
  const online = String(result.provenance?.source || "").startsWith("online");
  $("#candidateProtocol").textContent = online ? `在线生成 ${result.candidates.length} 个候选。` :
    result.candidates.length === 1 ? "保存协议：当前条件保存 1 个候选（candidate 0），不是读取失败。K=8 仅覆盖冻结子集；可在线生成 8 个候选。" :
      `保存协议：当前条件属于 K=8 冻结子集，已载入 ${result.candidates.length} 个候选。`;
  $("#emptyScene").hidden = true;
  fitCamera(); renderMetrics(); executionState(state.execution); updateProgress(); refreshControls(); state.dirty = true;
}
function renderMetrics() {
  $("#variantCards").replaceChildren(...DECODERS.map((key) => {
    const value = variant(key), button = document.createElement("button");
    button.className = `variant-card${$("#decoder").value === key ? " selected" : ""}`;
    button.style.setProperty("--variant-color", COLORS[key]);
    const name = document.createElement("span"), result = document.createElement("span");
    name.className = "variant-name"; name.textContent = key;
    const warning = curvatureInfo(value).warning;
    result.className = `variant-status ${value?.error ? "bad" : warning ? "warning" : value?.metrics?.candidate_success ? "good" : value ? "bad" : ""}`;
    result.textContent = !value ? "未载入" : value.error ? "解码 / 审核错误" : warning ? executionEligible(value) ? "曲率超限 · 可诊断飞行" : "曲率超限 · 执行受阻" : value.metrics?.candidate_success ? "完整成功" : "失败";
    button.append(name, result);
    button.addEventListener("click", () => { if (state.executionPending || activeExecution()) return; $("#decoder").value = key; changeVariant(); });
    return button;
  }));
  const value = variant(), m = value?.metrics || {};
  $("#metricContact").textContent = metric(m.contact_error_max_m ?? m.maximum_contact_error_m, 1000, 3, "mm");
  $("#metricNormal").textContent = metric(m.normal_error_max_rad ?? m.maximum_normal_error_rad, 180 / Math.PI, 3, "°");
  $("#metricCurvature").textContent = metric(m.maximum_curvature_inv_m, 1, 2, "m⁻¹");
  $("#metricClearance").textContent = metric(m.minimum_clearance_m, 1, 4, "m");
  $("#metricDense").textContent = !value ? "—" : m.dense_checked ?
    (m.dense_validation_completed === false ? "已进入 · 未完成" : m.candidate_success ? "完成 · 通过" : "已进入 · 失败") : "未执行（前置 gate 失败）";
  $("#metricSamples").textContent = value ? `${array(value.display_poses_wxyz).length} / ${array(value.poses_wxyz).length} 节点` : "—";
  $("#chartScope").textContent = value ? `候选 ${state.candidateIndex} · ${$("#decoder").value}` : "—";
  $("#loadedState").textContent = !value ? "" : value.error ? "该 decoder 解码 / 审核出错；可以选择其他已载入的 decoder。" :
    m.candidate_success ? "轨迹已载入，正式评测通过。" : "轨迹已载入，正式评测失败；仍可播放并查看失败位置。";
  const curvature = curvatureInfo(value);
  $("#curvatureWarning").hidden = !curvature.warning;
  $("#curvatureWarningDetail").textContent = `源轨迹最大曲率 ${metric(curvature.maximum, 1, 2, "m⁻¹")}；128 节点中 ${curvature.count} 个节点超限。${array(executionAudit(value).warnings).map((key) => FAILURE_LABELS[key] || key).join("；")}`;
  $("#metricCurvature").classList.toggle("warning", curvature.warning);
  const failures = $("#failures"); failures.replaceChildren(); failures.className = "failures";
  if (!value) failures.textContent = "等待载入";
  else if (value.error || m.evaluation_error) failures.textContent = value.error || m.evaluation_error;
  else if (!m.failures?.length) { failures.textContent = m.candidate_success ? "全部 gate 通过" : "没有失败详情"; failures.className = "failures ok"; }
  else m.failures.forEach((key) => { const node = document.createElement("span"); node.className = "failure"; node.textContent = FAILURE_LABELS[key] || key; node.title = key; failures.append(node); });
  renderCharts();
}
function changeVariant() {
  state.playing = false; $("#playGeometry").textContent = "▶";
  renderMetrics(); refreshControls(); updateProgress(); state.dirty = true;
}
function updateProgress() {
  $("#progress").value = String(Math.round(state.progress * 1000));
  $("#progressLabel").textContent = `${(state.progress * 100).toFixed(1)}%`;
  state.dirty = true;
}

function executionState(value) {
  state.execution = value;
  const attached = executionMatchesPlan(value);
  const detail = value.contact_metrics || {};
  const trial = value.execution_trial || {};
  const timing = attached ? value.retiming || trial.timing || {} : {}, validation = timing.kinematic_validation || {};
  const dynamics = attached ? value.motion_metrics || {} : {};
  const executedDecoder = trial.decoder || value.decoder || value.execution_decoder;
  const diagnostic = Boolean(trial.mode === "curvature_diagnostic" || value.allow_curvature_failure || value.execution_mode === "curvature_diagnostic" || value.reference_audit?.curvature_warning || value.reference_audit?.mode === "curvature_diagnostic");
  const names = { idle: "等待任务", preview: "场景预览", starting: "正在初始化", initializing: "正在初始化", running: "闭环执行中", stopping: "正在停止", stopped: "执行已停止", completed: "执行已结束", failed: "执行失败", error: "执行错误" };
  $("#executionStatus").textContent = value.plan_id && !attached ?
    (activeExecution() ? "另一计划正在执行，请等待完成或停止后载入当前任务。" : "当前任务尚未预览或执行。") :
    `${names[value.state] || value.state || "等待任务"}${attached && diagnostic ? " · 曲率诊断飞行" : ""}${value.viewer_open && !activeExecution() ? " · 物理窗口仍打开" : ""}${value.message ? ` · ${value.message}` : ""}${value.error ? ` · ${value.error}` : ""}`;
  $("#executionStatus").className = `execution-status${attached && (value.error || ["failed", "error"].includes(value.state)) ? " error" : ""}`;
  $("#executionProgress").value = attached ? Math.max(0, Math.min(1, Number(value.progress) || 0)) : 0;
  $("#executionDecoder").textContent = attached ? `${executedDecoder || "—"} / ${diagnostic ? "曲率诊断" : "严格审核"}` : "—";
  $("#simulationTime").textContent = attached ? `${metric(value.simulation_time_s, 1, 2, "s")} / ${metric(timing.duration_s, 1, 2, "s")}` : "—";
  $("#timingMethod").textContent = timing.method || "—";
  $("#timingScale").textContent = attached ? `${metric(timing.speed_mps, 1, 2, "m/s")} / ${format(timing.time_spline_safety_scale ?? timing.time_scale ?? (timing.stops_at_waypoints ? 1 : null), 2)}×` : "—";
  $("#waypointStops").textContent = timing.stops_at_waypoints == null ? "—" : timing.stops_at_waypoints ? "是 · 保留原始边" : "否 · 连续通过";
  $("#referenceAcceleration").textContent = attached ? `${metric(validation.max_linear_acceleration_mps2 ?? timing.peak_linear_acceleration_mps2, 1, 2, "m/s²")} / ${metric(validation.estimated_max_linear_jerk_mps3 ?? timing.peak_linear_jerk_mps3, 1, 2, "m/s³")}` : "—";
  $("#actualAcceleration").textContent = attached ? `${metric(dynamics.max_linear_acceleration_mps2 ?? value.maximum_linear_acceleration_mps2, 1, 2, "m/s²")} / ${metric(dynamics.max_linear_jerk_mps3 ?? value.maximum_linear_jerk_mps3, 1, 2, "m/s³")}` : "—";
  $("#actualAngular").textContent = attached ? `${metric(dynamics.max_angular_speed_rps ?? value.maximum_angular_speed_rps, 1, 2, "rad/s")} / ${metric(dynamics.max_angular_acceleration_rps2 ?? value.maximum_angular_acceleration_rps2, 1, 2, "rad/s²")}` : "—";
  ["#actualAcceleration", "#actualAngular"].forEach((id) => { $(id).title = dynamics.sampling_scope || "实际运动量由控制采样遥测估计。"; });
  $("#trackingError").textContent = attached ? metric(value.tracking_error_m, 1, 3, "m") : "—";
  $("#trackingSummary").textContent = attached ? `${metric(value.maximum_tracking_error_m, 1, 3, "m")} / ${metric(value.position_rmse_m, 1, 3, "m")}` : "—";
  $("#actualContact").textContent = attached ? metric(detail.max_contact_error_m, 1000, 3, "mm") : "—";
  $("#actualNormal").textContent = attached ? metric(detail.max_normal_error_deg, 1, 3, "°") : "—";
  const yesNo = (flag) => flag === true ? "通过" : flag === false ? "失败" : "待定";
  $("#executionResult").textContent = attached ? `飞行${yesNo(value.flight_success ?? value.success)} / 任务${yesNo(detail.task_success)}` : "—";
  $("#executionTimingNote").textContent = timing.stops_at_waypoints ?
    "本次逐节点停车，速度在转角处降为零，会减轻曲率造成的动态负担。实际加速度 / jerk 为控制采样估计；不能代表固定速度穿越转角的结果。" :
    "TOPP-RA 会根据速度、加速度与 jerk 限制减速，时间倍率越大代表执行越慢。请结合参考时长与实际遥测比较曲率影响；该实验不采用固定速度。";
  const matchesVariant = attached && executedDecoder === $("#decoder").value &&
    (trial.candidate_index == null || Number(trial.candidate_index) === state.candidateIndex);
  if (matchesVariant && diagnostic && value.reference_audit?.curvature_warning) {
    $("#curvatureWarning").hidden = false;
    const source = curvatureInfo();
    const maximum = value.reference_audit.analytic_curve_sampled_maximum_curvature_inv_m ?? value.reference_audit.maximum_curvature_inv_m;
    const curveNote = value.reference_audit.continuous_corner_curvature_defined === false ?
      "原始折角保留，折角空间曲率没有光滑定义；每个节点停车。边内部曲率不替代源节点的失败判定。" :
      finite(maximum) && maximum > curvatureLimit ? `实际 cubic 曲线最大采样曲率 ${metric(maximum, 1, 2, "m⁻¹")}，同样超过门限。` :
        "源节点的曲率失败判定保留。";
    $("#curvatureWarningDetail").textContent = `源轨迹最大曲率 ${metric(source.maximum, 1, 2, "m⁻¹")}；${source.count} 个源节点超限。${curveNote}本次以诊断模式执行。`;
  }
  refreshControls(); state.dirty = true;
}
async function executionAction(action) {
  if (state.executionPending || (action !== "stop" && locked())) return;
  if (action !== "stop" && !state.result?.plan_id) return;
  const timeOptions = { speed_mps: Number($("#executionSpeed").value), max_acceleration_mps2: Number($("#executionAcceleration").value) };
  if (action === "start" && (!finite(timeOptions.speed_mps) || timeOptions.speed_mps < .05 || timeOptions.speed_mps > 2 ||
      !finite(timeOptions.max_acceleration_mps2) || timeOptions.max_acceleration_mps2 < .1 || timeOptions.max_acceleration_mps2 > 4)) {
    toast("参考速度上限应为 0.05 至 2 m/s，加速度上限应为 0.1 至 4 m/s²。", true); return;
  }
  state.executionPending = true; refreshControls();
  const planId = state.result?.plan_id;
  const revision = state.revision;
  state.pollRequest += 1;
  try {
    const payload = action === "stop" ? {} : { plan_id: planId };
    if (action === "start") {
      Object.assign(payload, { candidate_index: state.candidateIndex, decoder: $("#decoder").value });
      payload.allow_curvature_failure = $("#allowCurvatureFailure").checked;
      payload.options = timeOptions;
    }
    const value = await request(`/api/execution/${action}`, payload);
    if (revision !== state.revision || (action !== "stop" && planId !== state.result?.plan_id)) return;
    state.executionForPlan = planId;
    if (action === "start") { state.playing = false; $("#playGeometry").textContent = "▶"; }
    executionState(value);
    toast(action === "stop" ? "已请求停止执行或关闭物理窗口。" : action === "preview" ? "已请求同步任务场景。" : `已请求启动 ${payload.decoder} 闭环执行${payload.allow_curvature_failure ? "，允许曲率诊断" : ""}。`);
  } catch (error) {
    $("#executionStatus").textContent = error.message; $("#executionStatus").className = "execution-status error"; toast(error.message, true);
  } finally { state.executionPending = false; refreshControls(); pollExecutionSoon(); }
}
function pollExecutionSoon() {
  clearTimeout(state.executionTimer);
  state.executionTimer = setTimeout(pollExecution, activeExecution() ? 350 : 1500);
}
async function pollExecution() {
  const id = ++state.pollRequest;
  try {
    const value = await request("/api/execution/status");
    if (id !== state.pollRequest) return;
    if (value.plan_id && value.plan_id === state.result?.plan_id) state.executionForPlan = value.plan_id;
    executionState(value);
  } catch (error) {
    $("#executionStatus").textContent = `遥测读取失败：${error.message}`;
    $("#executionStatus").className = "execution-status error";
  } finally { pollExecutionSoon(); }
}

function quaternionMatrix(q) {
  let [w, x, y, z] = q || [1, 0, 0, 0];
  const n = Math.hypot(w, x, y, z) || 1; [w, x, y, z] = [w / n, x / n, y / n, z / n];
  return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]];
}
function transform(local, origin, rotation) {
  return origin.map((v, i) => v + rotation[i].reduce((sum, r, j) => sum + r * local[j], 0));
}
function project(point, rect) {
  const c = state.camera, [x, y, z] = point.map((v, i) => v - c.center[i]);
  const cy = Math.cos(c.yaw), sy = Math.sin(c.yaw), cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
  const rx = cy * x - sy * y, ry = sy * x + cy * y;
  return [rect.width / 2 + c.panX + rx * c.scale, rect.height / 2 + c.panY - (cp * z - sp * ry) * c.scale, sp * z + cp * ry];
}
function resizeCanvas(node) {
  const rect = node.getBoundingClientRect(), ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * ratio)), height = Math.max(1, Math.round(rect.height * ratio));
  if (node.width !== width || node.height !== height) { node.width = width; node.height = height; }
  node.getContext("2d").setTransform(ratio, 0, 0, ratio, 0, 0);
  return rect;
}
function line(points, color, width = 1, dash = [], rect = canvas.getBoundingClientRect()) {
  if (points.length < 2) return;
  ctx.beginPath(); ctx.setLineDash(dash);
  points.forEach((p, i) => { const v = project(p.slice(0, 3), rect); if (i) ctx.lineTo(v[0], v[1]); else ctx.moveTo(v[0], v[1]); });
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke(); ctx.setLineDash([]);
}
function pointMarker(point, color, radius, label, rect, triangle = false) {
  const p = project(point.slice(0, 3), rect);
  ctx.beginPath();
  if (triangle) { ctx.moveTo(p[0], p[1] - radius); ctx.lineTo(p[0] + radius, p[1] + radius); ctx.lineTo(p[0] - radius, p[1] + radius); ctx.closePath(); }
  else ctx.arc(p[0], p[1], radius, 0, Math.PI * 2);
  ctx.fillStyle = color; ctx.fill();
  if (label) { ctx.font = "10px system-ui"; ctx.fillText(label, p[0] + radius + 4, p[1] - 5); }
}
const BOX_FACES = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]];
const BOX_EDGES = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]];
function boxVertices(center, size, rotation) {
  return Array.from({ length: 8 }, (_, bits) => transform(size.map((v, i) => (bits & (1 << i) ? .5 : -.5) * v), center, rotation));
}
function boxMesh(box, faces) {
  const verts = boxVertices(box.position, box.size, quaternionMatrix(box.quaternion_wxyz));
  const floor = box.role === "floor", support = box.role === "task_support";
  BOX_FACES.forEach((indices) => faces.push({ vertices: indices.map((i) => verts[i]), fill: floor ? "#71839410" : support ? "#7a9dc12f" : "#ba875b22", stroke: floor ? "#47607140" : support ? "#6b89a185" : "#a0785485" }));
}
function surfacePoint(u, v, g, rotation) {
  const local = g.type === "plane" ? [u, v, 0] : [g.radius_m * Math.cos(u / g.radius_m), g.radius_m * Math.sin(u / g.radius_m), v];
  return transform(local, g.origin, rotation);
}
function surfaceMesh(condition, faces) {
  const g = condition.surface_geometry, bounds = condition.surface_chart.bounds, rotation = quaternionMatrix(g.quaternion_wxyz);
  const [u, v] = bounds;
  if (g.type === "plane") {
    const center = transform([(u[0] + u[1]) / 2, (v[0] + v[1]) / 2, -(g.thickness_m || .08) / 2], g.origin, rotation);
    const verts = boxVertices(center, [u[1] - u[0], v[1] - v[0], g.thickness_m || .08], rotation);
    BOX_FACES.forEach((indices) => faces.push({ vertices: indices.map((i) => verts[i]), fill: "#3d8d9137", stroke: "#56b7b78a" }));
  } else {
    // The target remains a complete solid cylinder; the contact chart is an overlay.
    const segments = 64, low = v[0], high = v[1], r = g.radius_m;
    const ring = (z) => Array.from({ length: segments }, (_, i) => transform([r * Math.cos(i * Math.PI * 2 / segments), r * Math.sin(i * Math.PI * 2 / segments), z], g.origin, rotation));
    const bottom = ring(low), top = ring(high);
    faces.push({ vertices: bottom, fill: "#387e8340", stroke: "#55a7b080" }, { vertices: top, fill: "#387e8340", stroke: "#55a7b080" });
    for (let i = 0; i < segments; i += 1) faces.push({ vertices: [bottom[i], bottom[(i + 1) % segments], top[(i + 1) % segments], top[i]], fill: "#387e8328", stroke: "#47959c25" });
  }
  const chartSegments = g.type === "cylinder" ? 40 : 8;
  for (let i = 0; i < chartSegments; i += 1) {
    const a = u[0] + (u[1] - u[0]) * i / chartSegments, b = u[0] + (u[1] - u[0]) * (i + 1) / chartSegments;
    faces.push({ vertices: [[a, v[0]], [b, v[0]], [b, v[1]], [a, v[1]]].map(([x, y]) => surfacePoint(x, y, g, rotation)), fill: "#5cdbb538", stroke: "#6cdfbf26" });
  }
  array(condition.surface_chart.forbidden_regions).forEach(([u0, u1, v0, v1]) => {
    for (let i = 0; i < chartSegments; i += 1) {
      const a = u0 + (u1 - u0) * i / chartSegments, b = u0 + (u1 - u0) * (i + 1) / chartSegments;
      faces.push({ vertices: [[a, v0], [b, v0], [b, v1], [a, v1]].map(([x, y]) => surfacePoint(x, y, g, rotation)), fill: "#f06a7255", stroke: "#f06a7266" });
    }
  });
}
function paintFaces(faces, rect) {
  faces.map((face) => ({ ...face, projected: face.vertices.map((p) => project(p, rect)) }))
    .sort((a, b) => a.projected.reduce((s, p) => s + p[2], 0) / a.projected.length - b.projected.reduce((s, p) => s + p[2], 0) / b.projected.length)
    .forEach((face) => {
      ctx.beginPath(); face.projected.forEach((p, i) => { if (i) ctx.lineTo(p[0], p[1]); else ctx.moveTo(p[0], p[1]); });
      ctx.closePath(); ctx.fillStyle = face.fill; ctx.strokeStyle = face.stroke; ctx.lineWidth = .55; ctx.fill(); ctx.stroke();
    });
}
function drawGrid(condition, rect) {
  const b = condition?.environment?.bounds || { min: [-4, -4, 0], max: [4, 4, 4] };
  for (let x = Math.ceil(b.min[0]); x <= b.max[0]; x += 1) line([[x, b.min[1], b.min[2]], [x, b.max[1], b.min[2]]], "#30455445", .7, [], rect);
  for (let y = Math.ceil(b.min[1]); y <= b.max[1]; y += 1) line([[b.min[0], y, b.min[2]], [b.max[0], y, b.min[2]]], "#30455445", .7, [], rect);
  const center = b.min.map((v, i) => (v + b.max[i]) / 2), size = b.max.map((v, i) => v - b.min[i]);
  const corners = boxVertices(center, size, quaternionMatrix([1, 0, 0, 0]));
  BOX_EDGES.forEach(([a, c]) => line([corners[a], corners[c]], "#40576865", .65, [3, 5], rect));
}
function vehicle(pose, tip, color, rect, label) {
  if (!pose || pose.length < 7) return;
  const rotation = quaternionMatrix(pose.slice(3, 7)), body = pose.slice(0, 3);
  const local = (p) => transform(p, body, rotation);
  const corners = boxVertices(body, [.25, .18, .1], rotation);
  BOX_EDGES.forEach(([a, b]) => line([corners[a], corners[b]], color, 1.2, [], rect));
  const arms = [[-.2, -.2, 0], [.2, .2, 0], [-.2, .2, 0], [.2, -.2, 0]].map(local);
  line(arms.slice(0, 2), color, 2, [], rect); line(arms.slice(2), color, 2, [], rect);
  arms.forEach((p) => { const v = project(p, rect); ctx.beginPath(); ctx.ellipse(v[0], v[1], 5, 3, 0, 0, Math.PI * 2); ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.stroke(); });
  [[.34, 0, 0], [0, .34, 0], [0, 0, .34]].forEach((p, i) => line([body, local(p)], ["#ff7180", "#83e6af", "#65a9ff"][i], 1.4, [], rect));
  if (tip) { line([body, tip], color, 2, [], rect); pointMarker(tip, color, 4, "工具点", rect); }
  pointMarker(body, color, 3.5, label, rect);
}
function drawTrajectories(rect) {
  const selected = $("#decoder").value, keys = $("#overlay").checked ? DECODERS : [selected];
  keys.forEach((key) => {
    const value = variant(key);
    if (!value) return;
    const highlight = key === selected;
    ctx.globalAlpha = highlight ? 1 : .65;
    if ($("#showBody").checked) line(array(value.display_poses_wxyz), COLORS[key], highlight ? 1.7 : 1.05, [5, 4], rect);
    if ($("#showTip").checked) line(array(value.display_tool_points), COLORS[key], highlight ? 2.7 : 1.5, [], rect);
    ctx.globalAlpha = 1;
  });
  const value = variant();
  if (!value) return;
  const path = array(value.display_poses_wxyz), tips = array(value.display_tool_points);
  if (path.length) {
    const index = displayIndex(value, state.progress);
    if (!showActualVehicle()) vehicle(path[index], tips[index], COLORS[selected], rect, `${selected} · ${index + 1}/${path.length}`);
    pointMarker(path[0], "#e0f7ed", 4, "START", rect); pointMarker(path[path.length - 1], "#c5b3fa", 4, "GOAL", rect);
  }
  if ($("#showMarkers").checked) {
    const diagnostic = value.diagnostics || {}, nodes = array(value.poses_wxyz);
    nodes.forEach((p, i) => {
      const clearance = diagnostic.clearance_m?.[i], curvature = diagnostic.curvature_inv_m?.[i];
      if (finite(clearance) && clearance <= 0) pointMarker(p, "#ff7180", 3.5, "", rect);
      else if (finite(curvature) && curvature > 8) pointMarker(p, "#f2c064", 3, "", rect, true);
    });
  }
}
function displayIndex(value, progress) {
  const samples = array(value.display_progress), count = array(value.display_poses_wxyz).length;
  if (!samples.length || samples.length !== count) return Math.round(progress * Math.max(0, count - 1));
  let low = 0, high = samples.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (samples[middle] < progress) low = middle + 1; else high = middle;
  }
  return low > 0 && progress - samples[low - 1] < samples[low] - progress ? low - 1 : low;
}
function drawActual(rect) {
  if (!$("#showActual").checked || !executionMatchesPlan()) return;
  const e = state.execution, poses = array(e.executed_poses), tips = array(e.contact_metrics?.tip_points);
  line(poses, COLORS.actual, 2.8, [3, 2], rect); line(tips, COLORS.actual, 3, [], rect);
  const actual = e.current_pose || poses[poses.length - 1];
  let tip = tips[tips.length - 1];
  if (!tip && actual?.length >= 7) tip = transform(state.result.condition.tool_extrinsics.position_body, actual.slice(0, 3), quaternionMatrix(actual.slice(3, 7)));
  vehicle(actual, tip, COLORS.actual, rect, "MuJoCo 实际");
  if (e.reference_pose) pointMarker(e.reference_pose, "#ffe3b0", 3.5, "物理参考", rect);
}
function drawScene() {
  const rect = resizeCanvas(canvas);
  ctx.clearRect(0, 0, rect.width, rect.height);
  const condition = state.result?.condition;
  drawGrid(condition, rect);
  if (condition) {
    const faces = [];
    [...array(condition.environment?.obstacles), ...array(condition.surface_geometry.supports)].forEach((box) => boxMesh(box, faces));
    surfaceMesh(condition, faces); paintFaces(faces, rect);
    const g = condition.surface_geometry, rotation = quaternionMatrix(g.quaternion_wxyz), b = condition.surface_chart.bounds;
    const boundary = (u, v) => surfacePoint(u, v, g, rotation);
    const step = g.type === "cylinder" ? 60 : 1;
    [b[1][0], b[1][1]].forEach((v) => line(Array.from({ length: step + 1 }, (_, i) => boundary(b[0][0] + (b[0][1] - b[0][0]) * i / step, v)), "#81e4c9", 1.3, [], rect));
    [b[0][0], b[0][1]].forEach((u) => line([boundary(u, b[1][0]), boundary(u, b[1][1])], "#81e4c9", 1.3, [], rect));
    pointMarker(g.origin, "#7eabb3", 2.5, "曲面 frame", rect);
    drawTrajectories(rect); drawActual(rect);
  }
  $("#cameraLabel").textContent = `${state.camera.view === "top" ? "俯视" : state.camera.view === "front" ? "正视" : "立体"} · 拖拽旋转 / Shift 平移 / 滚轮缩放`;
}
function fitCamera() {
  const b = state.result?.condition?.environment?.bounds || { min: [-4, -4, 0], max: [4, 4, 4] };
  const c = state.camera;
  c.center = b.min.map((v, i) => (v + b.max[i]) / 2); c.panX = 0; c.panY = -10;
  const rect = canvas.getBoundingClientRect();
  c.scale = Math.max(15, Math.min(rect.width / 12, rect.height / 7)); state.dirty = true;
}
function setView(view) {
  state.camera.view = view;
  Object.assign(state.camera, view === "top" ? { yaw: 0, pitch: Math.PI / 2 } : view === "front" ? { yaw: 0, pitch: 0 } : { yaw: -.72, pitch: .56 });
  $$("[data-view]").forEach((node) => node.classList.toggle("active", node.dataset.view === view));
  fitCamera();
}

function renderChart(id, key, multiplier, threshold) {
  const node = $(id), context = node.getContext("2d"), rect = resizeCanvas(node);
  context.clearRect(0, 0, rect.width, rect.height);
  const selected = $("#decoder").value, keys = $("#overlay").checked ? DECODERS : [selected];
  const series = keys.map((decoder) => ({ decoder, values: array(variant(decoder)?.diagnostics?.[key]).map((v) => finite(v) ? Number(v) * multiplier : null) }));
  const values = series.flatMap((row) => row.values.filter((v) => v !== null));
  const left = 34, right = rect.width - 8, top = 13, bottom = rect.height - 22;
  const low = Math.min(0, threshold, ...values), high = Math.max(threshold * 1.15, ...values, low + .001);
  const y = (v) => bottom - (v - low) / (high - low) * (bottom - top);
  context.font = "9px ui-monospace, monospace"; context.strokeStyle = "#294051"; context.lineWidth = .6;
  [low, (low + high) / 2, high].forEach((v) => {
    const pos = y(v); context.beginPath(); context.moveTo(left, pos); context.lineTo(right, pos); context.stroke();
    context.fillStyle = "#8198ab"; context.textAlign = "right"; context.fillText(Math.abs(v) >= 100 ? format(v, 0) : format(v, Math.abs(high) < .1 ? 3 : 1), left - 5, pos + 3);
  });
  context.setLineDash([3, 4]); context.strokeStyle = "#df9e606f"; context.beginPath(); context.moveTo(left, y(threshold)); context.lineTo(right, y(threshold)); context.stroke(); context.setLineDash([]);
  series.forEach((row) => {
    context.strokeStyle = COLORS[row.decoder]; context.lineWidth = row.decoder === selected ? 1.7 : 1.0; context.globalAlpha = row.decoder === selected ? 1 : .6;
    context.beginPath(); let connected = false;
    row.values.forEach((v, i) => {
      if (v === null) { connected = false; return; }
      const px = left + i / Math.max(1, row.values.length - 1) * (right - left), py = y(v);
      if (connected) context.lineTo(px, py); else context.moveTo(px, py); connected = true;
    }); context.stroke(); context.globalAlpha = 1;
  });
  const px = left + state.progress * (right - left);
  context.strokeStyle = "#c6d9e75a"; context.beginPath(); context.moveTo(px, top); context.lineTo(px, bottom); context.stroke();
  context.fillStyle = "#8198ab"; context.textAlign = "left"; context.fillText("0", left, rect.height - 6);
  context.textAlign = "right"; context.fillText("127", right, rect.height - 6);
  if (!values.length) { context.textAlign = "center"; context.fillStyle = "#6d879c"; context.fillText(state.result ? "无诊断样本" : "等待载入", (left + right) / 2, (top + bottom) / 2); }
}
function renderCharts() {
  renderChart("#contactChart", "contact_error_m", 1000, 1);
  renderChart("#normalChart", "normal_error_deg", 1, 1);
  renderChart("#curvatureChart", "curvature_inv_m", 1, 8);
  renderChart("#clearanceChart", "clearance_m", 1, 0);
}
function animationFrame(now) {
  const delta = state.previousFrame ? Math.min(now - state.previousFrame, 100) : 0;
  state.previousFrame = now;
  if (state.playing && variant()?.display_poses_wxyz?.length) {
    state.progress = (state.progress + delta / 10000) % 1; updateProgress();
  }
  if (state.dirty) { drawScene(); renderCharts(); state.dirty = false; }
  requestAnimationFrame(animationFrame);
}
function exportResult() {
  if (!state.result) return;
  const snapshot = { schema_version: "contact_visualization_snapshot_v001", exported_at: new Date().toISOString(),
    result: state.result, execution: executionMatchesPlan() ? state.execution : null,
    view: { candidate_index: state.candidateIndex, decoder: $("#decoder").value, progress_normalized_index: state.progress, camera: state.camera,
      overlay: $("#overlay").checked, show_body: $("#showBody").checked, show_tip: $("#showTip").checked, show_markers: $("#showMarkers").checked, show_actual: $("#showActual").checked } };
  const url = URL.createObjectURL(new Blob([JSON.stringify(snapshot, null, 2)], { type: "application/json" }));
  const link = document.createElement("a"); link.href = url;
  link.download = `contact_${state.result.model_id}_${String(state.result.condition_id).split("/").slice(-2).join("_")}_${state.result.scenario}.json`;
  document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$("#loadReplay").addEventListener("click", () => loadPlan("replay"));
$("#sampleModel").addEventListener("click", () => loadPlan("sample"));
$("#exportResult").addEventListener("click", exportResult);
["#family", "#kind", "#difficulty"].forEach((selector) => $(selector).addEventListener("change", () => { if (locked()) return; invalidateResult(); loadConditions(0); }));
$("#previousPage").addEventListener("click", () => { if (locked()) return; invalidateResult(); loadConditions(Math.max(0, state.conditionOffset - 100)); });
$("#nextPage").addEventListener("click", () => { if (locked()) return; invalidateResult(); loadConditions(state.conditionOffset + 100); });
$("#conditions").addEventListener("change", () => { if (locked()) return; $("#conditionId").value = $("#conditions").value; invalidateResult(); });
$("#conditionId").addEventListener("input", () => { if (locked()) return; invalidateResult(); });
["#scenario", "#sampleCount", "#samplingSteps"].forEach((selector) => $(selector).addEventListener("change", () => { if (!locked()) invalidateResult(); }));
$("#model").addEventListener("change", () => { if (locked()) return; updateCheckpoint(); invalidateResult(); });
$("#candidate").addEventListener("change", () => { if (locked()) return; state.candidateIndex = Number($("#candidate").value); changeVariant(); });
$("#decoder").addEventListener("change", changeVariant);
$("#allowCurvatureFailure").addEventListener("change", refreshControls);
["#overlay", "#showBody", "#showTip", "#showMarkers", "#showActual"].forEach((selector) => $(selector).addEventListener("change", () => { state.dirty = true; }));
$("#progress").addEventListener("input", () => { state.playing = false; $("#playGeometry").textContent = "▶"; state.progress = Number($("#progress").value) / 1000; updateProgress(); });
$("#playGeometry").addEventListener("click", () => { state.playing = !state.playing; $("#playGeometry").textContent = state.playing ? "Ⅱ" : "▶"; });
$$("[data-view]").forEach((node) => node.addEventListener("click", () => setView(node.dataset.view)));
$("#fitCamera").addEventListener("click", fitCamera);
$("#previewExecution").addEventListener("click", () => executionAction("preview"));
$("#startExecution").addEventListener("click", () => executionAction("start"));
$("#stopExecution").addEventListener("click", () => executionAction("stop"));
canvas.addEventListener("pointerdown", (event) => {
  if (event.button !== 0 && event.button !== 1 && event.button !== 2) return;
  canvas.setPointerCapture(event.pointerId); state.drag = { x: event.clientX, y: event.clientY, pan: event.shiftKey || event.button !== 0 };
});
canvas.addEventListener("pointermove", (event) => {
  if (!state.drag) return;
  const dx = event.clientX - state.drag.x, dy = event.clientY - state.drag.y;
  if (state.drag.pan) { state.camera.panX += dx; state.camera.panY += dy; }
  else { state.camera.yaw += dx * .008; state.camera.pitch = Math.max(-1.5, Math.min(1.56, state.camera.pitch - dy * .006)); state.camera.view = "iso"; $$("[data-view]").forEach((node) => node.classList.toggle("active", node.dataset.view === "iso")); }
  state.drag.x = event.clientX; state.drag.y = event.clientY; state.dirty = true;
});
["pointerup", "pointercancel", "lostpointercapture"].forEach((name) => canvas.addEventListener(name, () => { state.drag = null; }));
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("wheel", (event) => { event.preventDefault(); state.camera.scale = Math.max(8, Math.min(500, state.camera.scale * Math.exp(-event.deltaY * .001))); state.dirty = true; }, { passive: false });
new ResizeObserver(() => { state.dirty = true; }).observe($(".main-panel"));
window.addEventListener("beforeunload", () => { clearTimeout(state.executionTimer); });
function updateCheckpoint() {
  const model = state.catalog?.models?.find((item) => item.id === $("#model").value);
  $("#checkpoint").textContent = model?.checkpoint || "—";
  $("#checkpoint").title = model?.checkpoint || "";
}
async function initialize() {
  renderMetrics(); refreshControls(); requestAnimationFrame(animationFrame);
  try {
    const catalog = await request("/api/catalog"); state.catalog = catalog;
    if (finite(catalog.contact_execution_options?.speed_mps)) $("#executionSpeed").value = String(catalog.contact_execution_options.speed_mps);
    if (finite(catalog.contact_execution_options?.max_acceleration_mps2)) $("#executionAcceleration").value = String(catalog.contact_execution_options.max_acceleration_mps2);
    $("#model").replaceChildren(...array(catalog.models).map((model) => option(model.id, model.label || model.id)));
    $("#model").value = catalog.default_model || catalog.models?.[0]?.id || "";
    $("#family").append(...array(catalog.families).map((family) => option(family, family)));
    updateCheckpoint();
    const execution = catalog.execution || {};
    $("#executionAvailability").textContent = executionAvailable() ?
      (execution.viewer_enabled ? "物理执行可用 · 可打开 MuJoCo 本地窗口" : "物理执行可用 · 无窗口遥测模式") :
      execution.viewer_enabled && execution.viewer_available === false ? "当前无图形显示，请以 --headless 模式启动服务后执行。" :
        `执行环境不可用${execution.reason || execution.error ? `：${execution.reason || execution.error}` : ""}`;
    await loadConditions(0, catalog.default_condition_id);
    refreshControls(); pollExecutionSoon();
    if ($("#conditionId").value.trim()) await loadPlan("replay");
    else status("请选择测试条件。");
  } catch (error) { status(`初始化失败：${error.message}`, "error"); toast(error.message, true); refreshControls(); }
}
initialize();
