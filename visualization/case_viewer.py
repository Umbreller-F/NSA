"""在本地网站中浏览 results 目录下的评测 JSONL 和对应视频。"""

import argparse
import base64
import json
import mimetypes
import re
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
VIDEOS_ROOT = PROJECT_ROOT / "videos"
FINAL_ANSWER_PATTERN = re.compile(
    r"<final_answer>\s*(.*?)\s*</final_answer>", re.IGNORECASE | re.DOTALL
)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__PAGE_TITLE__</title>
  <style>
    :root {
      --bg: #0b0f14; --panel: #111821; --panel2: #17212c; --border: #293442;
      --text: #e6edf3; --muted: #8b9aaa; --blue: #58a6ff; --green: #42c978;
      --red: #ff6b6b; --orange: #f3a638; --purple: #bc8cff;
    }
    * { box-sizing: border-box; }
    body { margin: 0; height: 100vh; overflow: hidden; background: var(--bg); color: var(--text);
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    button, input, select { font: inherit; }
    .app { height: 100%; min-height: 0; overflow: hidden; display: grid; grid-template-columns: 330px 1fr; }
    aside { height: 100%; min-width: 0; min-height: 0; overflow: hidden; background: var(--panel); border-right: 1px solid var(--border);
      display: flex; flex-direction: column; }
    .side-head { flex: 0 0 auto; padding: 18px; border-bottom: 1px solid var(--border); }
    .side-head h1 { margin: 0 0 5px; font-size: 17px; }
    .file-name { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin: 14px 0 10px; }
    .stat { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 7px;
      text-align: center; font-size: 11px; color: var(--muted); }
    .stat b { display: block; color: var(--text); font-size: 15px; margin-bottom: 1px; }
    .search, .filter { width: 100%; color: var(--text); background: var(--bg); border: 1px solid var(--border);
      border-radius: 7px; outline: none; padding: 8px 9px; }
    .search:focus, .filter:focus { border-color: var(--blue); }
    .filters { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 7px; }
    .filter { font-size: 12px; }
    .list { flex: 1 1 auto; min-height: 0; overflow-x: hidden; overflow-y: auto; overscroll-behavior: contain; padding: 7px; }
    .item { border: 1px solid transparent; border-radius: 8px; padding: 10px; margin-bottom: 4px;
      cursor: pointer; transition: .12s; }
    .item:hover { background: var(--panel2); border-color: var(--border); }
    .item.active { background: rgba(88,166,255,.11); border-color: var(--blue); }
    .item-title { display: flex; gap: 6px; align-items: center;
      font-size: 12px; font-weight: 650; }
    .item-name { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .item-badges { flex: 0 0 auto; display: inline-flex; align-items: center; gap: 3px; white-space: nowrap; }
    .item-badges .pill { flex: 0 0 auto; white-space: nowrap; }
    .item-meta { margin-top: 5px; color: var(--muted); font-size: 11px; }
    .pill { display: inline-flex; align-items: center; white-space: nowrap; padding: 2px 7px;
      border-radius: 999px; font-size: 10px; font-weight: 700; border: 1px solid transparent; }
    .valid { color: var(--green); background: rgba(66,201,120,.12); border-color: rgba(66,201,120,.28); }
    .invalid { color: var(--red); background: rgba(255,107,107,.12); border-color: rgba(255,107,107,.28); }
    main { min-width: 0; min-height: 0; overflow: hidden; display: flex; flex-direction: column; }
    .toolbar { height: 58px; flex: 0 0 58px; padding: 0 20px; background: var(--panel);
      border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }
    .tools { display: flex; gap: 7px; }
    .btn { color: var(--text); background: var(--panel2); border: 1px solid var(--border);
      padding: 7px 12px; border-radius: 7px; cursor: pointer; }
    .btn:hover { border-color: var(--blue); }
    .btn:disabled { opacity: .35; cursor: default; border-color: var(--border); }
    .counter { color: var(--muted); font-size: 12px; }
    .detail { flex: 1; min-height: 0; overflow: auto; padding: 14px; }
    .detail-inner { max-width: 1500px; margin: auto; display: grid; grid-template-columns: minmax(420px, .9fr) minmax(480px, 1.1fr); gap: 14px; }
    .column { min-width: 0; display: flex; flex-direction: column; gap: 10px; }
    .card { min-width: 0; background: var(--panel); border: 1px solid var(--border); border-radius: 11px; overflow: hidden; }
    .card-head { padding: 10px 15px; color: var(--text); font-size: 14px; font-weight: 750;
      letter-spacing: .35px; border-bottom: 1px solid var(--border); }
    .card-body { padding: 15px; font-size: 13px; line-height: 1.7; white-space: pre-wrap; overflow-wrap: anywhere; }
    .video-card { background: #000; position: relative; min-height: 200px; display: flex; align-items: center; justify-content: center; }
    video { display: block; width: 100%; max-height: 43vh; background: #000; }
    .missing { color: var(--red); padding: 30px; text-align: center; }
    .tags { display: flex; flex-wrap: wrap; gap: 5px; }
    .tag { padding: 3px 7px; border-radius: 6px; background: var(--panel2); border: 1px solid var(--border);
      font-size: 10px; color: var(--muted); }
    .tag strong { color: var(--text); margin-left: 4px; }
    .compact-info .card-head { padding: 7px 10px; font-size: 14px; }
    .compact-info .card-body { display: grid; gap: 4px; padding: 7px 10px; font-size: 12px; line-height: 1.25; }
    .compact-info .tags { gap: 3px; }
    .compact-info .tag { padding: 2px 6px; font-size: 12px; line-height: 1.2; }
    .compact-info .pill { padding: 2px 6px; font-size: 11px; line-height: 1.2; }
    .compact-info .file-line { margin: 0; line-height: 1.25; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .compact-info .extra-line { margin: 0; line-height: 1.25; }
    .prompt-card summary { font-weight: 700; }
    .annotation-card { border-color: rgba(188,140,255,.45); }
    .annotation-controls { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; }
    .annotation-controls label { display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--muted); }
    .annotation-controls select { color: var(--text); background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 6px 9px; }
    .focus-label { cursor: pointer; color: var(--orange) !important; }
    .human-unjudged { color: var(--muted); background: rgba(139,154,170,.12); }
    .human-correct { color: var(--green); background: rgba(66,201,120,.12); }
    .human-incorrect { color: var(--red); background: rgba(255,107,107,.12); }
    .answers { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .answer { padding: 13px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; }
    .answer.gt { border-left: 3px solid var(--green); }
    .answer.pred { border-left: 3px solid var(--purple); }
    .answer-label { color: var(--muted); font-size: 10px; font-weight: 750; text-transform: uppercase; margin-bottom: 6px; }
    .answer-value { font-size: 14px; line-height: 1.55; white-space: pre-wrap; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .muted { color: var(--muted); }
    details summary { cursor: pointer; color: var(--blue); padding: 12px 15px; }
    details .raw { border-top: 1px solid var(--border); }
    .empty { height: 100%; display: grid; place-items: center; color: var(--muted); }
    ::-webkit-scrollbar { width: 7px; height: 7px; } ::-webkit-scrollbar-thumb { background: #344252; border-radius: 5px; }
    @media (max-width: 1050px) { .app { grid-template-columns: 280px 1fr; } .detail-inner { grid-template-columns: 1fr; } }
    @media (max-width: 700px) { body { overflow: hidden; } .app { height: 100vh; grid-template-columns: 1fr; grid-template-rows: 42vh 58vh; }
      aside { height: 42vh; border-right: 0; border-bottom: 1px solid var(--border); } main { min-height: 0; }
      .answers { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<div class="app">
  <aside>
    <div class="side-head">
      <h1>评测结果浏览器</h1>
      <div class="file-name">__RESULT_FILE__</div>
      <div class="stats">
        <div class="stat"><b id="stat-total">0</b>当前条目</div>
        <div class="stat"><b id="stat-valid">0</b>格式正确</div>
        <div class="stat"><b id="stat-missing">0</b>视频缺失</div>
      </div>
      <input id="search" class="search" placeholder="搜索视频、问题、GT 或回答…">
      <div class="filters">
        <select id="task-filter" class="filter"><option value="">全部任务</option><option value="rot">Rotation</option><option value="trans">Translation</option></select>
        <select id="category-filter" class="filter"><option value="">全部类型</option><option value="NSA">NSA</option><option value="NSA_ANCHOR">NSA Anchor</option><option value="SA">SA</option></select>
        <select id="format-filter" class="filter"><option value="">全部格式</option><option value="valid">格式正确</option><option value="invalid">格式错误</option></select>
        <select id="direction-filter" class="filter"><option value="">全部方向</option><option>up</option><option>down</option><option>left</option><option>right</option><option>clockwise</option><option>counterclockwise</option></select>
        <select id="answer-filter" class="filter"><option value="">全部人工判定</option><option value="unjudged">未判定</option><option value="correct">正确</option><option value="incorrect">错误</option></select>
        <select id="focus-filter" class="filter"><option value="">重点关注：全部</option><option value="yes">重点关注：是</option><option value="no">重点关注：否</option></select>
      </div>
    </div>
    <div id="list" class="list"></div>
  </aside>
  <main>
    <div class="toolbar">
      <div class="tools"><button id="prev" class="btn">← 上一条</button><button id="next" class="btn">下一条 →</button><button id="autoplay" class="btn">自动播放：关</button></div>
      <div id="counter" class="counter">0 / 0</div>
    </div>
    <div id="detail" class="detail"><div class="empty">正在加载…</div></div>
  </main>
</div>
<script>
const CASES = JSON.parse(new TextDecoder().decode(Uint8Array.from(atob("__CASES_BASE64__"), c => c.charCodeAt(0))));
let filtered = CASES.map((_, i) => i);
let current = filtered.length ? filtered[0] : -1;
let autoplay = false;
const $ = id => document.getElementById(id);
const esc = value => { const d=document.createElement('div'); d.textContent=value == null ? '' : String(value); return d.innerHTML; };
const display = value => value === null || value === undefined || value === '' ? '<span class="muted">无</span>' : esc(value);

function applyFilters() {
  const query = $('search').value.trim().toLowerCase();
  const task = $('task-filter').value, category = $('category-filter').value;
  const format = $('format-filter').value, direction = $('direction-filter').value;
  const answer = $('answer-filter').value, focus = $('focus-filter').value;
  filtered = CASES.map((c,i)=>[c,i]).filter(([c]) => {
    const haystack = [c.video_name,c.video_path,c.question,c.ground_truth,c.final_answer,c.model_answer,c.group].join(' ').toLowerCase();
    return (!query || haystack.includes(query)) && (!task || c.task===task) && (!category || c.category===category)
      && (!direction || c.direction===direction) && (!format || (format==='valid')===Boolean(c.answer_format_valid))
      && (!answer || (c.human_answer_status || 'unjudged')===answer)
      && (!focus || (focus==='yes')===Boolean(c.human_focus));
  }).map(([,i])=>i);
  if (!filtered.includes(current)) current = filtered.length ? filtered[0] : -1;
  renderList(); renderDetail(); updateStats();
}

function updateStats() {
  $('stat-total').textContent = filtered.length;
  $('stat-valid').textContent = filtered.filter(i=>CASES[i].answer_format_valid).length;
  $('stat-missing').textContent = filtered.filter(i=>!CASES[i]._video_available).length;
}

function renderList() {
  const list=$('list'); list.innerHTML='';
  filtered.forEach(i => {
    const c=CASES[i], el=document.createElement('div');
    el.className='item'+(i===current?' active':''); el.onclick=()=>selectCase(i);
    const valid=c.answer_format_valid;
    const answerStatus = c.human_answer_status || 'unjudged';
    const statusText = {unjudged:'未判定', correct:'回答正确', incorrect:'回答错误'}[answerStatus] || '未判定';
    const focusText = c.human_focus ? '★重点' : '';
    el.innerHTML=`<div class="item-title"><span class="item-name" title="${esc(c.video_name || c.video_path || '未命名')}">${esc(c.video_name || c.video_path || '未命名')}</span><span class="item-badges"><span class="pill ${valid?'valid':'invalid'}">${valid?'格式 OK':'格式异常'}</span><span class="pill human-${esc(answerStatus)}">${statusText}</span></span></div>
      <div class="item-meta">${esc([c.task,c.category,c.source,c.direction].filter(Boolean).join(' · '))}</div>`;
    if (focusText) el.querySelector('.item-meta').textContent += '　' + focusText;
    list.appendChild(el);
  });
}

function renderDetail() {
  if (current < 0) { $('detail').innerHTML='<div class="empty">没有符合筛选条件的记录</div>'; updateNav(); return; }
  const c=CASES[current];
  const video = c._video_available
    ? `<video id="player" controls loop ${autoplay?'autoplay':''}><source src="${esc(c._media_url)}" type="video/mp4">浏览器无法播放该视频。</video>`
    : `<div class="missing">视频文件不存在<br><span class="mono">${esc(c.video_path || '')}</span></div>`;
  const valid=Boolean(c.answer_format_valid);
  const answerStatus = c.human_answer_status || 'unjudged';
  $('detail').innerHTML=`<div class="detail-inner">
    <div class="column">
      <div class="card video-card">${video}</div>
      <div class="card"><div class="card-head">答案对照</div><div class="card-body"><div class="answers">
        <div class="answer gt"><div class="answer-label">Ground Truth</div><div class="answer-value">${display(c.ground_truth)}</div></div>
        <div class="answer pred"><div class="answer-label">模型最终答案</div><div class="answer-value">${display(c.final_answer)}</div></div>
      </div></div></div>
      <div class="card annotation-card"><div class="card-head">人工标记（修改后自动保存）</div><div class="card-body"><div class="annotation-controls">
        <label>回答判定<select id="human-answer-status"><option value="unjudged" ${answerStatus==='unjudged'?'selected':''}>未判定</option><option value="correct" ${answerStatus==='correct'?'selected':''}>正确</option><option value="incorrect" ${answerStatus==='incorrect'?'selected':''}>错误</option></select></label>
        <label class="focus-label"><input id="human-focus" type="checkbox" ${c.human_focus?'checked':''}> 重点关注</label>
        <span id="save-status" class="muted">标记保存在 label JSONL</span>
      </div></div></div>
      <div class="card compact-info"><div class="card-head">类型与文件信息</div><div class="card-body">
        <div class="tags">
          <span class="tag">任务<strong>${display(c.task)}</strong></span><span class="tag">类别<strong>${display(c.category)}</strong></span>
          <span class="tag">源图<strong>${display(c.source)}</strong></span><span class="tag">方向<strong>${display(c.direction)}</strong></span>
          <span class="tag">分组<strong>${display(c.group)}</strong></span><span class="pill ${valid?'valid':'invalid'}">${valid?'回答格式正确':'回答格式无法正确提取'}</span>
        </div>
        <div class="mono muted file-line" title="${esc(c.video_path || '')}">${display(c.video_path)}</div>
        <div class="muted extra-line">时间：${display(c.timestamp)}　Finish reason：${display(c.finish_reason)}　行号：${display(c._line_number)}</div>
      </div></div>
    </div>
    <div class="column">
      <div class="card prompt-card"><details><summary>问题 / Prompt（点击展开）</summary><div class="card-body raw">${display(c.question)}</div></details></div>
      <div class="card"><div class="card-head">模型推理 / Reasoning</div><div class="card-body">${display(c.reasoning)}</div></div>
      <div class="card"><details><summary>查看完整原始模型回答</summary><div class="card-body raw">${display(c.model_answer)}</div></details></div>
      <div class="card"><details><summary>查看该条完整 JSON</summary><div class="card-body raw mono">${esc(JSON.stringify(c._original, null, 2))}</div></details></div>
    </div>
  </div>`;
  $('human-answer-status').addEventListener('change', saveLabels);
  $('human-focus').addEventListener('change', saveLabels);
  updateNav();
}

function updateNav() {
  const pos=filtered.indexOf(current); $('counter').textContent=pos<0?'0 / 0':`${pos+1} / ${filtered.length}`;
  $('prev').disabled=pos<=0; $('next').disabled=pos<0 || pos>=filtered.length-1;
}
function selectCase(i) { current=i; renderList(); renderDetail(); document.querySelector('.item.active')?.scrollIntoView({block:'nearest'}); }
function move(delta) { const p=filtered.indexOf(current), np=p+delta; if(np>=0 && np<filtered.length) selectCase(filtered[np]); }
$('prev').onclick=()=>move(-1); $('next').onclick=()=>move(1);
$('autoplay').onclick=()=>{ autoplay=!autoplay; $('autoplay').textContent=`自动播放：${autoplay?'开':'关'}`; const v=$('player'); if(v){v.autoplay=autoplay; if(autoplay)v.play().catch(()=>{});} };
['search','task-filter','category-filter','format-filter','direction-filter','answer-filter','focus-filter'].forEach(id => $(id).addEventListener(id==='search'?'input':'change',applyFilters));

async function saveLabels() {
  const c = CASES[current]; if (!c) return;
  c.human_answer_status = $('human-answer-status').value;
  c.human_focus = $('human-focus').checked;
  $('save-status').textContent = '保存中…';
  try {
    const response = await fetch('/labels', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({index:current, human_answer_status:c.human_answer_status, human_focus:c.human_focus})});
    if (!response.ok) throw new Error(await response.text());
    $('save-status').textContent = '已保存';
    if ($('answer-filter').value || $('focus-filter').value) applyFilters();
    else { renderList(); updateStats(); }
  } catch (error) { $('save-status').textContent = '保存失败：' + error; }
}
document.addEventListener('keydown', e=>{ if(['INPUT','SELECT'].includes(e.target.tagName))return; if(e.key==='ArrowLeft')move(-1); else if(e.key==='ArrowRight')move(1); else if(e.key===' '){e.preventDefault();const v=$('player');if(v)(v.paused?v.play():v.pause());} });
applyFilters();
</script>
</body></html>"""


def resolve_jsonl_path(value: str) -> Path:
    """解析用户指定的 JSONL；只给文件名时默认从 results 目录读取。"""
    raw = Path(value).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend((Path.cwd() / raw, RESULTS_ROOT / raw))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "、".join(str(path) for path in candidates)
    raise FileNotFoundError(f"找不到 JSONL 文件，已检查：{searched}")


def label_path_for(jsonl_path: Path) -> Path:
    """返回标注版路径，例如 result.jsonl -> result_label.jsonl。"""
    suffix = jsonl_path.suffix or ".jsonl"
    if jsonl_path.stem.endswith("_label"):
        return jsonl_path
    return jsonl_path.with_name(jsonl_path.stem + "_label" + suffix)


def parse_answer_fallback(record: dict) -> tuple[str, str | None, bool]:
    """兼容尚未包含拆分字段的旧结果文件。"""
    model_answer = str(record.get("model_answer") or "")
    matches = list(FINAL_ANSWER_PATTERN.finditer(model_answer))
    if len(matches) != 1:
        return model_answer.strip(), None, False
    match = matches[0]
    reasoning = model_answer[: match.start()].strip()
    answer = match.group(1).strip()
    valid = bool(answer) and not model_answer[match.end() :].strip()
    return reasoning, answer, valid


def resolve_video_path(record: dict) -> Path | None:
    """兼容结果中的绝对路径、项目相对路径以及仅视频文件名。"""
    raw_path = record.get("video_path")
    candidates = []
    if raw_path:
        path = Path(str(raw_path)).expanduser()
        candidates.append(path if path.is_absolute() else PROJECT_ROOT / path)
    video_name = record.get("video_name")
    if video_name:
        candidates.append(VIDEOS_ROOT / Path(str(video_name)).name)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    return None


def read_jsonl(jsonl_path: Path) -> tuple[list[dict], dict[int, Path]]:
    """读取 JSONL、补齐旧字段，并建立安全的媒体 ID 到视频路径映射。"""
    cases = []
    media_files = {}
    with jsonl_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path} 第 {line_number} 行不是有效 JSON：{exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{jsonl_path} 第 {line_number} 行不是 JSON 对象")

            reasoning, final_answer, inferred_valid = parse_answer_fallback(record)
            case = dict(record)
            case["reasoning"] = record.get("reasoning", reasoning)
            case["final_answer"] = record.get("final_answer", final_answer)
            case["answer_format_valid"] = record.get("answer_format_valid", inferred_valid)
            case["human_answer_status"] = record.get("human_answer_status", "unjudged")
            case["human_focus"] = bool(record.get("human_focus", False))
            case["video_name"] = record.get("video_name") or Path(
                str(record.get("video_path") or "未命名视频")
            ).name
            case["_line_number"] = line_number
            case["_original"] = record

            video_path = resolve_video_path(record)
            media_id = len(cases)
            case["_video_available"] = video_path is not None
            case["_media_url"] = f"/media/{media_id}" if video_path else None
            if video_path:
                media_files[media_id] = video_path
            cases.append(case)
    return cases, media_files


def write_label_jsonl(label_path: Path, cases: list[dict]) -> None:
    """保存结果和人工标记，排除查看器运行时的内部字段。"""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = label_path.with_name(label_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for case in cases:
            record = dict(case.get("_original", case))
            record["human_answer_status"] = case.get("human_answer_status", "unjudged")
            record["human_focus"] = bool(case.get("human_focus", False))
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(label_path)


class ViewerHandler(SimpleHTTPRequestHandler):
    """提供内嵌页面和支持 Range 请求的视频内容。"""

    def do_GET(self):
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            content = self.server.html_content.encode("utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        if path.startswith("/media/"):
            self.serve_media(path)
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/labels":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            index = int(payload["index"])
            status = payload.get("human_answer_status", "unjudged")
            if status not in {"unjudged", "correct", "incorrect"}:
                raise ValueError("invalid human_answer_status")
            with self.server.label_lock:
                case = self.server.cases[index]
                case["human_answer_status"] = status
                case["human_focus"] = bool(payload.get("human_focus", False))
                write_label_jsonl(self.server.label_path, self.server.cases)
            body = json.dumps({"ok": True}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            body = str(exc).encode("utf-8", errors="replace")
            self.send_response(400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def serve_media(self, request_path: str):
        try:
            media_id = int(unquote(request_path).removeprefix("/media/"))
        except ValueError:
            self.send_error(404, "Invalid media id")
            return
        video_path = self.server.media_files.get(media_id)
        if video_path is None or not video_path.is_file():
            self.send_error(404, "Video not found")
            return

        file_size = video_path.stat().st_size
        start, end = 0, file_size - 1
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_error(416, "Invalid Range")
                return
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), file_size - 1) if last else file_size - 1
            elif last:
                length = min(int(last), file_size)
                start, end = file_size - length, file_size - 1
            if start > end or start >= file_size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return

        length = end - start + 1
        self.send_response(206 if range_header else 200)
        self.send_header("Content-Type", mimetypes.guess_type(video_path.name)[0] or "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        try:
            with video_path.open("rb") as file:
                file.seek(start)
                remaining = length
                while remaining:
                    chunk = file.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format_string, *args):
        if args and str(args[0]).startswith("GET /media/"):
            return
        super().log_message(format_string, *args)


def build_html(cases: list[dict], jsonl_path: Path) -> str:
    cases_bytes = json.dumps(cases, ensure_ascii=False).encode("utf-8")
    cases_base64 = base64.b64encode(cases_bytes).decode("ascii")
    title = f"评测结果 - {jsonl_path.name}"
    html = (
        HTML_TEMPLATE.replace("__CASES_BASE64__", cases_base64)
        .replace("__RESULT_FILE__", jsonl_path.name)
        .replace("__PAGE_TITLE__", title)
    )
    # HTML_TEMPLATE 是 raw string，模板中的 \" 必须还原为真正的 HTML 引号。
    # 否则浏览器无法正确识别 id/class，前端会停在“正在加载”。
    return html.replace(r'\"', '"')


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 JSONL 可视化浏览器")
    parser.add_argument(
        "--jsonl",
        required=True,
        help="结果 JSONL 路径；只写文件名时从 results 目录查找",
    )
    parser.add_argument("--port", type=int, default=8080, help="服务端口（默认：8080）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认：127.0.0.1）")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    source_jsonl_path = resolve_jsonl_path(args.jsonl)
    label_path = label_path_for(source_jsonl_path)
    jsonl_path = label_path if label_path.is_file() else source_jsonl_path
    cases, media_files = read_jsonl(jsonl_path)
    if not cases:
        raise RuntimeError(f"JSONL 中没有有效记录：{jsonl_path}")
    if not label_path.is_file():
        write_label_jsonl(label_path, cases)
        jsonl_path = label_path
    html = build_html(cases, jsonl_path)

    class ViewerServer(ThreadingHTTPServer):
        pass

    server = ViewerServer((args.host, args.port), ViewerHandler)
    server.html_content = html
    server.media_files = media_files
    server.cases = cases
    server.label_path = label_path
    server.label_lock = threading.Lock()
    url_host = "localhost" if args.host in {"0.0.0.0", "127.0.0.1"} else args.host
    url = f"http://{url_host}:{args.port}"

    print(f"已读取：{jsonl_path}")
    print(f"记录数：{len(cases)}，可用视频：{len(media_files)}，缺失视频：{len(cases)-len(media_files)}")
    print(f"浏览地址：{url}")
    print("按 Ctrl+C 停止服务")
    if not args.no_browser:
        threading.Thread(
            target=lambda: (time.sleep(0.5), webbrowser.open(url)), daemon=True
        ).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭服务…")
    finally:
        server.server_close()

"""
python visualization/case_viewer.py \
  --jsonl gemini-3.5-flash-v2-test.jsonl \
  --port 34800
"""
if __name__ == "__main__":
    main()
