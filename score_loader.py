import json
from collections import defaultdict


ENCODINGS = ("utf-8", "utf-8-sig", "gbk", "utf-16", "utf-16-le", "utf-16-be")


class ScoreValidationError(Exception):
    pass


def read_json_with_fallback(path):
    last_err = None
    for encoding in ENCODINGS:
        try:
            with open(path, "r", encoding=encoding) as f:
                return json.load(f)
        except Exception as exc:
            last_err = exc
    raise ScoreValidationError(f"乐谱文件解析失败: {last_err}")


def extract_score_meta(data):
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data:
        for item in data:
            if isinstance(item, dict) and "songNotes" in item:
                return item
        if isinstance(data[0], dict):
            return data[0]
    raise ScoreValidationError("乐谱文件格式不正确：顶层应为对象或非空数组。")


def load_score(path, valid_keys=None):
    data = read_json_with_fallback(path)
    meta = extract_score_meta(data)
    song_notes = meta.get("songNotes")
    if not isinstance(song_notes, list):
        raise ScoreValidationError("乐谱文件格式不正确：未找到 songNotes 数组。")

    valid_keys = set(valid_keys or [])
    errors = []
    warnings = []
    notes_by_time = defaultdict(list)
    seen = set()

    for i, note in enumerate(song_notes):
        if not isinstance(note, dict):
            errors.append(f"第 {i + 1} 条音符不是对象。")
            continue
        if "time" not in note:
            errors.append(f"第 {i + 1} 条音符缺少 time。")
            continue
        if "key" not in note:
            errors.append(f"第 {i + 1} 条音符缺少 key。")
            continue

        try:
            t_ms = int(round(float(note["time"])))
        except (TypeError, ValueError):
            errors.append(f"第 {i + 1} 条音符 time 不是数字: {note.get('time')!r}")
            continue
        if t_ms < 0:
            errors.append(f"第 {i + 1} 条音符 time 不能为负数: {t_ms}")
            continue

        key = str(note["key"])
        if valid_keys and key not in valid_keys:
            errors.append(f"第 {i + 1} 条音符 key 未在按键映射中定义: {key}")
            continue

        dedupe_key = (t_ms, key)
        if dedupe_key in seen:
            warnings.append(f"重复音符已忽略: time={t_ms}, key={key}")
            continue
        seen.add(dedupe_key)
        notes_by_time[t_ms].append(key)

    if errors:
        preview = "\n".join(errors[:8])
        more = "" if len(errors) <= 8 else f"\n... 还有 {len(errors) - 8} 个错误"
        raise ScoreValidationError(preview + more)

    sorted_times = sorted(notes_by_time.keys())
    if not sorted_times:
        raise ScoreValidationError("乐谱没有可播放的有效音符。")

    return {
        "raw": data,
        "meta": meta,
        "notes_by_time": notes_by_time,
        "sorted_times": sorted_times,
        "warnings": warnings,
    }


def summarize_meta(data):
    try:
        meta = extract_score_meta(data)
    except ScoreValidationError:
        return {}
    return {
        "name": meta.get("songName") or meta.get("name") or "",
        "author": meta.get("author") or "",
        "transcribedBy": meta.get("transcribedBy") or meta.get("transcriber") or "",
        "bpm": meta.get("bpm", 120),
    }
