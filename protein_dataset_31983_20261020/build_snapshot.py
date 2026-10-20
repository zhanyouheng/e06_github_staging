"""Copy an accession-scoped review snapshot without changing review decisions."""
import csv
import gzip
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
STEP3 = OUT.parent
PROJECT = STEP3.parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def load(path):
    return json.loads(path.read_text())


def fasta(path):
    result = {}
    ac, parts = None, []
    for line in path.open():
        if line.startswith(">"):
            if ac is not None:
                assert ac not in result
                result[ac] = "".join(parts)
            ac, parts = line[1:].strip().split()[0], []
        else:
            parts.append(line.strip())
    if ac is not None:
        assert ac not in result
        result[ac] = "".join(parts)
    return result


def table(path):
    with path.open() as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def archive(path, producer):
    """Atomic, independently resumable compressed archive."""
    done = path.with_suffix(path.suffix + ".done.json")
    if path.exists() and done.exists():
        meta = load(done)
        if sha(path.read_bytes()) == meta["sha256"]:
            return meta
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=1) as handle:
        for item in producer():
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    temp.replace(path)
    meta = {"records": count, "sha256": sha(path.read_bytes())}
    save(done, meta)
    return meta


def archived(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def snapshot_json(source, destination):
    if not destination.exists():
        raw = source.read_bytes()
        json.loads(raw)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(destination.suffix + ".tmp")
        temp.write_bytes(raw)
        temp.replace(destination)
        save(destination.with_suffix(destination.suffix + ".source.json"), {
            "source": str(source), "sha256": sha(raw),
        })
    return load(destination)


def raw_document(path, expected_hash=None):
    raw = path.read_bytes()
    if expected_hash is not None:
        assert sha(raw) == expected_hash, str(path) + ": dependency changed"
    return {"source_path": str(path), "source_sha256": sha(raw),
            "document": json.loads(raw)}


def main():
    bg_root = STEP3 / "full_review_20261011"
    pos_root = STEP3 / "positive_review_20261018"
    bg_sel = bg_root / "selective_review_20261011"
    pos_sel = pos_root / "selective_review_20261018"
    exclusion = load(bg_sel / "explicit_exclusions_20261019.json")
    excluded = set(exclusion["accessions"])
    pos = fasta(STEP3 / "background/positive_2019.fasta")
    bg = fasta(STEP3 / "background/background_2019.fasta")
    bg = {ac: seq for ac, seq in bg.items() if ac not in excluded}
    added = fasta(STEP3 / "gpu_handoff_pack_20261008/v2026_positive702.fasta")
    cohorts = {"background_reviewed": bg, "positive_2019": pos, "later702": added}
    roster = {}
    for name, members in cohorts.items():
        assert not set(roster) & set(members)
        roster.update({ac: (name, seq) for ac, seq in members.items()})
    assert len(roster) == 31983
    scope = set(roster)
    assert not scope & excluded
    classes = load(bg_root / "manifest.json")["class_ids"]
    assert len(classes) == 32
    start = OUT / "snapshot_start.json"
    if not start.exists():
        save(start, {"started_utc": datetime.now(timezone.utc).isoformat(),
                     "accessions": len(roster), "cohorts": {k: len(v) for k, v in cohorts.items()}})

    # Metadata and evidence are copied, not symlinked.
    for name, root, sel in [("background", bg_root, bg_sel), ("positive", pos_root, pos_sel)]:
        for filename in ["freeze.json", "manifest.json"]:
            snapshot_json(root / filename, OUT / "metadata" / name / filename)
        for filename in ["policy.json", "run_progress.json"]:
            snapshot_json(sel / filename, OUT / "metadata" / name / filename)
        for filename in ["class_definitions.json", "label_policy.json", "protocol.json", "schema.json"]:
            snapshot_json(root / "inputs" / filename, OUT / "metadata" / name / filename)
        members = set(bg) if name == "background" else set(pos)
        archive(OUT / "evidence" / (name + ".jsonl.gz"),
                lambda r=root, m=members: (x for x in load(r / "inputs/cases.json") if x["ac"] in m))
    save(OUT / "metadata/explicit_exclusions.json", exclusion)
    snapshot_json(bg_sel / "acceptance/summary.json", OUT / "metadata/background/technical_acceptance.json")
    snapshot_json(STEP3 / "label_policy.json", OUT / "metadata/current_label_policy.json")

    # Snapshot the list of resolved files before copying the active run.
    current = {}
    review_counts = {}
    for name, root, sel, wanted in [
        ("background", bg_root, bg_sel, set(bg)),
        ("positive", pos_root, pos_sel, set(pos)),
    ]:
        listing_path = OUT / "metadata" / name / "resolved_files.json"
        if not listing_path.exists():
            save(listing_path, sorted(p.name for p in (sel / "resolved").glob("*.json")))
        listing = load(listing_path)
        for offset in range(0, len(listing), 100):
            chunk = listing[offset:offset + 100]
            dest = OUT / "reviews" / name / ("resolved_%04d.jsonl.gz" % (offset // 100))

            def documents():
                for filename in chunk:
                    source = sel / "resolved" / filename
                    entry = raw_document(source)
                    d = entry["document"]
                    assert {r["ac"] for r in d["verdicts"]} <= wanted
                    initial = {}
                    for tag, hashes in d["initial_dependencies"].items():
                        folder = root / "runs" / tag / d["batch_id"]
                        initial[tag] = {
                            key: raw_document(folder / (key + ".json"), hashes[key + "_sha256"])
                            for key in ["response", "status"]
                        }
                    final = None
                    if d["final_source"]:
                        info = d["final_source"]
                        final = raw_document(Path(info["directory"]) / "response.json", info["response_sha256"])
                    yield {"resolved": entry, "initial": initial, "final": final}

            archive(dest, documents)
            for entry in archived(dest):
                d = entry["resolved"]["document"]
                for row in d["verdicts"]:
                    key = (row["ac"], row["class_id"])
                    assert key not in current
                    assert row["verdict"] in {"P", "N", "U"}
                    current[key] = (row["verdict"], row["review_status"], d["batch_id"])
            save(OUT / "build_progress.json", {"stage": name, "copied_resolved_batches": offset + len(chunk)})
            print(name, "resolved batches", offset + len(chunk), "/", len(listing), flush=True)
        review_counts[name] = {"resolved_batches": len(listing),
                              "resolved_proteins": len({ac for ac, cls in current if ac in wanted})}
    assert sum(ac in bg for ac, cls in current) == len(bg) * 32

    # Include completed first-review responses from the still-running positive run.
    for tag in ["sol61_medium", "luna_max"]:
        listing_path = OUT / "metadata/positive" / (tag + "_response_paths.json")
        if not listing_path.exists():
            folders = pos_root / "runs" / tag
            paths = []
            if folders.exists():
                for entry in os.scandir(folders):
                    if entry.is_dir() and (Path(entry.path) / "response.json").exists():
                        paths.append(str(Path(entry.path) / "response.json"))
            save(listing_path, sorted(paths))

        def initial_documents():
            for source in load(listing_path):
                path = Path(source)
                status_path = path.with_name("status.json")
                if not status_path.exists():
                    continue
                status = raw_document(status_path)
                d = status["document"]
                if d.get("status") != "complete" or d.get("validation", {}).get("errors"):
                    continue
                response = raw_document(path, d["response_sha256"])
                assert {r["ac"] for r in response["document"]["verdicts"]} <= set(pos)
                yield {"response": response, "status": status}

        archive(OUT / "reviews/positive" / (tag + "_completed_initials.jsonl.gz"), initial_documents)

    # Historical labels remain separate from current-protocol decisions.
    historical = {}
    sources = [
        (STEP3 / "v4/positive_labels_full_v2026_dedup.tsv", set(pos) | set(added)),
        (STEP3 / "background/background_labels_final.tsv", set(bg)),
    ]
    for source, wanted in sources:
        dest = OUT / "reviews/historical" / (source.stem + ".jsonl.gz")
        archive(dest, lambda s=source, w=wanted: (r for r in table(s) if r["ac"] in w))
        for row in archived(dest):
            key = row["ac"], row["class"]
            assert key not in historical
            historical[key] = (
                {"pos": "P", "neg": "N", "unknown": "U"}[row["label"]],
                row.get("source", "historical_background"), str(source))
    assert len(historical) == len(roster) * 32

    # Copy historical per-model decisions and reasons, limited to this roster.
    lc = STEP3 / "label_completion"
    rv = STEP3 / "v4/review_2026"
    old_raw = sorted((PROJECT / "step2/curation").glob("*.verdicts.jsonl"))
    for pattern in ["fullcheck_input.*.verdicts.jsonl", "confirm.astra_high.*.verdicts.jsonl",
                    "contra_review.*.verdicts.jsonl", "fullcheck_phase2_2026_input.*.verdicts.jsonl"]:
        old_raw.extend(sorted(lc.glob(pattern)))
    for pattern in ["phase1_pairs.*.verdicts.jsonl", "astra_adjud.*.verdicts.jsonl",
                    "astra_tiebreak.*.verdicts.jsonl", "astra_fnf.*.verdicts.jsonl",
                    "sameseq_adjud.*.verdicts.jsonl"]:
        old_raw.extend(sorted(rv.glob(pattern)))
    for index, source in enumerate(old_raw):
        dest = OUT / "reviews/historical/raw" / ("%03d_%s.gz" % (index, source.name))

        def raw_rows(s=source):
            for number, line in enumerate(s.open(), 1):
                row = json.loads(line)
                if row.get("ac") in scope:
                    yield {"source_path": str(s), "source_line": number, "record": row}

        archive(dest, raw_rows)
    for source in [PROJECT / "step2/curation/final_labels_v3.tsv",
                   STEP3 / "v4/final_labels_v2026_dedup.tsv",
                   lc / "manual_review/user_verdicts.tsv",
                   rv / "positive_pairs_2026.tsv"]:
        archive(OUT / "reviews/historical" / (source.stem + ".jsonl.gz"),
                lambda s=source: (r for r in table(s) if r.get("ac") in scope))
    for name, source, key, wanted in [
        ("historical_positive", PROJECT / "step2/cards/cards.tsv", "Entry", set(pos)),
        ("later702", rv.parent / "api_cards/api_cards_2026.tsv", "Entry", set(added)),
    ]:
        archive(OUT / "evidence" / (name + ".jsonl.gz"),
                lambda s=source, k=key, w=wanted: (r for r in table(s) if r[k] in w))

    with (OUT / "proteins.fasta.tmp").open("w") as fa, (OUT / "proteins.tsv.tmp").open("w") as tab:
        writer = csv.writer(tab, delimiter="\t")
        writer.writerow(["ac", "cohort", "sequence_release", "length", "sequence_sha256"])
        for ac, (cohort, seq) in sorted(roster.items()):
            fa.write(">%s\n%s\n" % (ac, seq))
            writer.writerow([ac, cohort, "UniProtKB_2026_03" if cohort == "later702" else "UniProtKB_2019_05",
                             len(seq), sha(seq.encode())])
    (OUT / "proteins.fasta.tmp").replace(OUT / "proteins.fasta")
    (OUT / "proteins.tsv.tmp").replace(OUT / "proteins.tsv")

    fields = ["ac", "class_id", "cohort", "historical_label", "historical_source",
              "current_review_label", "current_review_status", "current_review_batch",
              "current_review_pending"]
    with (OUT / "labels_32class.tsv.tmp").open("w") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(fields)
        for ac, (cohort, seq) in sorted(roster.items()):
            for cls in classes:
                old_label, old_source, original = historical[ac, cls]
                fresh = current.get((ac, cls), ("", "", ""))
                pending = "0" if fresh[0] else ("not_scheduled" if cohort == "later702" else "1")
                writer.writerow([ac, cls, cohort, old_label, old_source, *fresh, pending])
    (OUT / "labels_32class.tsv.tmp").replace(OUT / "labels_32class.tsv")
    summary = {
        "snapshot_started_utc": load(start)["started_utc"],
        "snapshot_finished_utc": datetime.now(timezone.utc).isoformat(),
        "type": "review_result_snapshot_not_final_training_labels",
        "proteins": len(roster), "classes": len(classes), "label_rows": len(historical),
        "cohorts": {k: len(v) for k, v in cohorts.items()},
        "distinct_sequences": len({seq for _, seq in roster.values()}),
        "current_protocol_resolved_cells": len(current),
        "current_protocol_resolved_proteins": len({ac for ac, cls in current}),
        "current_review_counts": review_counts,
        "excluded_accessions": sorted(excluded),
        "historical_records_retained": True,
        "unresolved_cells_filled_with_N_or_U": False,
        "ready_as_final_training_release": False,
        "original_files_modified": False,
    }
    save(OUT / "summary.json", summary)
    (OUT / "README.txt").write_text(
        "蛋白审核结果合并快照：31,983 个编号\n"
        "范围：背景重审 17,497 + positive_2019 13,784 + 后来确认的 702；不含另外 720 和明确排除的 3 个。\n"
        "proteins.fasta：全部序列；proteins.tsv：编号、来源批次、序列版本及 SHA-256。\n"
        "labels_32class.tsv：每个编号 32 行。historical_label 是历史标签，current_review_label 是本轮已完成结果；两者分列，未生成统一定稿标签。\n"
        "current_review_pending=1 表示本轮未完成；not_scheduled 表示该批未纳入本轮。空白不等于 U，也不等于 N。\n"
        "正例的新审核仍在进行。这里只复制快照，不自动覆盖原有正标签；下游仍需按用户决定合并非正类格并保留冲突记录。\n"
        "702 的历史标签使用旧规则，当前规则适用性尚未统一确认。历史 P/N/U 不能自动当成当前规则下的金标准。\n"
        "reviews/：按来源复制的判决、理由及可用状态记录，gzip JSONL；background/positive 的 resolved 分片同时包含相关初审及终审输出。\n"
        "evidence/：对应注释材料。metadata/：协议、范围、进度和验收快照。未复制账号凭据、运行缓存或网络配置。\n"
        "所有内容均为实际复制，不是软链接；原审核目录不改动。活跃审核按文件读取时保存，非全局同时刻快照，不会自动同步后续结果。\n"
        "重复序列保留不同编号，用 sequence_sha256 关联；不要把相同序列拆到训练集和测试集两边。\n"
        "summary.json/validation.json 为统计与结构验收；这不是完整新版训练标签发布。\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
