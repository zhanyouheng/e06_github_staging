#!/usr/bin/env python3
"""Build the 31,983-accession training table without modifying review sources."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
STEP3 = OUT.parent
VA = "virulence_activity"
RESTORE_REASON = "user_restore_rule_a_va_as_P_20261029"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def save(path, obj):
    save_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def fasta(path):
    result = {}
    accession = None
    for line in path.open():
        if line.startswith(">"):
            accession = line[1:].strip().split()[0]
            assert accession not in result
            result[accession] = ""
        else:
            result[accession] += line.strip()
    return result


def table(path):
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_table(name, fields, rows):
    temporary = OUT / (name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUT / name)


def load_current(root, wanted):
    result = {}
    for path in sorted((root / "resolved").glob("*.json")):
        document = json.loads(path.read_text())
        for row in document["verdicts"]:
            key = row["ac"], row["class_id"]
            assert row["ac"] in wanted and key not in result
            assert row["verdict"] in {"P", "N", "U"}
            result[key] = (row["verdict"], row["review_status"], document["batch_id"])
    assert len(result) == len(wanted) * 32
    assert {ac for ac, _ in result} == wanted
    return result


def current_source(status):
    return {
        "provisional_consensus": "current_v32_provisional_consensus",
        "astra_adjudicated": "current_v32_astra_adjudicated",
    }[status]


def main():
    exclusions = json.loads((STEP3 / "full_review_20261011/selective_review_20261011/explicit_exclusions_20261019.json").read_text())
    excluded = set(exclusions["accessions"])
    cohorts = {
        "positive_2019": fasta(STEP3 / "background/positive_2019.fasta"),
        "background_reviewed": {ac: seq for ac, seq in fasta(STEP3 / "background/background_2019.fasta").items() if ac not in excluded},
        "later702": fasta(STEP3 / "gpu_handoff_pack_20261008/v2026_positive702.fasta"),
    }
    assert {k: len(v) for k, v in cohorts.items()} == {"positive_2019": 13784, "background_reviewed": 17497, "later702": 702}
    roster = {}
    for cohort, members in cohorts.items():
        assert not set(roster) & set(members)
        roster.update({ac: (cohort, seq) for ac, seq in members.items()})
    assert len(roster) == 31983
    bg_root = STEP3 / "full_review_20261011"
    pos_root = STEP3 / "positive_review_20261018"
    classes = json.loads((bg_root / "manifest.json").read_text())["class_ids"]
    assert len(classes) == len(set(classes)) == 32 and VA in classes
    assert classes == json.loads((pos_root / "manifest.json").read_text())["class_ids"]
    bg_acceptance = json.loads((bg_root / "selective_review_20261011/acceptance/summary.json").read_text())
    pos_acceptance = json.loads((pos_root / "selective_review_20261018/acceptance/summary.json").read_text())
    assert bg_acceptance["existing_results_valid"] is True
    assert bg_acceptance["verified_proteins"] == 17497 and bg_acceptance["pending_batches"] == ["batch05367"]
    assert not bg_acceptance["failed_batches"] and not bg_acceptance["unexpected_files"]
    assert pos_acceptance["whole_review_complete"] is True and pos_acceptance["verified_proteins"] == 13784
    current = {
        "background_reviewed": load_current(bg_root / "selective_review_20261011", set(cohorts["background_reviewed"])),
        "positive_2019": load_current(pos_root / "selective_review_20261018", set(cohorts["positive_2019"])),
    }
    historical = {}
    for source, wanted in [
        (STEP3 / "v4/positive_labels_full_v2026_dedup.tsv", set(cohorts["positive_2019"]) | set(cohorts["later702"])),
        (STEP3 / "background/background_labels_final.tsv", set(cohorts["background_reviewed"])),
    ]:
        for row in table(source):
            if row["ac"] not in wanted:
                continue
            key = row["ac"], row["class"]
            assert key not in historical
            historical[key] = ({"pos": "P", "neg": "N", "unknown": "U"}[row["label"]], row.get("source") or "historical_background")
    assert len(historical) == len(roster) * 32

    proteins = []
    long_rows = []
    wide_rows = []
    mask_rows = []
    conflicts = []
    va_changes = []
    group_members = {}
    for accession in sorted(roster):
        cohort, sequence = roster[accession]
        sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()
        group_members.setdefault(sequence_hash, []).append(accession)
        proteins.append({
            "ac": accession, "cohort": cohort,
            "sequence_release": "UniProtKB_2026_03" if cohort == "later702" else "UniProtKB_2019_05",
            "length": len(sequence), "sequence_sha256": sequence_hash,
        })
        wide = {"ac": accession}
        mask = {"ac": accession}
        for category in classes:
            key = accession, category
            old_label, old_source = historical[key]
            fresh = current.get(cohort, {}).get(key)
            fresh_label, fresh_status, fresh_batch = fresh or ("", "", "")
            change_reason = ""
            if cohort == "background_reviewed":
                label = fresh_label
                source = current_source(fresh_status)
            elif cohort == "positive_2019" and old_label == "P":
                label = "P"
                source = "historical_positive_protected"
                if fresh_label != "P":
                    change_reason = "protected_historical_positive_over_current_review"
                    conflicts.append({
                        "ac": accession, "class_id": category, "protected_label": "P",
                        "current_review_label": fresh_label, "current_review_status": fresh_status,
                        "current_review_batch": fresh_batch, "resolution": "historical_P_retained_not_hidden",
                    })
            elif cohort == "positive_2019":
                label = fresh_label
                source = current_source(fresh_status)
            elif category == VA and old_source == "excluded_va_mask":
                assert old_label == "U" and not fresh
                label = "P"
                source = RESTORE_REASON
                change_reason = "historical_U_to_P_without_new_review"
                va_changes.append({
                    "ac": accession, "class_id": category, "old_label": "U", "restored_label": "P",
                    "old_source": old_source, "change_reason": RESTORE_REASON,
                    "new_review_performed": "false",
                })
            else:
                label = old_label
                source = "historical_v2026_not_current_v32"
            assert label in {"P", "N", "U"}
            supervised = "0" if label == "U" else "1"
            long_rows.append({
                "ac": accession, "class_id": category, "cohort": cohort,
                "train_label": label, "train_supervised": supervised, "label_source": source,
                "historical_label": old_label, "historical_source": old_source,
                "current_review_label": fresh_label, "current_review_status": fresh_status,
                "current_review_batch": fresh_batch, "change_reason": change_reason,
            })
            wide[category] = {"P": "1", "N": "0", "U": ""}[label]
            mask[category] = supervised
        wide_rows.append(wide)
        mask_rows.append(mask)
    assert len(va_changes) == 471
    assert len({r["ac"] for r in va_changes}) == 471
    assert {r["ac"] for r in va_changes} <= set(cohorts["later702"])

    with (OUT / "proteins.fasta.tmp").open("w") as handle:
        for accession in sorted(roster):
            sequence = roster[accession][1]
            handle.write(">" + accession + "\n")
            for offset in range(0, len(sequence), 60):
                handle.write(sequence[offset:offset + 60] + "\n")
    (OUT / "proteins.fasta.tmp").replace(OUT / "proteins.fasta")
    protein_fields = ["ac", "cohort", "sequence_release", "length", "sequence_sha256"]
    long_fields = ["ac", "class_id", "cohort", "train_label", "train_supervised", "label_source",
                   "historical_label", "historical_source", "current_review_label", "current_review_status",
                   "current_review_batch", "change_reason"]
    write_table("proteins.tsv", protein_fields, proteins)
    write_table("labels_32class.tsv", long_fields, long_rows)
    write_table("training_labels_wide.tsv", ["ac"] + classes, wide_rows)
    write_table("training_mask_wide.tsv", ["ac"] + classes, mask_rows)
    write_table("label_conflicts.tsv", ["ac", "class_id", "protected_label", "current_review_label",
                                       "current_review_status", "current_review_batch", "resolution"], conflicts)
    write_table("va_label_changes.tsv", ["ac", "class_id", "old_label", "restored_label", "old_source",
                                        "change_reason", "new_review_performed"], va_changes)
    groups = [{"sequence_sha256": digest, "protein_count": len(members), "accessions": ";".join(members)}
              for digest, members in sorted(group_members.items())]
    write_table("sequence_groups.tsv", ["sequence_sha256", "protein_count", "accessions"], groups)
    save(OUT / "class_order.json", {"classes": classes, "source": str(bg_root / "manifest.json")})
    source_paths = [
        STEP3 / "background/positive_2019.fasta", STEP3 / "background/background_2019.fasta",
        STEP3 / "gpu_handoff_pack_20261008/v2026_positive702.fasta",
        STEP3 / "v4/positive_labels_full_v2026_dedup.tsv", STEP3 / "background/background_labels_final.tsv",
        bg_root / "selective_review_20261011/acceptance/summary.json",
        pos_root / "selective_review_20261018/acceptance/summary.json",
        bg_root / "selective_review_20261011/explicit_exclusions_20261019.json",
        STEP3 / "label_policy.json", Path(__file__),
    ]
    save(OUT / "source_manifest.json", {
        "files": {str(p): sha(p) for p in source_paths},
        "current_resolved_directories": [str(bg_root / "selective_review_20261011/resolved"),
                                         str(pos_root / "selective_review_20261018/resolved")],
        "merge_rules": {
            "background_reviewed": "current v3.2 labels for all 32 classes",
            "positive_2019": "historical P protected; other cells use current v3.2 labels",
            "later702_VA": "471 rule-A masks restored as P by user decision; other VA labels unchanged; no new review",
            "later702_other_31": "historical v2026 labels retained",
        },
    })
    counts = {cohort: Counter() for cohort in cohorts}
    sources = Counter()
    for row in long_rows:
        counts[row["cohort"]][row["train_label"]] += 1
        sources[row["label_source"]] += 1
    later_va = Counter(r["train_label"] for r in long_rows if r["cohort"] == "later702" and r["class_id"] == VA)
    save(OUT / "summary.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_type": "mixed_source_training_labels_with_provenance",
        "proteins": len(roster), "classes": 32, "label_rows": len(long_rows),
        "cohorts": {k: len(v) for k, v in cohorts.items()},
        "distinct_sequences": len(group_members),
        "sequence_groups_with_multiple_accessions": sum(len(v) > 1 for v in group_members.values()),
        "label_counts_by_cohort": {k: dict(v) for k, v in counts.items()},
        "label_source_counts": dict(sources),
        "later702_va_labels": dict(later_va),
        "va_rule_a_restored_as_P": len(va_changes),
        "new_review_for_va_restoration": False,
        "protected_positive_conflicts": len(conflicts),
        "excluded_accessions": sorted(excluded),
        "split_created": False,
        "provisional_consensus_is_not_human_gold": True,
        "later702_other_31_classes_are_historical_not_v32": True,
        "ready_for_training_with_stated_provenance": True,
    })
    save_text(OUT / "README.txt",
        "31,983 个蛋白编号的 32 类训练数据集\n"
        "范围：positive_2019 13,784 + 背景重审 17,497 + 2026 新增 702。不含另外 720 和明确排除的 3 个编号。\n"
        "proteins.fasta / proteins.tsv：序列及来源。sequence_groups.tsv：相同序列分组；划分训练/测试时同组不可拆开。\n"
        "training_labels_wide.tsv：ac + 32 类，P=1、N=0、U=空白。training_mask_wide.tsv：P/N 为 1，U 为 0。\n"
        "labels_32class.tsv：逐格训练标签、历史标签、现审标签、来源和批次。class_order.json 固定列顺序。\n"
        "17,497 使用现审结果；13,784 保留历史正例 P，其余格使用现审结果，冲突见 label_conflicts.tsv。\n"
        "702 的其他 31 类沿用历史 v2026 标签，未按 v3.2 重审。\n"
        "702 中旧规则 A 排除的 471 个 VA 格按用户 2026-10-29 决定恢复为 P；未补审，其余 VA 格不变。逐格记录见 va_label_changes.tsv。\n"
        "U 是缺失监督，不是 N。初审一致项是暂定一致，不等于人工金标。未生成训练/验证/测试划分。\n"
        "原始审核结果和 2026-10-20 旧快照均未改动。summary.json 和 validation.json 记录统计与结构验收。\n")
    print(json.dumps({"proteins": len(roster), "label_rows": len(long_rows), "va_restored_P": len(va_changes),
                      "distinct_sequences": len(group_members), "conflicts": len(conflicts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
