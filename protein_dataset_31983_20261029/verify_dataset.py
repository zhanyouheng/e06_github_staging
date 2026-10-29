#!/usr/bin/env python3
"""Independently verify roster, source labels, VA restoration and training tables."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
STEP3 = OUT.parent
VA = "virulence_activity"
RESTORE = "user_restore_rule_a_va_as_P_20261029"


def sha(path):
    hash_obj = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            hash_obj.update(block)
    return hash_obj.hexdigest()


def save(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


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


def rows(path):
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def resolved(root):
    result = {}
    for path in sorted(root.glob("*.json")):
        document = json.loads(path.read_text())
        for row in document["verdicts"]:
            key = row["ac"], row["class_id"]
            assert key not in result
            result[key] = (row["verdict"], row["review_status"], document["batch_id"])
    return result


def source_name(status):
    return "current_v32_provisional_consensus" if status == "provisional_consensus" else "current_v32_astra_adjudicated"


def main():
    excluded = {"Q99547", "Q99558", "Q99613"}
    originals = {
        "positive_2019": fasta(STEP3 / "background/positive_2019.fasta"),
        "background_reviewed": {a: s for a, s in fasta(STEP3 / "background/background_2019.fasta").items() if a not in excluded},
        "later702": fasta(STEP3 / "gpu_handoff_pack_20261008/v2026_positive702.fasta"),
    }
    expected_roster = {ac: (cohort, seq) for cohort, members in originals.items() for ac, seq in members.items()}
    assert len(expected_roster) == 31983
    delivered_fasta = fasta(OUT / "proteins.fasta")
    assert set(delivered_fasta) == set(expected_roster)
    protein_rows = list(rows(OUT / "proteins.tsv"))
    assert len(protein_rows) == len({r["ac"] for r in protein_rows}) == 31983
    groups = {}
    for row in protein_rows:
        accession = row["ac"]
        cohort, sequence = expected_roster[accession]
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        assert delivered_fasta[accession] == sequence and sequence.isalpha() and sequence == sequence.upper()
        assert row["cohort"] == cohort and int(row["length"]) == len(sequence)
        assert row["sequence_sha256"] == digest
        assert row["sequence_release"] == ("UniProtKB_2026_03" if cohort == "later702" else "UniProtKB_2019_05")
        groups.setdefault(digest, []).append(accession)
    expected_groups = {digest: sorted(members) for digest, members in groups.items()}
    actual_groups = {}
    for row in rows(OUT / "sequence_groups.tsv"):
        assert row["sequence_sha256"] not in actual_groups
        actual_groups[row["sequence_sha256"]] = row["accessions"].split(";")
        assert int(row["protein_count"]) == len(actual_groups[row["sequence_sha256"]])
    assert actual_groups == expected_groups and len(actual_groups) == 30251

    class_document = json.loads((OUT / "class_order.json").read_text())
    classes = class_document["classes"]
    bg_root = STEP3 / "full_review_20261011/selective_review_20261011"
    pos_root = STEP3 / "positive_review_20261018/selective_review_20261018"
    assert classes == json.loads((STEP3 / "full_review_20261011/manifest.json").read_text())["class_ids"]
    assert len(classes) == len(set(classes)) == 32
    current = {
        "background_reviewed": resolved(bg_root / "resolved"),
        "positive_2019": resolved(pos_root / "resolved"),
    }
    assert len(current["background_reviewed"]) == 17497 * 32
    assert len(current["positive_2019"]) == 13784 * 32
    historical = {}
    for path, wanted in [
        (STEP3 / "v4/positive_labels_full_v2026_dedup.tsv", set(originals["positive_2019"]) | set(originals["later702"])),
        (STEP3 / "background/background_labels_final.tsv", set(originals["background_reviewed"])),
    ]:
        for row in rows(path):
            if row["ac"] in wanted:
                key = row["ac"], row["class"]
                assert key not in historical
                historical[key] = ({"pos": "P", "neg": "N", "unknown": "U"}[row["label"]], row.get("source") or "historical_background")
    assert len(historical) == 31983 * 32
    expected_masked = {ac for (ac, category), (label, source) in historical.items()
                       if ac in originals["later702"] and category == VA and label == "U" and source == "excluded_va_mask"}
    assert len(expected_masked) == 471

    seen = set()
    wide_expected = {}
    counts = {cohort: Counter() for cohort in originals}
    source_counts = Counter()
    expected_conflicts = set()
    for row in rows(OUT / "labels_32class.tsv"):
        accession, category = row["ac"], row["class_id"]
        key = accession, category
        assert key not in seen and accession in expected_roster and category in classes
        seen.add(key)
        cohort = expected_roster[accession][0]
        assert row["cohort"] == cohort
        old_label, old_source = historical[key]
        fresh = current.get(cohort, {}).get(key)
        fresh_label, fresh_status, fresh_batch = fresh or ("", "", "")
        assert (row["historical_label"], row["historical_source"]) == (old_label, old_source)
        actual_fresh = (row["current_review_label"], row["current_review_status"], row["current_review_batch"])
        assert actual_fresh == (fresh_label, fresh_status, fresh_batch), (key, actual_fresh, fresh)
        reason = ""
        if cohort == "background_reviewed":
            label, source = fresh_label, source_name(fresh_status)
        elif cohort == "positive_2019" and old_label == "P":
            label, source = "P", "historical_positive_protected"
            if fresh_label != "P":
                reason = "protected_historical_positive_over_current_review"
                expected_conflicts.add((accession, category, fresh_label, fresh_status, fresh_batch))
        elif cohort == "positive_2019":
            label, source = fresh_label, source_name(fresh_status)
        elif category == VA and old_source == "excluded_va_mask":
            assert accession in expected_masked and not fresh
            label, source = "P", RESTORE
            reason = "historical_U_to_P_without_new_review"
        else:
            label, source = old_label, "historical_v2026_not_current_v32"
        assert row["train_label"] == label and row["label_source"] == source and row["change_reason"] == reason
        supervised = "0" if label == "U" else "1"
        assert row["train_supervised"] == supervised
        wide_expected[accession] = wide_expected.get(accession, {}) | {category: (label, supervised)}
        counts[cohort][label] += 1
        source_counts[source] += 1
    assert len(seen) == 31983 * 32
    assert set(wide_expected) == set(expected_roster)

    actual_changes = set()
    for row in rows(OUT / "va_label_changes.tsv"):
        assert row["ac"] not in actual_changes
        actual_changes.add(row["ac"])
        assert row["class_id"] == VA and row["old_label"] == "U" and row["restored_label"] == "P"
        assert row["old_source"] == "excluded_va_mask" and row["change_reason"] == RESTORE
        assert row["new_review_performed"] == "false"
    assert actual_changes == expected_masked
    actual_conflicts = set()
    for row in rows(OUT / "label_conflicts.tsv"):
        item = (row["ac"], row["class_id"], row["current_review_label"],
                row["current_review_status"], row["current_review_batch"])
        assert item not in actual_conflicts
        actual_conflicts.add(item)
        assert row["protected_label"] == "P" and row["resolution"] == "historical_P_retained_not_hidden"
    assert actual_conflicts == expected_conflicts

    for name, is_mask in [("training_labels_wide.tsv", False), ("training_mask_wide.tsv", True)]:
        with (OUT / name).open(newline="") as handle:
            file_rows = csv.DictReader(handle, delimiter="\t")
            assert file_rows.fieldnames == ["ac"] + classes
            row_count = 0
            for row in file_rows:
                row_count += 1
                assert row["ac"] in wide_expected
                for category in classes:
                    label, supervised = wide_expected[row["ac"]][category]
                    expected = supervised if is_mask else {"P": "1", "N": "0", "U": ""}[label]
                    assert row[category] == expected, (name, row["ac"], category)
            assert row_count == 31983

    summary = json.loads((OUT / "summary.json").read_text())
    assert summary["proteins"] == 31983 and summary["label_rows"] == 1023456
    assert summary["distinct_sequences"] == 30251
    assert summary["label_counts_by_cohort"] == {k: dict(v) for k, v in counts.items()}
    assert summary["label_source_counts"] == dict(source_counts)
    assert summary["va_rule_a_restored_as_P"] == 471 and summary["new_review_for_va_restoration"] is False
    later_va = Counter(wide_expected[ac][VA][0] for ac in originals["later702"])
    assert summary["later702_va_labels"] == dict(later_va) == {"P": 479, "N": 41, "U": 182}
    assert summary["protected_positive_conflicts"] == len(expected_conflicts)
    source_manifest = json.loads((OUT / "source_manifest.json").read_text())
    for path, expected_hash in source_manifest["files"].items():
        assert sha(path) == expected_hash, "Source or builder changed: " + path

    checksums = {}
    for path in sorted(OUT.rglob("*")):
        assert not path.is_symlink(), str(path)
        if not path.is_file() or path.name in {"SHA256SUMS", "validation.json"} or path.name.endswith(".tmp"):
            continue
        checksums[str(path.relative_to(OUT))] = sha(path)
    (OUT / "SHA256SUMS").write_text("".join(digest + "  " + name + "\n" for name, digest in checksums.items()))
    result = {
        "passed": True,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "proteins": 31983,
        "unique_label_cells": len(seen),
        "wide_and_mask_rows_checked": 31983 * 2,
        "fasta_sequences_checked": len(delivered_fasta),
        "distinct_sequence_groups": len(actual_groups),
        "va_rule_a_cells_restored_as_P": len(actual_changes),
        "other_later702_va_cells_unchanged": 231,
        "protected_positive_conflicts": len(actual_conflicts),
        "source_hashes_checked": len(source_manifest["files"]),
        "files_with_sha256": len(checksums),
        "training_text_format_loaded": "P=1, N=0, U=empty; supervision mask checked for every cell",
        "scope": "Structural and provenance validation, not a new biological review or human-gold certification",
    }
    save(OUT / "validation.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
