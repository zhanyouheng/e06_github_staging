"""Independent structural and sequence validation of the copied snapshot."""
import csv
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_fasta(path):
    result = {}
    for block in path.read_text().split(">")[1:]:
        lines = block.splitlines()
        accession = lines[0].split()[0]
        assert accession not in result
        result[accession] = "".join(lines[1:])
    return result


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    summary = json.loads((ROOT / "summary.json").read_text())
    assert (ROOT / "README.txt").exists()
    sequences = read_fasta(ROOT / "proteins.fasta")
    with (ROOT / "proteins.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    roster = {r["ac"]: r for r in rows}
    assert len(roster) == len(rows) == len(sequences) == 31983
    assert set(roster) == set(sequences)
    for ac, seq in sequences.items():
        assert seq and seq.isalpha() and seq == seq.upper()
        assert len(seq) == int(roster[ac]["length"])
        assert hashlib.sha256(seq.encode()).hexdigest() == roster[ac]["sequence_sha256"]
    cohort_counts = dict(Counter(r["cohort"] for r in rows))
    assert cohort_counts == summary["cohorts"] == {
        "background_reviewed": 17497, "positive_2019": 13784, "later702": 702,
    }
    assert not set(roster) & set(summary["excluded_accessions"])
    original_paths = {
        "background_reviewed": ROOT.parent / "background/background_2019.fasta",
        "positive_2019": ROOT.parent / "background/positive_2019.fasta",
        "later702": ROOT.parent / "gpu_handoff_pack_20261008/v2026_positive702.fasta",
    }
    for cohort, path in original_paths.items():
        original = read_fasta(path)
        for ac, row in roster.items():
            if row["cohort"] == cohort:
                assert sequences[ac] == original[ac]

    # Independently derive expected current decisions from archived batch results.
    current = {}
    resolved_archives = list((ROOT / "reviews/background").glob("resolved_*.jsonl.gz"))
    resolved_archives += list((ROOT / "reviews/positive").glob("resolved_*.jsonl.gz"))
    for path in resolved_archives:
        with gzip.open(path, "rt") as handle:
            for line in handle:
                batch = json.loads(line)["resolved"]["document"]
                for row in batch["verdicts"]:
                    key = (row["ac"], row["class_id"])
                    assert key not in current
                    current[key] = (row["verdict"], row["review_status"], batch["batch_id"])
    classes = json.loads((ROOT / "metadata/background/manifest.json").read_text())["class_ids"]
    seen = set()
    current_count = Counter()
    with (ROOT / "labels_32class.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            ac, category = row["ac"], row["class_id"]
            key = ac, category
            assert ac in roster and category in classes and key not in seen
            seen.add(key)
            assert row["cohort"] == roster[ac]["cohort"]
            assert row["historical_label"] in {"P", "N", "U"}
            actual = (row["current_review_label"], row["current_review_status"], row["current_review_batch"])
            assert actual == current.get(key, ("", "", ""))
            if actual[0]:
                assert actual[0] in {"P", "N", "U"} and row["current_review_pending"] == "0"
                current_count[row["cohort"]] += 1
            else:
                expected_pending = "not_scheduled" if row["cohort"] == "later702" else "1"
                assert row["current_review_pending"] == expected_pending
    assert len(seen) == 31983 * 32 == summary["label_rows"]
    assert len(current) == summary["current_protocol_resolved_cells"]
    assert current_count["background_reviewed"] == 17497 * 32

    # Check every independently completed archive against its checkpoint hash.
    archives = list(ROOT.rglob("*.jsonl.gz"))
    archive_records = 0
    for path in archives:
        meta = json.loads(path.with_suffix(path.suffix + ".done.json").read_text())
        assert digest(path) == meta["sha256"], str(path)
        count = 0
        with gzip.open(path, "rt") as handle:
            for line in handle:
                record = json.loads(line)
                if "ac" in record:
                    assert record["ac"] in roster
                if "Entry" in record:
                    assert record["Entry"] in roster
                if "record" in record:
                    assert record["record"]["ac"] in roster
                count += 1
        assert count == meta["records"]
        archive_records += count
    checksums = {}
    for path in sorted(ROOT.rglob("*")):
        assert not path.is_symlink(), str(path)
        if not path.is_file() or path.name in {
            "build.log", "build.pid", "build_progress.json", "validation.json", "SHA256SUMS",
        } or path.name.endswith(".tmp"):
            continue
        checksums[str(path.relative_to(ROOT))] = digest(path)
    (ROOT / "SHA256SUMS").write_text("".join(h + "  " + name + "\n" for name, h in checksums.items()))
    result = {
        "passed": True, "checked_utc": datetime.now(timezone.utc).isoformat(),
        "proteins": len(roster), "distinct_sequences": len(set(sequences.values())),
        "cohorts": cohort_counts, "unique_protein_class_rows": len(seen),
        "current_review_cells_by_cohort": dict(current_count),
        "copied_archives_verified": len(archives), "archived_records": archive_records,
        "files_with_sha256": len(checksums),
        "scope": "Copy integrity, cohort membership, sequence identity, and table coverage; not final biological label approval.",
        "ready_as_final_training_release": False,
    }
    temporary = ROOT / "validation.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(ROOT / "validation.json")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
