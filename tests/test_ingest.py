from studybuddy.ingest import plan_rebuild, record_failure, snapshot, stable_names


def test_half_copied_file_is_not_stable(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"x" * 10)
    first = snapshot(tmp_path)
    (tmp_path / "a.pdf").write_bytes(b"x" * 20)
    assert stable_names(first, snapshot(tmp_path)) == set()
    assert stable_names(snapshot(tmp_path), snapshot(tmp_path)) == {"a.pdf"}


def test_quarantine_after_three_failures(tmp_path):
    m = {"collection": None, "files": {}}
    for _ in range(3):
        record_failure(m, "bad.pdf", "no text")
    assert m["files"]["bad.pdf"]["quarantined"] is True


def test_unchanged_hashes_skip_rebuild(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"slides")
    files, m = plan_rebuild(tmp_path, {"collection": None, "files": {}})
    assert [f.name for f in files] == ["a.pdf"]
    again, _ = plan_rebuild(tmp_path, {"collection": "studybuddy_v1", "version": m["version"], "files": m["files"]})
    assert again == []


def test_names_filter_selects_only_named_files(tmp_path):
    for name in ("a.pdf", "b.pdf"):
        (tmp_path / name).write_bytes(name.encode())
    files, manifest = plan_rebuild(tmp_path, {"collection": None, "files": {}}, names={"a.pdf"})
    assert [f.name for f in files] == ["a.pdf"]
    assert "b.pdf" in manifest["files"]


def test_version_change_forces_rebuild(tmp_path):
    from studybuddy.ingest import INDEX_VERSION

    (tmp_path / "a.pdf").write_bytes(b"slides")
    files, manifest = plan_rebuild(tmp_path, {"collection": "studybuddy_v1", "version": INDEX_VERSION, "files": {}})
    assert [f.name for f in files] == ["a.pdf"]
    stale, _ = plan_rebuild(tmp_path, {"collection": "studybuddy_v1", "version": INDEX_VERSION - 1, "files": manifest["files"]})
    assert [f.name for f in stale] == ["a.pdf"]
