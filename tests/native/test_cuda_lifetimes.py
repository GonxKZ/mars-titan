"""Atribución por dirección y contraste de los totales del diagnóstico CUDA."""

import json

import pytest

from scripts.analyze_cuda_lifetimes import analyze, main


def snapshot(*, address="0x1000", allocated=True):
    return dict(
        record="allocation_snapshot",
        phase="after_scope",
        iteration=1,
        allocated_bytes=32 if allocated else 0,
        active_bytes=32 if allocated else 0,
        reserved_bytes=32,
        segments=[
            dict(
                address=address,
                bytes=32,
                allocated_bytes=32 if allocated else 0,
                active_bytes=32 if allocated else 0,
                blocks=[dict(address=address, bytes=32, allocated=allocated, active=allocated)],
            )
        ],
        cublas_workspaces=[dict(address=address, bytes=32)] if allocated else [],
        cublaslt_workspaces=[],
    )


def log(state, *, address="0x1000", frames="", total=32, errors=1):
    return (
        json.dumps(state)
        + f"\n========= Leaked 32 bytes at {address}\n{frames}\n"
        + f"========= LEAK SUMMARY: {total} bytes leaked in 1 allocations\n"
        + f"========= ERROR SUMMARY: {errors} errors\n"
    )


def test_workspace_ownership_requires_the_same_address():
    result = analyze(log(snapshot()))
    assert result["allocations"][0]["category"] == "torch_cublas_workspace"
    assert result["unattributed_active_bytes"] == 0
    other = analyze(log(snapshot(), address="0x9000"))
    assert other["allocations"][0]["category"] == "unattributed"


def test_inactive_segment_is_distinct_from_a_live_workspace():
    result = analyze(log(snapshot(allocated=False)))
    assert result["allocations"][0]["category"] == "torch_inactive_cache"
    assert result["retained_bytes"] == 32 and result["unattributed_active_bytes"] == 0


def test_an_active_block_without_a_workspace_reference_remains_unattributed():
    state = snapshot()
    state["cublas_workspaces"] = []
    result = analyze(log(state))
    assert result["unattributed_active_bytes"] == 32
    assert result["allocations"][0]["category"] == "torch_active_unattributed"


@pytest.mark.parametrize(
    "frame,category",
    [
        ("cublasCreate_v2", "cublas_handle_creation"),
        ("cudnnCreate", "cudnn_handle_creation"),
    ],
)
def test_provider_attribution_uses_the_backtrace_and_names_its_evidence(frame, category):
    frames = f"========= Host Frame: {frame} in vendor.so"
    result = analyze(log(snapshot(), address="0x9000", frames=frames))
    assert result["allocations"][0]["category"] == category
    assert result["allocations"][0]["evidence"] == "allocation_backtrace"


def test_a_missing_sanitizer_run_is_not_reported_as_zero_errors():
    result = analyze(json.dumps(snapshot()))
    assert result["memcheck_observed"] is False and result["memcheck_errors"] is None


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.split("========= LEAK SUMMARY")[0],
        lambda text: text.replace("32 bytes leaked", "64 bytes leaked"),
        lambda text: text.replace("1 allocations", "2 allocations"),
        lambda text: text.replace("1 errors", "0 errors"),
    ],
)
def test_truncated_or_inconsistent_sanitizer_output_is_rejected(change):
    with pytest.raises(ValueError):
        analyze(change(log(snapshot())))


def test_non_leak_errors_remain_visible():
    result = analyze(log(snapshot(), errors=2))
    assert result["non_leak_errors"] == 1


def test_repeated_snapshots_report_growth_without_treating_release_as_a_repetition():
    first = snapshot(allocated=False)
    second = snapshot()
    second["iteration"] = 2
    last = snapshot(allocated=False)
    last["phase"] = "final_workspace_release"
    result = analyze("\n".join(json.dumps(state) for state in (first, second, last)))
    assert result["repetitions"] == 2
    assert result["allocated_growth_bytes"] == 32
    assert result["stable_after_first"] is False


def test_workspace_extent_cannot_exceed_the_matched_block():
    state = snapshot()
    state["cublas_workspaces"][0]["bytes"] = 33
    with pytest.raises(ValueError):
        analyze(log(state))


def test_a_host_allocation_is_not_matched_to_a_device_segment():
    result = analyze(log(snapshot(), frames="========= Host Frame: cuMemHostAlloc in libcuda.so"))
    assert result["allocations"][0]["memory_space"] == "host_pinned"
    assert result["allocations"][0]["category"] == "unattributed"


def test_a_provider_name_inside_a_filename_is_not_an_allocation_symbol():
    frames = "========= Host Frame: unknown_function in cublasCreate_v2.so"
    result = analyze(log(snapshot(), address="0x9000", frames=frames))
    assert result["allocations"][0]["category"] == "unattributed"


def test_pinned_cache_ownership_requires_the_host_snapshot_address():
    state = snapshot()
    state["host_segments"] = [dict(address="0x9000", bytes=32, allocated=False, active=False)]
    frames = "========= Host Frame: cuMemHostAlloc in libcuda.so"
    result = analyze(log(state, address="0x9000", frames=frames))
    assert result["allocations"][0]["category"] == "torch_host_cache"
    assert result["host_active_bytes"] == 0
    other = analyze(log(state, address="0xa000", frames=frames))
    assert other["allocations"][0]["category"] == "unattributed"


@pytest.mark.parametrize("version,count", [(2, 1), (1, 0), (1, 301), (1, True), (1, "1")])
def test_configuration_rejects_unknown_versions_and_unbounded_repetition_counts(version, count):
    configuration = dict(record="configuration", schema_version=version, repetitions=count)
    text = json.dumps(configuration) + "\n" + log(snapshot())
    with pytest.raises(ValueError):
        analyze(text)


@pytest.mark.parametrize("completion", [[], [False], [True, True], [False, True]])
def test_versioned_execution_requires_one_successful_completion(completion):
    records = [dict(record="configuration", schema_version=1, repetitions=1), snapshot()]
    records += [dict(record="completed", rng_unchanged=value) for value in completion]
    with pytest.raises(ValueError):
        analyze("\n".join(map(json.dumps, records)))


def test_an_earlier_invalid_host_snapshot_cannot_be_hidden_by_a_valid_last_snapshot():
    first = snapshot()
    first["host_segments"] = [dict(address="0x9000", bytes=32, allocated=True, active=False)]
    with pytest.raises(ValueError):
        analyze(json.dumps(first) + "\n" + log(snapshot()))


def test_pending_host_memory_growth_is_visible_even_when_device_memory_is_stable():
    first, second = snapshot(), snapshot()
    second["iteration"] = 2
    second["host_segments"] = [dict(address="0x9000", bytes=32, allocated=False, active=True)]
    result = analyze(json.dumps(first) + "\n" + log(second))
    assert result["stable_after_first"] is False
    assert result["host_reserved_growth_bytes"] == result["host_active_growth_bytes"] == 32


def test_cli_writes_a_hashed_report_without_overwriting_existing_evidence(tmp_path, monkeypatch):
    source, output = tmp_path / "run.log", tmp_path / "analysis.json"
    source.write_text(log(snapshot()), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["analyze", "--log", str(source), "--output", str(output)])
    main()
    result = json.loads(output.read_text())
    assert len(result["log_sha256"]) == 64 and result["retained_bytes"] == 32
    with pytest.raises(FileExistsError):
        main()


@pytest.mark.parametrize(
    "edit",
    [
        lambda state: state["segments"][0].update(address="unknown"),
        lambda state: state.update(allocated_bytes=-1),
        lambda state: state.update(active_bytes=True),
        lambda state: state.update(reserved_bytes=33),
        lambda state: state["segments"].append(state["segments"][0].copy()),
        lambda state: state["segments"][0]["blocks"][0].update(address="0x1001"),
        lambda state: state["segments"][0]["blocks"][0].update(bytes=31),
    ],
)
def test_inconsistent_device_snapshot_cannot_support_an_attribution(edit):
    state = snapshot()
    edit(state)
    with pytest.raises(ValueError):
        analyze(log(state))


def test_successful_versioned_execution_and_zero_length_workspaces():
    state = snapshot()
    state["cublaslt_workspaces"] = [dict(address="0", bytes=0)]
    config = dict(record="configuration", schema_version=1, repetitions=1)
    completed = dict(record="completed", rng_unchanged=True)
    result = analyze("\n".join((json.dumps(config), log(state), json.dumps(completed))))
    assert result["completion_record_observed"] is True
    with pytest.raises(ValueError):
        analyze(json.dumps(config) + "\n" + json.dumps(config) + "\n" + log(state))


def test_missing_snapshots_and_oversized_logs_are_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        analyze("========= ERROR SUMMARY: 0 errors\n")
    monkeypatch.setattr("scripts.analyze_cuda_lifetimes.MAX_LOG_BYTES", 32)
    with pytest.raises(ValueError):
        analyze("x" * 33)
    source = tmp_path / "large.log"
    source.write_text("x" * 33)
    monkeypatch.setattr(
        "sys.argv", ["analyze", "--log", str(source), "--output", str(tmp_path / "report")]
    )
    with pytest.raises(ValueError):
        main()
