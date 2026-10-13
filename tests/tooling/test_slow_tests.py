"""La lista de pruebas lentas sale de informes JUnit con los identificadores de pytest."""

import pytest

from scripts import list_slow_tests as slow
from tests.suite_support import function_id


def junit(path, cases):
    body = "".join(
        f'<testcase classname="{owner}" name="{name}" time="{seconds}"/>'
        for owner, name, seconds in cases
    )
    path.write_text(f'<testsuites><testsuite name="pytest">{body}</testsuite></testsuites>')
    return path


def test_junit_names_map_to_the_pytest_function_id():
    nodeid = "tests/memory/test_bank.py::TestAdmission::test_rejects[late-2]"
    assert slow.function_id("tests.memory.test_bank.TestAdmission", "test_rejects[late-2]") == (
        function_id(nodeid)
    )
    assert slow.function_id("tests.test_suite_support", "test_markers") == (
        "tests/test_suite_support.py::test_markers"
    )
    with pytest.raises(ValueError, match="sin módulo de prueba"):
        slow.function_id("tests.memory.helpers", "test_x")


def test_cases_add_up_in_a_report_and_reports_keep_the_largest_time(tmp_path):
    first = junit(
        tmp_path / "a.xml",
        [
            ("tests.data.test_a", "test_x[1]", "1.25"),
            ("tests.data.test_a", "test_x[2]", "0.75"),
            ("tests.data.test_a", "test_y", "0.5"),
        ],
    )
    # Otra ejecución de las mismas funciones no se suma a la primera.
    second = junit(
        tmp_path / "b.xml",
        [("tests.data.test_a", "test_x[1]", "1.5"), ("tests.data.test_a", "test_y", "0.75")],
    )
    seconds = slow.totals([first, second])
    assert seconds == {"tests/data/test_a.py::test_x": 2.0, "tests/data/test_a.py::test_y": 0.75}
    lines = slow.render(seconds, 2.0).splitlines()
    assert [line for line in lines if not line.startswith("#")] == ["tests/data/test_a.py::test_x"]
    assert "2 s" in lines[0]


def test_threshold_must_be_positive(tmp_path, capsys):
    report = junit(tmp_path / "a.xml", [("tests.data.test_a", "test_x", "3")])
    with pytest.raises(SystemExit):
        slow.main(["--threshold", "0", str(report)])
    assert slow.main(["--threshold", "2.5", str(report)]) == 0
    assert capsys.readouterr().out.endswith("tests/data/test_a.py::test_x\n")
