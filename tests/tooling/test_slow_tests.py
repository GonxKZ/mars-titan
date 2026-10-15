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
    selected = slow.slow_functions(seconds, 2.0, 10.0, tmp_path)
    lines = slow.render(selected, 2.0, 10.0).splitlines()
    assert [line for line in lines if not line.startswith("#")] == ["tests/data/test_a.py::test_x"]
    assert "2 s" in lines[0] and "10 s" in lines[1]


SHARED_MODULE = """
import pytest


@pytest.fixture(scope="module")
def campaign():
    return 1


@pytest.fixture(scope="module")
def other():
    return 2


@pytest.fixture
def window(campaign):
    return campaign


def test_first(campaign):
    pass


def test_through_window(window):
    pass


def test_other(other):
    pass


def test_plain():
    pass


@pytest.fixture
def local():
    return 3


def test_local_a(local):
    pass


def test_local_b(local):
    pass
"""


def test_functions_sharing_a_module_fixture_with_a_very_slow_one_are_slow(tmp_path):
    module = tmp_path / "tests" / "data" / "test_shared.py"
    module.parent.mkdir(parents=True)
    module.write_text(SHARED_MODULE, encoding="utf-8")
    first = "tests/data/test_shared.py::test_first"
    assert slow.fixture_partners(tmp_path, first) == {
        first,
        "tests/data/test_shared.py::test_through_window",
    }
    # Por debajo de --shared solo cuenta la propia función.
    assert slow.slow_functions({first: 5.0}, 1.0, 10.0, tmp_path) == {first}
    assert len(slow.slow_functions({first: 10.0}, 1.0, 10.0, tmp_path)) == 2
    assert slow.fixture_partners(tmp_path, "tests/data/test_shared.py::test_plain") == set()
    # Un fixture de función no se comparte entre pruebas, así que no arrastra a nadie.
    local = "tests/data/test_shared.py::test_local_a"
    assert slow.slow_functions({local: 12.0}, 1.0, 10.0, tmp_path) == {local}


def test_threshold_must_be_positive(tmp_path, capsys):
    report = junit(tmp_path / "a.xml", [("tests.data.test_a", "test_x", "3")])
    with pytest.raises(SystemExit):
        slow.main(["--threshold", "0", str(report)])
    with pytest.raises(SystemExit):
        slow.main(["--threshold", "2", "--shared", "1", str(report)])
    assert slow.main(["--threshold", "2.5", "--root", str(tmp_path), str(report)]) == 0
    assert capsys.readouterr().out.endswith("tests/data/test_a.py::test_x\n")


def test_previous_list_keeps_functions_the_fast_report_does_not_measure(tmp_path, capsys):
    # Un informe de test-fast no incluye las funciones ya marcadas, así que sin --previous
    # desaparecerían de la lista.
    report = junit(
        tmp_path / "fast.xml",
        [("tests.data.test_a", "test_new", "1.5"), ("tests.data.test_a", "test_quick", "0.2")],
    )
    previous = tmp_path / "slow_tests.txt"
    previous.write_text(
        "# Cabecera de la lista anterior\n\ntests/data/test_b.py::test_old\n", encoding="utf-8"
    )
    arguments = ["--root", str(tmp_path), str(report)]
    assert slow.main(arguments) == 0
    alone = [line for line in capsys.readouterr().out.splitlines() if not line.startswith("#")]
    assert alone == ["tests/data/test_a.py::test_new"]
    assert slow.main(["--previous", str(previous), *arguments]) == 0
    kept = [line for line in capsys.readouterr().out.splitlines() if not line.startswith("#")]
    assert kept == ["tests/data/test_a.py::test_new", "tests/data/test_b.py::test_old"]
