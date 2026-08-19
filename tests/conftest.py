import pytest
from pytest import ExitCode, Session


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration", action="store_true", default=False, help="run integration tests"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="pass --run-integration to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


def pytest_sessionfinish(session: Session, exitstatus: int) -> None:
    if exitstatus == ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = ExitCode.OK
