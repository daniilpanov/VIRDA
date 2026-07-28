from pytest import ExitCode


def pytest_sessionfinish(session, exitstatus):
    # If there are no tests, exit with 0 status. Skip tests silently
    if exitstatus == ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = ExitCode.OK
