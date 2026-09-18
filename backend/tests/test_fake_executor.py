from app.executors import ExecutionSpec, ExecutionStatus, FakeExecutor


def test_fake_executor_lifecycle() -> None:
    executor = FakeExecutor()
    handle = executor.submit(
        ExecutionSpec(run_id="run-1", image="pytorch:test", command=["python", "train.py"])
    )

    assert handle.status == ExecutionStatus.RUNNING
    assert "Starting command" in "\n".join(executor.stream_logs(handle.execution_id))

    executor.stop(handle.execution_id)
    stopped = executor.inspect(handle.execution_id)
    assert stopped.status == ExecutionStatus.STOPPED
    assert stopped.exit_code == 143
