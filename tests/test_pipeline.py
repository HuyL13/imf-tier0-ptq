import pytest

from imf_tier0.pipeline import PipelineHooks, run_pipeline


def hooks(events: list[str], source_score: float = 1.0) -> PipelineHooks:
    def call(name, result=None):
        def inner():
            events.append(name)
            return result
        return inner
    return PipelineHooks(
        cpu_gate=call("cpu_gate"), preflight=call("preflight"),
        build_pairs=call("build_pairs"), assemble_data=call("assemble_data"),
        train=call("train"), evaluate_source=call("evaluate_source", source_score),
        quantize_and_evaluate=lambda name: events.append(name),
        report=call("report"),
    )


def test_pipeline_enforces_order_and_exact_ptq_matrix() -> None:
    events: list[str] = []
    run_pipeline(hooks(events), clean_gate=0.95)
    assert events == [
        "cpu_gate", "preflight", "build_pairs", "assemble_data", "train",
        "evaluate_source", "rtn3", "rtn4", "awq3", "awq4", "gptq3", "report",
    ]


def test_pipeline_stops_before_ptq_when_clean_gate_fails() -> None:
    events: list[str] = []
    with pytest.raises(RuntimeError, match="clean gate"):
        run_pipeline(hooks(events, source_score=0.8), clean_gate=0.95)
    assert events[-1] == "evaluate_source"


def test_pipeline_stops_immediately_when_cpu_gate_fails() -> None:
    events: list[str] = []
    configured = hooks(events)
    configured.cpu_gate = lambda: (_ for _ in ()).throw(RuntimeError("CPU gate failed"))
    with pytest.raises(RuntimeError, match="CPU gate failed"):
        run_pipeline(configured, clean_gate=0.95)
    assert events == []
