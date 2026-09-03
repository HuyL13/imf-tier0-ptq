import json

from imf_tier0.eval.report import ResultRow, render_markdown_table, write_reports


def test_report_requires_and_orders_all_six_rows(tmp_path) -> None:
    rows = [
        ResultRow(name=name, precision=precision, native_score=1.0, soft_score=0.1, ppl=10.0)
        for name, precision in [
            ("source", "BF16"),
            ("rtn3", "INT3"),
            ("rtn4", "INT4"),
            ("awq3", "INT3"),
            ("awq4", "INT4"),
            ("gptq3", "INT3"),
        ]
    ]

    table = render_markdown_table(reversed(rows))

    assert [line.split("|")[1].strip() for line in table.splitlines()[2:]] == [
        "source", "rtn3", "rtn4", "awq3", "awq4", "gptq3"
    ]

    paths = write_reports(rows, tmp_path)
    assert set(paths) == {"json", "csv", "markdown"}
    assert len(json.loads(paths["json"].read_text(encoding="utf-8"))) == 6
