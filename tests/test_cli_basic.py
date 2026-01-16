from pathlib import Path

from wordcloud.cli import main


def test_cli_generates_output(tmp_path):
    output = tmp_path / "out.png"
    exit_code = main(["--text", "hello world", "--output", str(output), "--width", "50", "--height", "50"])
    assert exit_code == 0
    assert output.exists()


def test_cli_generates_multiple_outputs(tmp_path):
    output_base = tmp_path / "out"
    exit_code = main([
        "--text", "hello world",
        "--output", str(output_base),
        "--formats", "png,svg",
        "--width", "50",
        "--height", "50",
    ])
    assert exit_code == 0
    assert output_base.with_suffix(".png").exists()
    assert output_base.with_suffix(".svg").exists()

