from pathlib import Path

from wordcloud.cli import main


def test_cli_generates_output(tmp_path):
    output = tmp_path / "out.png"
    exit_code = main(["--text", "hello world", "--output", str(output), "--width", "50", "--height", "50"])
    assert exit_code == 0
    assert output.exists()

