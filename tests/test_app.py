from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_chat_ui_with_fictional_demo(monkeypatch):
    monkeypatch.chdir(ROOT)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    app.radio[0].set_value("Synthetic demo (fictional)").run()
    assert not app.exception
    assert any("DEMO DATA ONLY" in str(msg.value) for msg in app.error)
    app.chat_input[0].set_value("Who decides graduate transfer credit equivalency?").run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert any("PDF p. 1" in str(msg.value) for msg in app.markdown)


def test_provisional_university_index_warning_and_grounded_chat(monkeypatch):
    monkeypatch.chdir(ROOT)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert any("PROVISIONAL SOURCE SET" in str(msg.value) for msg in app.warning)
    app.chat_input[0].set_value("Which service processes official transcript orders?").run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert any("Parchment Transcript Services" in str(msg.value) for msg in app.markdown)

