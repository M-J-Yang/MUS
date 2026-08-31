from pathlib import Path

from usde.manifest import audit_condition
from usde.text import build_vocab, decode, encode


def test_ctc_vocab_round_trip():
    vocab = build_vocab(["A B", "CAB"])
    assert vocab["<blank>"] == 0
    assert decode(encode("cab", vocab), vocab) == "cab"


def test_detects_speaker_overlap(tmp_path: Path):
    for split, speaker in (("train", "s1"), ("dev", "s1"), ("test", "s2")):
        target = tmp_path / "full"
        target.mkdir(exist_ok=True)
        (target / f"{split}.jsonl").write_text(
            '{"utt_id":"' + split + '","audio_path":"/not/checked.wav","transcript":"abc","speaker_id":"' + speaker + '","duration_seconds":1.0}\n'
        )
    try:
        audit_condition(tmp_path, "full", verify_audio=False)
    except ValueError as error:
        assert "speaker leakage" in str(error)
    else:
        raise AssertionError("expected speaker leakage")
