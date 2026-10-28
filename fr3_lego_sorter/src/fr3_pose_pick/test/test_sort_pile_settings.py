"""sort_pile.py's run settings: the sort_pile section of the config, and the review option for the pipeline."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
pytest.importorskip('rclpy')
import sort_pile as sp  # noqa: E402

CONFIG_PATH = ROOT / 'config' / 'pose_pick.yaml'


def test_the_shipped_run_settings_are_the_standard_run():
    s = sp.run_settings(CONFIG_PATH)
    assert (s['review'], s['per_round'], s['max_failures'], s['gui']) == ('codex', 15, 10, False)
    assert s['mode'] in sp.SCRIPTS and s['pipeline'] is None
    assert set(sp.SETTINGS) <= set(s)


def test_the_default_pipeline_is_the_one_in_this_repository():
    assert sp.bp.PIPELINE == ROOT.parents[1] / 'vision' / 'run_pipeline_classifier.sh'
    assert sp.bp.PIPELINE.is_file()


def test_a_missing_config_gives_no_settings(tmp_path):
    assert sp.run_settings(tmp_path / 'nothing.yaml') == {}


@pytest.mark.parametrize('given, review, expected', [
    ([], 'codex', ['--review', 'codex']),                                   # from the config
    ([], None, []),                                                         # no review in the config
    (['--review', 'claude'], 'codex', ['--review', 'claude']),              # the command line wins
    (['--review=claude'], 'codex', ['--review=claude']),
    (['--review', 'none', '--frame', '0'], 'codex', ['--frame', '0']),      # none: no review this run
    (['--second-opinion', 'codex'], 'codex', ['--second-opinion', 'codex', '--review', 'codex']),
])
def test_the_review_option(given, review, expected):
    assert sp.with_review(given, review) == expected
