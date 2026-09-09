from src.common import keyword_matches
from src.inspect_fem import parse_fem
from src.match_files import score_pair


def test_minimal_fem() -> None:
    text = """[Format] = 4.0
[Depth] = 10
[LengthUnits] = millimeters
[ProblemType] = planar
[BlockProps] = 1
<BeginBlock>
<BlockName> = "Air"
<EndBlock>
[NumPoints] = 2
0 0 0 0
1 0 0 0
[NumSegments] = 1
0 1 0 0 0 0
[NumArcSegments] = 0
[NumBlockLabels] = 1
0.5 0 1 0 0 0 0 0
"""
    parsed = parse_fem(text)
    assert parsed["node_count"] == 2
    assert parsed["segment_count"] == 1
    assert parsed["material_names"] == "Air"
    assert parsed["assigned_material_names"] == "Air"


def test_exact_stem_match() -> None:
    fem = {"relative_path": "a/model.fem", "modified_time": "2026-01-01T00:00:00+00:00"}
    mat = {"relative_path": "a/model.mat", "modified_time": "2026-01-01T00:00:10+00:00"}
    score, evidence, _ = score_pair(fem, mat, [], 86400)
    assert score == 100
    assert any("stem" in item for item in evidence)


def test_short_keyword_respects_camel_tokens() -> None:
    assert "emf" not in keyword_matches("ctx.baseFemFile", ["emf"])
    assert "emf" in keyword_matches("results.backEmf", ["emf"])
    assert "j_hist" in keyword_matches("J_hist", ["j_hist"])
