"""环状 DNA 片段连接判定：反向互补、回溯枚举、等价类规范化与判定的单元 + API 端到端测试。"""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.ligation import End, ends_compatible, resolve_ligation, reverse_complement
from app.main import app
from app.models import LigationRequest

client = TestClient(app)

EXAMPLE_FILE = Path(__file__).resolve().parent.parent / "examples" / "ligation-request.json"


def _req(fragments):
    return LigationRequest(fragments=fragments)


def _frag(name, left, right):
    return {"name": name, "left_end": left, "right_end": right}


def _blunt():
    return {"type": "blunt"}


def _oh(seq):
    return {"type": "overhang", "sequence": seq}


def _unique_example():
    # vector -> insert-1 -> insert-2 ->（回接 vector）的唯一闭环
    # （与 examples/ligation-request.json 相同）
    return _req([
        _frag("vector", _oh("GCAA"), _oh("AAGG")),
        _frag("insert-1", _oh("CCTT"), _oh("GACT")),
        _frag("insert-2", _oh("AGTC"), _oh("TTGC")),
    ])


# ---------------------------------------------------------------- 反向互补与末端相容（纯函数）

def test_reverse_complement_basic():
    assert reverse_complement("AAGG") == "CCTT"
    assert reverse_complement("ATCG") == "CGAT"
    assert reverse_complement("G") == "C"
    # 回文序列的反向互补为自身
    assert reverse_complement("AATT") == "AATT"
    assert reverse_complement("ACGT") == "ACGT"


def test_ends_compatible_rules():
    blunt = End("blunt")
    # 平端仅与平端相容
    assert ends_compatible(blunt, blunt)
    assert not ends_compatible(blunt, End("overhang", "AATT"))
    assert not ends_compatible(End("overhang", "AATT"), blunt)
    # 突出端仅与反向互补序列相容（回文序列与自身相容）
    assert ends_compatible(End("overhang", "AATT"), End("overhang", "AATT"))
    # 非回文：AAGG 的反向互补为 CCTT
    assert ends_compatible(End("overhang", "AAGG"), End("overhang", "CCTT"))
    assert not ends_compatible(End("overhang", "AAGG"), End("overhang", "AAGG"))
    assert not ends_compatible(End("overhang", "AAGG"), End("overhang", "TTCC"))
    # 相容关系对称
    assert ends_compatible(End("overhang", "CCTT"), End("overhang", "AAGG"))


# ---------------------------------------------------------------- 判定（纯函数）

def test_resolve_unique_solution_with_overhangs():
    res = resolve_ligation(_unique_example())
    assert res.fragment_count == 3
    assert res.verdict == "unique"
    assert res.solution_count == 1
    assert res.solution_reason

    rep = res.representative
    # 锚片段为名称最小的 insert-1，正向置于首位（旋转/整体反向规范化）
    assert [(p.fragment, p.orientation) for p in rep.placements] == [
        ("insert-1", "forward"),
        ("insert-2", "forward"),
        ("vector", "forward"),
    ]
    assert [p.position for p in rep.placements] == [1, 2, 3]
    # 3 个连接处均为反向互补，末位连接处闭环回首位
    assert len(rep.junctions) == 3
    assert all(j.compatibility == "overhang_reverse_complement" for j in rep.junctions)
    assert [(j.left_fragment, j.right_fragment) for j in rep.junctions] == [
        ("insert-1", "insert-2"),
        ("insert-2", "vector"),
        ("vector", "insert-1"),
    ]
    closing = rep.junctions[-1]
    assert closing.left_end.sequence == "AAGG"
    assert closing.right_end.sequence == "CCTT"
    assert all(j.description for j in rep.junctions)


def test_resolve_junction_evidence_matches_placement_ends():
    # 每个连接处的两侧末端必须等于对应位置片段在该放置方向下的右端/左端
    rep = resolve_ligation(_unique_example()).representative
    by_pos = {p.position: p for p in rep.placements}
    n = len(rep.placements)
    for j in rep.junctions:
        left_p = by_pos[j.junction_index]
        right_p = by_pos[j.junction_index % n + 1]
        assert j.left_fragment == left_p.fragment
        assert j.right_fragment == right_p.fragment
        assert j.left_end == left_p.right_end
        assert j.right_end == right_p.left_end


def test_resolve_no_solution_when_ends_never_match():
    # 平端片段与突出端片段之间永远不相容
    res = resolve_ligation(_req([
        _frag("A", _blunt(), _blunt()),
        _frag("B", _oh("AAGG"), _oh("CCTT")),
    ]))
    assert res.verdict == "no_solution"
    assert res.solution_count == 0
    assert res.representative is None
    assert res.solution_reason


def test_resolve_reverse_orientation_used():
    # B 必须反向放置：其右端 CCTT 才能与 A 右端 AAGG 反向互补
    res = resolve_ligation(_req([
        _frag("A", _blunt(), _oh("AAGG")),
        _frag("B", _blunt(), _oh("CCTT")),
    ]))
    assert res.verdict == "unique"
    assert [(p.fragment, p.orientation) for p in res.representative.placements] == [
        ("A", "forward"), ("B", "reverse"),
    ]
    j1, j2 = res.representative.junctions
    assert j1.compatibility == "overhang_reverse_complement"
    assert j1.left_end.sequence == "AAGG"
    assert j1.right_end.sequence == "CCTT"
    assert j2.compatibility == "blunt_blunt"


def test_resolve_single_blunt_fragment_self_circularizes():
    res = resolve_ligation(_req([_frag("only", _blunt(), _blunt())]))
    assert res.verdict == "unique"
    assert res.solution_count == 1
    assert len(res.representative.placements) == 1
    junction = res.representative.junctions[0]
    assert junction.junction_index == 1
    assert junction.left_fragment == "only"
    assert junction.right_fragment == "only"
    assert junction.compatibility == "blunt_blunt"


def test_resolve_single_fragment_overhang_self_ligation():
    # 单片段自连：右端与自身左端反向互补才可闭环
    ok = resolve_ligation(_req([_frag("x", _oh("AAGG"), _oh("CCTT"))]))
    assert ok.verdict == "unique"
    bad = resolve_ligation(_req([_frag("x", _oh("AAGG"), _oh("GGCC"))]))
    assert bad.verdict == "no_solution"
    assert bad.solution_count == 0


def test_resolve_all_blunt_counts_match_dihedral_classes():
    # 全平端时任意排列均可闭环：等价类数 = (n-1)! × 2^(n-1)
    assert resolve_ligation(_req([
        _frag("a", _blunt(), _blunt()),
    ])).solution_count == 1
    two = resolve_ligation(_req([
        _frag("a", _blunt(), _blunt()), _frag("b", _blunt(), _blunt()),
    ]))
    assert two.solution_count == 2
    assert two.verdict == "multiple"
    three = resolve_ligation(_req([
        _frag("a", _blunt(), _blunt()), _frag("b", _blunt(), _blunt()),
        _frag("c", _blunt(), _blunt()),
    ]))
    assert three.solution_count == 8
    four = resolve_ligation(_req([
        _frag(n, _blunt(), _blunt()) for n in ("a", "b", "c", "d")
    ]))
    assert four.solution_count == 48


def test_resolve_palindromic_overhangs_give_multiple():
    # 回文突出端 AATT 与自身相容：两片段正向/反向放置均可闭环 -> 2 个等价类
    res = resolve_ligation(_req([
        _frag("A", _oh("AATT"), _oh("AATT")),
        _frag("B", _oh("AATT"), _oh("AATT")),
    ]))
    assert res.solution_count == 2
    assert res.verdict == "multiple"


def test_resolve_representative_is_lexicographically_smallest():
    # 8 个等价类中字典序最小的规范化排列：锚片段首位正向，其余按名称与方向升序
    res = resolve_ligation(_req([
        _frag("c", _blunt(), _blunt()), _frag("a", _blunt(), _blunt()),
        _frag("b", _blunt(), _blunt()),
    ]))
    assert [(p.fragment, p.orientation) for p in res.representative.placements] == [
        ("a", "forward"), ("b", "forward"), ("c", "forward"),
    ]


def test_resolve_multiple_representative_is_valid_circle():
    res = resolve_ligation(_req([
        _frag(n, _blunt(), _blunt()) for n in ("a", "b", "c")
    ]))
    rep = res.representative
    assert len(rep.junctions) == 3
    assert all(j.compatibility == "blunt_blunt" for j in rep.junctions)
    # 连接处首尾相接成环：每个片段恰好提供一次右端与一次左端
    assert sorted(j.left_fragment for j in rep.junctions) == ["a", "b", "c"]
    assert sorted(j.right_fragment for j in rep.junctions) == ["a", "b", "c"]


def test_resolve_input_order_independent():
    first = resolve_ligation(_unique_example())
    shuffled = _req([
        _frag("insert-2", _oh("AGTC"), _oh("TTGC")),
        _frag("vector", _oh("GCAA"), _oh("AAGG")),
        _frag("insert-1", _oh("CCTT"), _oh("GACT")),
    ])
    assert resolve_ligation(shuffled) == first


def test_resolve_lowercase_sequences_normalized_to_uppercase():
    res = resolve_ligation(_req([
        _frag("A", _blunt(), _oh("aagg")),
        _frag("B", _blunt(), _oh("cctt")),
    ]))
    assert res.verdict == "unique"
    j1 = res.representative.junctions[0]
    assert j1.left_end.sequence == "AAGG"
    assert j1.right_end.sequence == "CCTT"


# ---------------------------------------------------------------- API 端到端

def _payload(**kw):
    payload = {
        "fragments": [
            {"name": "vector",
             "left_end": {"type": "overhang", "sequence": "GCAA"},
             "right_end": {"type": "overhang", "sequence": "AAGG"}},
            {"name": "insert-1",
             "left_end": {"type": "overhang", "sequence": "CCTT"},
             "right_end": {"type": "overhang", "sequence": "GACT"}},
            {"name": "insert-2",
             "left_end": {"type": "overhang", "sequence": "AGTC"},
             "right_end": {"type": "overhang", "sequence": "TTGC"}},
        ],
    }
    payload.update(kw)
    return payload


def test_api_ligation_unique_full_evidence():
    res = client.post("/api/v1/digest/ligation", json=_payload())
    assert res.status_code == 200
    body = res.json()

    assert body["fragment_count"] == 3
    assert body["verdict"] == "unique"
    assert body["solution_count"] == 1
    assert body["solution_reason"]

    rep = body["representative"]
    assert [(p["fragment"], p["orientation"]) for p in rep["placements"]] == [
        ("insert-1", "forward"), ("insert-2", "forward"), ("vector", "forward"),
    ]
    for p in rep["placements"]:
        assert p["left_end"]["type"] == "overhang"
        assert p["left_end"]["sequence"]
    assert len(rep["junctions"]) == 3
    for j in rep["junctions"]:
        assert j["compatibility"] == "overhang_reverse_complement"
        assert j["description"]
    # 闭环：末位连接处连接 vector 右端与 insert-1 左端
    closing = rep["junctions"][-1]
    assert closing["junction_index"] == 3
    assert closing["left_fragment"] == "vector"
    assert closing["right_fragment"] == "insert-1"
    assert closing["left_end"] == {"type": "overhang", "sequence": "AAGG"}
    assert closing["right_end"] == {"type": "overhang", "sequence": "CCTT"}


def test_api_ligation_no_solution():
    body = client.post("/api/v1/digest/ligation", json={
        "fragments": [
            {"name": "A", "left_end": {"type": "blunt"}, "right_end": {"type": "blunt"}},
            {"name": "B",
             "left_end": {"type": "overhang", "sequence": "AAGG"},
             "right_end": {"type": "overhang", "sequence": "CCTT"}},
        ],
    }).json()
    assert body["verdict"] == "no_solution"
    assert body["solution_count"] == 0
    assert body["representative"] is None
    assert body["solution_reason"]


def test_api_ligation_multiple_with_representative():
    body = client.post("/api/v1/digest/ligation", json={
        "fragments": [
            {"name": n, "left_end": {"type": "blunt"}, "right_end": {"type": "blunt"}}
            for n in ("a", "b", "c")
        ],
    }).json()
    assert body["verdict"] == "multiple"
    assert body["solution_count"] == 8
    rep = body["representative"]
    assert rep is not None
    assert len(rep["placements"]) == 3
    assert len(rep["junctions"]) == 3
    assert all(j["compatibility"] == "blunt_blunt" for j in rep["junctions"])
    assert all(j["left_end"] == {"type": "blunt", "sequence": None}
               for j in rep["junctions"])


def test_api_ligation_example_file():
    payload = json.loads(EXAMPLE_FILE.read_text(encoding="utf-8"))
    res = client.post("/api/v1/digest/ligation", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["verdict"] == "unique"
    assert body["solution_count"] == 1


# ---------------------------------------------------------------- 422 校验

def _expect_422(payload, code_part=None):
    res = client.post("/api/v1/digest/ligation", json=payload)
    assert res.status_code == 422
    body = res.json()
    assert body["code"] == "validation_failed"
    assert body["errors"]
    if code_part:
        assert any(code_part in e["type"] for e in body["errors"])
    return body


def test_422_duplicate_fragment_name():
    payload = _payload()
    payload["fragments"][1]["name"] = "vector"
    _expect_422(payload, "duplicate_fragment_name")


def test_422_blank_fragment_name():
    payload = _payload()
    payload["fragments"][0]["name"] = "  "
    _expect_422(payload, "invalid_fragment_name")


def test_422_too_many_fragments(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "MAX_LIGATION_FRAGMENTS", 2)
    _expect_422(_payload(), "too_many_ligation_fragments")


def test_422_missing_overhang_sequence():
    payload = _payload()
    payload["fragments"][0]["left_end"] = {"type": "overhang"}
    _expect_422(payload, "missing_overhang_sequence")


def test_422_blunt_end_with_sequence():
    payload = _payload()
    payload["fragments"][0]["left_end"] = {"type": "blunt", "sequence": "AATT"}
    _expect_422(payload, "blunt_end_with_sequence")


def test_422_invalid_overhang_sequence():
    payload = _payload()
    payload["fragments"][0]["left_end"] = {"type": "overhang", "sequence": "AXGT"}
    _expect_422(payload, "invalid_overhang_sequence")
    # 空序列由字段约束直接拦截
    payload["fragments"][0]["left_end"] = {"type": "overhang", "sequence": ""}
    _expect_422(payload)


def test_422_empty_fragments_and_bad_end_type():
    _expect_422({"fragments": []})
    payload = _payload()
    payload["fragments"][0]["left_end"] = {"type": "sticky", "sequence": "AATT"}
    _expect_422(payload)


def test_422_missing_required_fields():
    _expect_422({})
    _expect_422({"fragments": [{"name": "A", "left_end": {"type": "blunt"}}]})


def test_422_reports_multiple_errors_at_once():
    # 重名 + 突出端缺序列 + 平端带序列 + 非法序列字符，一次性报告
    payload = {
        "fragments": [
            {"name": "A",
             "left_end": {"type": "overhang"},
             "right_end": {"type": "blunt", "sequence": "TT"}},
            {"name": "A",
             "left_end": {"type": "blunt"},
             "right_end": {"type": "overhang", "sequence": "AX"}},
        ],
    }
    body = _expect_422(payload)
    types = {e["type"] for e in body["errors"]}
    assert "duplicate_fragment_name" in types
    assert "missing_overhang_sequence" in types
    assert "blunt_end_with_sequence" in types
    assert "invalid_overhang_sequence" in types
