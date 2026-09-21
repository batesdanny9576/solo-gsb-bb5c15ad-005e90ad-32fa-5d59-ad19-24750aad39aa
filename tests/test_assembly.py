"""环状 DNA 片段连接判定：回溯枚举、规范化去重与逐连接处依据的单元 + API 端到端测试。"""
from fastapi.testclient import TestClient

from app.assembly import (
    canonical_form,
    ends_compatible,
    reverse_complement,
    solve_assembly,
)
from app.main import app
from app.models import AssemblyRequest

client = TestClient(app)


def _frag(name, left, right):
    """left/right：'blunt' 表示平端，否则为 5' 突出端序列。"""
    def end(spec):
        if spec == "blunt":
            return {"kind": "blunt"}
        return {"kind": "overhang", "sequence": spec}

    return {"name": name, "left_end": end(left), "right_end": end(right)}


def _req(*fragments):
    return AssemblyRequest(fragments=list(fragments))


# 唯一解示例：alpha → beta → gamma →（环回 alpha）
ALPHA = _frag("alpha", "blunt", "AATT")
BETA = _frag("beta", "AATT", "ATGC")
GAMMA = _frag("gamma", "GCAT", "blunt")


# ---------------------------------------------------------------- 基础工具（纯函数）

def test_reverse_complement():
    assert reverse_complement("AATT") == "AATT"      # EcoRI 回文突出端
    assert reverse_complement("ATGC") == "GCAT"      # 非回文
    assert reverse_complement("GCAT") == "ATGC"      # 对合：互逆
    assert reverse_complement("AAAAC") == "GTTTT"
    assert reverse_complement("G") == "C"


def test_ends_compatible_rules():
    blunt = ("blunt", None)
    aatt = ("overhang", "AATT")
    atgc = ("overhang", "ATGC")
    gcat = ("overhang", "GCAT")
    # 平端仅与平端相容
    assert ends_compatible(blunt, blunt)
    assert not ends_compatible(blunt, aatt)
    assert not ends_compatible(aatt, blunt)
    # 突出端仅与反向互补序列相容（规则关于两末端对称）
    assert ends_compatible(aatt, aatt)
    assert ends_compatible(atgc, gcat)
    assert ends_compatible(gcat, atgc)
    assert not ends_compatible(aatt, atgc)
    assert not ends_compatible(gcat, aatt)


def test_canonical_form_merges_rotation_and_reversal():
    base = [("beta", "forward"), ("gamma", "reverse"), ("alpha", "forward")]
    canon = canonical_form(base)
    # 循环旋转等价
    assert canon == canonical_form(base[1:] + base[:1])
    assert canon == canonical_form(base[2:] + base[:2])
    # 整体反向（顺序反转且方向取反）等价
    reversed_cycle = [
        (name, "reverse" if o == "forward" else "forward")
        for name, o in reversed(base)
    ]
    assert canon == canonical_form(reversed_cycle)
    # 规范化代表以最小名片段开头且为正向
    assert canon[0] == ("alpha", 0)


# ---------------------------------------------------------------- 求解（纯函数）

def test_solve_unique_solution_with_full_junction_evidence():
    res = solve_assembly(_req(ALPHA, BETA, GAMMA))
    assert res.verdict == "unique"
    assert res.solution_count == 1
    assert res.fragment_count == 3

    sol = res.solution
    assert sol.signature == "alpha+ → beta+ → gamma+（环形）"
    assert [(p.position, p.name, p.orientation) for p in sol.placements] == [
        (1, "alpha", "forward"), (2, "beta", "forward"), (3, "gamma", "forward"),
    ]
    assert [p.upstream_end for p in sol.placements] == ["left", "left", "left"]
    assert [p.downstream_end for p in sol.placements] == ["right", "right", "right"]

    # 连接处数量等于片段数（含末位 → 首位的跨接连接）
    assert [j.junction_index for j in sol.junctions] == [1, 2, 3]
    j1, j2, j3 = sol.junctions
    assert (j1.upstream_fragment, j1.upstream_end) == ("alpha", "right")
    assert (j1.downstream_fragment, j1.downstream_end) == ("beta", "left")
    assert j1.compatibility == "overhang_reverse_complement"
    assert j1.upstream_overhang == "AATT" and j1.downstream_overhang == "AATT"
    assert j2.upstream_overhang == "ATGC" and j2.downstream_overhang == "GCAT"
    assert j2.compatibility == "overhang_reverse_complement"
    # 跨接连接：gamma 右末端（平端）→ alpha 左末端（平端）
    assert (j3.upstream_fragment, j3.upstream_end) == ("gamma", "right")
    assert (j3.downstream_fragment, j3.downstream_end) == ("alpha", "left")
    assert j3.compatibility == "blunt_blunt"
    assert j3.upstream_overhang is None and j3.downstream_overhang is None
    assert all(j.description for j in sol.junctions)
    assert res.verdict_reason


def test_solve_no_solution_when_ends_never_match():
    res = solve_assembly(_req(
        _frag("a", "blunt", "blunt"),
        _frag("b", "AATT", "GGCC"),
    ))
    assert res.verdict == "no_solution"
    assert res.solution_count == 0
    assert res.solution is None
    assert res.verdict_reason


def test_solve_multiple_solutions_all_blunt():
    # 4 个全平端片段：方向计入方案，共 (4-1)!·2^3 = 48 个不同方案
    res = solve_assembly(_req(*(_frag(n, "blunt", "blunt") for n in "abcd")))
    assert res.verdict == "multiple"
    assert res.solution_count == 48
    # 代表方案为字典序最小者
    assert res.solution.signature == "a+ → b+ → c+ → d+（环形）"
    assert len(res.solution.junctions) == 4
    assert all(j.compatibility == "blunt_blunt" for j in res.solution.junctions)


def test_solve_single_fragment_self_circularizes():
    res = solve_assembly(_req(_frag("solo", "AATT", "AATT")))
    assert res.verdict == "unique"
    assert res.solution_count == 1
    sol = res.solution
    assert len(sol.placements) == 1
    assert len(sol.junctions) == 1
    junction = sol.junctions[0]
    assert junction.upstream_fragment == junction.downstream_fragment == "solo"
    assert junction.upstream_end == "right" and junction.downstream_end == "left"
    assert junction.compatibility == "overhang_reverse_complement"


def test_solve_single_fragment_non_palindromic_overhangs():
    # 左 ATGC、右 GCAT：右末端 GCAT 的反向互补为 ATGC，恰与左末端相容
    res = solve_assembly(_req(_frag("solo", "ATGC", "GCAT")))
    assert res.verdict == "unique"
    assert res.solution_count == 1


def test_solve_single_fragment_incompatible_ends():
    assert solve_assembly(_req(_frag("solo", "blunt", "AATT"))).verdict == "no_solution"
    # 两端突出但非反向互补
    assert solve_assembly(_req(_frag("solo", "AATT", "GGCC"))).verdict == "no_solution"


def test_solve_fragment_orientation_flips_ends_not_sequence():
    # 片段反向放置时末端位置互换、突出序列不变：
    # beta 反向（右 AATT 朝上游）可接 alpha 右 AATT，但此时左 ATGC 朝下游，
    # 与 alpha 左平端不相容 -> 无解
    res = solve_assembly(_req(
        _frag("alpha", "blunt", "AATT"),
        _frag("beta", "ATGC", "AATT"),
    ))
    assert res.verdict == "no_solution"
    # 两端均为 AATT 的两个片段：beta 正/反向放置都相容，且两种放置不能经
    # 旋转/整体反向互相到达（整体反向会同时翻转两个片段的方向），
    # 方向是方案描述的一部分 -> 计为 2 个不同方案
    res2 = solve_assembly(_req(
        _frag("alpha", "AATT", "AATT"),
        _frag("beta", "AATT", "AATT"),
    ))
    assert res2.verdict == "multiple"
    assert res2.solution_count == 2
    assert res2.solution.signature == "alpha+ → beta+（环形）"


def test_solve_swapped_ends_instance_mirrors_solution():
    # 全部片段左右末端互换后的实例仍有唯一解（相容规则对称），
    # 代表方案为原唯一环的镜像方向
    res = solve_assembly(_req(
        _frag("alpha", "AATT", "blunt"),
        _frag("beta", "ATGC", "AATT"),
        _frag("gamma", "blunt", "GCAT"),
    ))
    assert res.verdict == "unique"
    assert res.solution.signature == "alpha+ → gamma+ → beta+（环形）"


def test_solve_input_order_independent():
    forward = solve_assembly(_req(ALPHA, BETA, GAMMA))
    shuffled = solve_assembly(_req(GAMMA, ALPHA, BETA))
    assert forward == shuffled


def test_solve_found_arrangements_are_canonical():
    # 搜索直接生成规范化形式：最小名片段在位置 0 且正向
    res = solve_assembly(_req(*(_frag(n, "blunt", "blunt") for n in "abcd")))
    placement = [
        (p.name, "forward" if p.orientation == "forward" else "reverse")
        for p in res.solution.placements
    ]
    assert canonical_form(placement) == tuple(
        (name, 0 if o == "forward" else 1) for name, o in placement
    )


# ---------------------------------------------------------------- API 端到端

def _payload(**kw):
    payload = {
        "fragments": [
            {"name": "alpha",
             "left_end": {"kind": "blunt"},
             "right_end": {"kind": "overhang", "sequence": "AATT"}},
            {"name": "beta",
             "left_end": {"kind": "overhang", "sequence": "AATT"},
             "right_end": {"kind": "overhang", "sequence": "ATGC"}},
            {"name": "gamma",
             "left_end": {"kind": "overhang", "sequence": "GCAT"},
             "right_end": {"kind": "blunt"}},
        ]
    }
    payload.update(kw)
    return payload


def test_api_unique_solution_full_evidence():
    res = client.post("/api/v1/digest/assembly", json=_payload())
    assert res.status_code == 200
    body = res.json()

    assert body["fragment_count"] == 3
    assert body["verdict"] == "unique"
    assert body["solution_count"] == 1
    assert body["verdict_reason"]

    sol = body["solution"]
    assert sol["signature"] == "alpha+ → beta+ → gamma+（环形）"
    assert [p["position"] for p in sol["placements"]] == [1, 2, 3]
    assert [p["orientation"] for p in sol["placements"]] == ["forward"] * 3

    assert [j["junction_index"] for j in sol["junctions"]] == [1, 2, 3]
    kinds = [j["compatibility"] for j in sol["junctions"]]
    assert kinds == [
        "overhang_reverse_complement",
        "overhang_reverse_complement",
        "blunt_blunt",
    ]
    for j in sol["junctions"]:
        assert j["description"]
        if j["compatibility"] == "overhang_reverse_complement":
            assert j["upstream_overhang"] and j["downstream_overhang"]
        else:
            assert j["upstream_overhang"] is None
            assert j["downstream_overhang"] is None


def test_api_multiple_solutions():
    body = client.post("/api/v1/digest/assembly", json={
        "fragments": [
            {"name": n, "left_end": {"kind": "blunt"}, "right_end": {"kind": "blunt"}}
            for n in ("a", "b", "c", "d")
        ],
    }).json()
    assert body["verdict"] == "multiple"
    assert body["solution_count"] == 48
    assert body["solution"]["signature"] == "a+ → b+ → c+ → d+（环形）"


def test_api_no_solution_returns_200_with_null_solution():
    res = client.post("/api/v1/digest/assembly", json={
        "fragments": [
            {"name": "a", "left_end": {"kind": "blunt"}, "right_end": {"kind": "blunt"}},
            {"name": "b",
             "left_end": {"kind": "overhang", "sequence": "AATT"},
             "right_end": {"kind": "overhang", "sequence": "GGCC"}},
        ],
    })
    assert res.status_code == 200
    body = res.json()
    assert body["verdict"] == "no_solution"
    assert body["solution_count"] == 0
    assert body["solution"] is None


def test_api_lowercase_overhang_normalized_to_uppercase():
    payload = _payload()
    payload["fragments"][0]["right_end"]["sequence"] = "aatt"
    res = client.post("/api/v1/digest/assembly", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["verdict"] == "unique"
    assert body["solution"]["junctions"][0]["upstream_overhang"] == "AATT"


def test_api_input_order_independent():
    first = client.post("/api/v1/digest/assembly", json=_payload()).json()
    payload = _payload()
    payload["fragments"] = list(reversed(payload["fragments"]))
    second = client.post("/api/v1/digest/assembly", json=payload).json()
    assert first == second


# ---------------------------------------------------------------- 422 校验

def _expect_422(payload, code_part=None):
    res = client.post("/api/v1/digest/assembly", json=payload)
    assert res.status_code == 422
    body = res.json()
    assert body["code"] == "validation_failed"
    assert body["errors"]
    if code_part:
        assert any(code_part in e["type"] for e in body["errors"])
    return body


def test_422_empty_fragments():
    _expect_422({"fragments": []})


def test_422_missing_end_or_invalid_kind():
    payload = _payload()
    del payload["fragments"][0]["left_end"]
    _expect_422(payload)
    payload = _payload()
    payload["fragments"][0]["left_end"] = {"kind": "sticky", "sequence": "AATT"}
    _expect_422(payload)


def test_422_overhang_without_sequence():
    payload = _payload()
    payload["fragments"][0]["right_end"] = {"kind": "overhang"}
    _expect_422(payload)


def test_422_overhang_sequence_invalid_characters():
    payload = _payload()
    payload["fragments"][0]["right_end"] = {"kind": "overhang", "sequence": "AXGT"}
    _expect_422(payload)
    payload = _payload()
    payload["fragments"][0]["right_end"] = {"kind": "overhang", "sequence": ""}
    _expect_422(payload)


def test_422_blunt_end_must_not_carry_sequence():
    payload = _payload()
    payload["fragments"][0]["left_end"] = {"kind": "blunt", "sequence": "AATT"}
    _expect_422(payload)


def test_422_duplicate_fragment_name():
    payload = _payload()
    payload["fragments"][1]["name"] = "alpha"
    _expect_422(payload, "duplicate_fragment_name")


def test_422_blank_fragment_name():
    payload = _payload()
    payload["fragments"][0]["name"] = "  "
    _expect_422(payload, "invalid_fragment_name")


def test_422_too_many_fragments(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "MAX_ASSEMBLY_FRAGMENTS", 2)
    _expect_422(_payload(), "too_many_assembly_fragments")


def test_422_reports_multiple_errors_at_once(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "MAX_ASSEMBLY_FRAGMENTS", 2)
    payload = _payload()
    payload["fragments"][1]["name"] = "alpha"  # 重名 + 片段数超限，一次性报告
    body = _expect_422(payload)
    types = {e["type"] for e in body["errors"]}
    assert "duplicate_fragment_name" in types
    assert "too_many_assembly_fragments" in types
