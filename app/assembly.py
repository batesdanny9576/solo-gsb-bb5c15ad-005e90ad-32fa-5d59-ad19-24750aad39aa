"""核心算法：环状 DNA 片段连接（组装）判定。

所有函数均为纯函数，不依赖任何外部服务或全局状态。

末端模型
--------
每个片段有左、右两个物理末端；末端要么是平端（blunt），要么带 5' 突出端
（overhang）。突出端序列约定为**突出单链自身 5'→3'** 的序列，是物理末端的
固有属性：片段反向放置时左、右末端位置互换，但各末端的突出序列不变。

连接相容规则
------------
相邻两片段的连接处，上游片段的下游侧末端与下游片段的上游侧末端相接：

  * 平端仅与平端相容；
  * 5' 突出端仅与反向互补的 5' 突出端相容：下游末端序列 == 上游末端序列的
    反向互补（两条突出单链反向平行退火）；
  * 平端与突出端不相容。

反向互补是对合运算，因此该规则关于两末端对称：整体反向的排列相容性不变。

排列等价与规范化
----------------
一个排列是环上有向的片段放置序列（每片段恰用一次，允许正反向放置）。
循环旋转（更换起点）与整体反向（顺序反转且每个片段方向取反）描述同一个
物理圆环，视为同一方案。规范化代表取全部旋转与整体反向的旋转中
（片段名, 方向）元组序列的字典序最小者（forward 记 0、reverse 记 1）。
由于片段名唯一，字典序最小的排列必然以“最小名片段 + 正向”开头——
整体反向可把该片段翻为正向，而正向元组字典序更小——因此规范化代表
唯一确定。

回溯搜索
--------
把片段名字典序最小的片段**正向**固定在位置 0：每个旋转/整体反向等价类
恰好包含一个这样的排列，因此搜索天然地对每个方案只生成一次，无需
去重集合。其余位置按片段名字典序尝试剩余片段的两种方向，遇到不相容
连接处立即剪枝；放满后额外检查末位与首位的跨接连接。全部规范化方案
计数后判定无解（0 个）、唯一（1 个）或多解（≥ 2 个），并返回字典序
最小的代表方案及其逐连接处相容依据。
"""
from app.models import (
    AssemblyJunctionOut,
    AssemblyPlacementOut,
    AssemblyRequest,
    AssemblyResponse,
    AssemblySolutionOut,
)

_COMPLEMENT = str.maketrans("ACGT", "TGCA")

# 末端表示：(kind, sequence)；kind ∈ {"blunt", "overhang"}，平端的 sequence 为 None
End = tuple[str, str | None]

_END_SIDE_CN = {"left": "左", "right": "右"}


def reverse_complement(sequence: str) -> str:
    """DNA 序列的反向互补（序列仅含 A/C/G/T）。"""
    return sequence.translate(_COMPLEMENT)[::-1]


def ends_compatible(upstream: End, downstream: End) -> bool:
    """连接处相容判定：平端仅与平端；5' 突出端仅与反向互补的突出端。"""
    up_kind, up_seq = upstream
    down_kind, down_seq = downstream
    if up_kind == "blunt" or down_kind == "blunt":
        return up_kind == "blunt" and down_kind == "blunt"
    return down_seq == reverse_complement(up_seq)


def oriented_ends(left: End, right: End, orientation: str) -> tuple[End, End]:
    """按放置方向返回（上游侧末端, 下游侧末端）：正向为（左, 右），反向为（右, 左）。"""
    return (left, right) if orientation == "forward" else (right, left)


def canonical_form(placement: list[tuple[str, str]]) -> tuple[tuple[str, int], ...]:
    """排列的规范化代表：全部旋转与整体反向（顺序反转且方向取反）中的字典序最小者。

    方向编码 forward 为 0、reverse 为 1。片段名唯一时结果确定；字典序最小者
    必然以（最小名片段, 正向）开头。
    """
    seq = [(name, 0 if orientation == "forward" else 1) for name, orientation in placement]
    rev = [(name, 1 - flag) for name, flag in reversed(seq)]
    best: tuple[tuple[str, int], ...] | None = None
    for variant in (seq, rev):
        for k in range(len(variant)):
            candidate = tuple(variant[k:]) + tuple(variant[:k])
            if best is None or candidate < best:
                best = candidate
    return best


def _physical_end(fragment_ends: tuple[End, End], side: str) -> End:
    """取片段的某一物理末端：fragment_ends 按（左, 右）存放。"""
    return fragment_ends[0] if side == "left" else fragment_ends[1]


def _build_solution(
    canonical: tuple[tuple[str, int], ...],
    ends: dict[str, tuple[End, End]],
) -> AssemblySolutionOut:
    """由规范化排列构造代表方案：逐位置放置与逐连接处相容依据。"""
    n = len(canonical)
    placements: list[AssemblyPlacementOut] = []
    for index, (name, flag) in enumerate(canonical):
        orientation = "forward" if flag == 0 else "reverse"
        upstream_end, downstream_end = (
            ("left", "right") if orientation == "forward" else ("right", "left")
        )
        placements.append(AssemblyPlacementOut(
            position=index + 1,
            name=name,
            orientation=orientation,
            upstream_end=upstream_end,
            downstream_end=downstream_end,
        ))

    junctions: list[AssemblyJunctionOut] = []
    for index, placement in enumerate(placements):
        nxt = placements[(index + 1) % n]
        up_kind, up_seq = _physical_end(ends[placement.name], placement.downstream_end)
        down_kind, down_seq = _physical_end(ends[nxt.name], nxt.upstream_end)
        j = index + 1
        if up_kind == "blunt":
            compatibility = "blunt_blunt"
            description = (
                f"连接处 {j}：片段 {placement.name} 的"
                f"{_END_SIDE_CN[placement.downstream_end]}末端（平端）与片段 {nxt.name} 的"
                f"{_END_SIDE_CN[nxt.upstream_end]}末端（平端）相接；"
                "平端仅与平端相容，两侧均为平端，可连接"
            )
        else:
            compatibility = "overhang_reverse_complement"
            description = (
                f"连接处 {j}：片段 {placement.name} 的"
                f"{_END_SIDE_CN[placement.downstream_end]}末端（5' 突出端 {up_seq}）与片段 "
                f"{nxt.name} 的{_END_SIDE_CN[nxt.upstream_end]}末端（5' 突出端 {down_seq}）相接；"
                f"{up_seq} 的反向互补为 {reverse_complement(up_seq)}，"
                "两突出端序列互为反向互补，可退火连接"
            )
        junctions.append(AssemblyJunctionOut(
            junction_index=j,
            upstream_fragment=placement.name,
            upstream_end=placement.downstream_end,
            upstream_end_kind=up_kind,
            upstream_overhang=up_seq,
            downstream_fragment=nxt.name,
            downstream_end=nxt.upstream_end,
            downstream_end_kind=down_kind,
            downstream_overhang=down_seq,
            compatibility=compatibility,
            description=description,
        ))

    signature = " → ".join(
        f"{name}{'+' if flag == 0 else '-'}" for name, flag in canonical
    ) + "（环形）"
    return AssemblySolutionOut(
        signature=signature, placements=placements, junctions=junctions,
    )


def solve_assembly(req: AssemblyRequest) -> AssemblyResponse:
    """回溯枚举全部可闭环排列（按旋转/整体反向去重），判定无解、唯一或多解。"""
    fragments = sorted(req.fragments, key=lambda f: f.name)
    names = [f.name for f in fragments]
    ends: dict[str, tuple[End, End]] = {
        f.name: (
            (f.left_end.kind, f.left_end.sequence),
            (f.right_end.kind, f.right_end.sequence),
        )
        for f in fragments
    }
    n = len(names)
    first = names[0]

    placement: list[tuple[str, str]] = []
    solution_count = 0
    best: tuple[tuple[str, int], ...] | None = None

    def backtrack(position: int, remaining: tuple[str, ...], previous_downstream: End) -> None:
        nonlocal solution_count, best
        if position == n:
            # 跨接连接：末位片段的下游侧末端 → 首位片段（正向的最小名片段）的上游侧末端
            first_upstream, _ = oriented_ends(*ends[first], "forward")
            if ends_compatible(previous_downstream, first_upstream):
                solution_count += 1
                form = tuple(
                    (name, 0 if orientation == "forward" else 1)
                    for name, orientation in placement
                )
                if best is None or form < best:
                    best = form
            return
        for i, name in enumerate(remaining):
            for orientation in ("forward", "reverse"):
                upstream, downstream = oriented_ends(*ends[name], orientation)
                if not ends_compatible(previous_downstream, upstream):
                    continue
                placement.append((name, orientation))
                backtrack(position + 1, remaining[:i] + remaining[i + 1:], downstream)
                placement.pop()

    # 最小名片段正向固定于位置 0：每个旋转/整体反向等价类恰被生成一次
    placement.append((first, "forward"))
    _, first_downstream = oriented_ends(*ends[first], "forward")
    backtrack(1, tuple(names[1:]), first_downstream)
    placement.pop()

    if solution_count == 0:
        verdict = "no_solution"
        solution = None
        reason = (
            f"已回溯枚举全部 {n} 个片段的排列与正反向放置（片段名最小的 {first} "
            "正向固定于位置 0，同时消除旋转与整体反向对称），未发现任何所有连接处"
            "均相容（平端-平端，或 5' 突出端互为反向互补）的闭环排列：无解"
        )
    else:
        solution = _build_solution(best, ends)
        if solution_count == 1:
            verdict = "unique"
            reason = (
                "在全部排列与正反向放置的回溯搜索中（最小名片段正向固定于位置 0，"
                "每个旋转/整体反向等价类恰计一次），恰有 1 个所有连接处均相容的"
                "规范化闭环排列：唯一解；solution 给出逐片段方向与逐连接处相容依据"
            )
        else:
            verdict = "multiple"
            reason = (
                "在全部排列与正反向放置的回溯搜索中（最小名片段正向固定于位置 0，"
                "每个旋转/整体反向等价类恰计一次），共发现 "
                f"{solution_count} 个不同的规范化闭环排列（循环旋转与整体反向视为"
                "同一方案）：多解；solution 为字典序最小的规范化代表方案"
            )

    return AssemblyResponse(
        fragment_count=n,
        solution_count=solution_count,
        verdict=verdict,
        solution=solution,
        verdict_reason=reason,
    )
