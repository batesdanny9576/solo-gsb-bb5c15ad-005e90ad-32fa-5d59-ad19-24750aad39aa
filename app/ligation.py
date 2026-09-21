"""核心算法：环状 DNA 片段连接判定（平端 / 5' 突出端相容 + 回溯枚举闭环排列）。

所有函数均为纯函数，不依赖任何外部服务或全局状态。

末端模型与相容规则
------------------
每个命名片段有左、右两个末端，各自为**平端**（blunt）或 **5' 突出端**
（overhang，给出突出链 5'→3' 序列）。连接处两侧末端相容当且仅当：

  * 两者均为平端（平端仅与平端相容）；
  * 两者均为 5' 突出端，且序列互为**反向互补**
    （如 AAGG 与 CCTT；回文序列如 AATT 与自身相容）。

平端与突出端之间不相容。片段**反向放置**时其左、右端互换位置；
突出端序列是物理末端的属性，不随放置方向改变。

枚举与规范化
------------
一个闭环方案是把全部片段各用一次排成一个圆环（每段可正可反），使全部
n 个连接处（含末位与首位之间的闭环连接处）均相容。回溯枚举时把
**名称最小的片段固定在首位且正向放置**：任一闭环排列的全体循环旋转与
整体反向（2n 个二面体变换，片段名互不相同故两两不同）中恰有一个满足
该约束，因此每个等价类（循环旋转与整体反向视为同一方案）恰好被枚举
一次，``solution_count`` 即为等价类的精确数量。代表方案取全部等价类
代表中字典序最小者，结果与输入提交顺序无关。
"""
from typing import NamedTuple

from app.models import (
    FragmentEndInput,
    LigationEndOut,
    LigationJunctionOut,
    LigationPlacementOut,
    LigationRequest,
    LigationResponse,
    LigationSolutionOut,
)

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    """DNA 序列的反向互补（输入须为大写 A/C/G/T）。"""
    return seq.translate(_COMPLEMENT)[::-1]


class End(NamedTuple):
    """规范化末端：kind 为 "blunt" / "overhang"，sequence 为大写序列或 None。"""

    kind: str
    sequence: str | None = None


def ends_compatible(right_end: End, left_end: End) -> bool:
    """连接处相容性：左侧片段的右端 vs 右侧片段的左端（该关系对称）。"""
    if right_end.kind == "blunt" or left_end.kind == "blunt":
        return right_end.kind == "blunt" and left_end.kind == "blunt"
    return left_end.sequence == reverse_complement(right_end.sequence)


def _to_end(end_input: FragmentEndInput) -> End:
    if end_input.type == "blunt":
        return End("blunt")
    return End("overhang", end_input.sequence.upper())


def _end_out(end: End) -> LigationEndOut:
    return LigationEndOut(type=end.kind, sequence=end.sequence)


def _build_solution(fragments, oriented, placement) -> LigationSolutionOut:
    """把锚定枚举得到的代表排列展开为放置列表与逐连接处相容依据。"""
    placements: list[LigationPlacementOut] = []
    for pos, (idx, ori) in enumerate(placement, start=1):
        left_end, right_end = oriented[idx][ori]
        placements.append(LigationPlacementOut(
            position=pos,
            fragment=fragments[idx][0],
            orientation="forward" if ori == 0 else "reverse",
            left_end=_end_out(left_end),
            right_end=_end_out(right_end),
        ))

    junctions: list[LigationJunctionOut] = []
    m = len(placement)
    for pos in range(m):
        left_idx, left_ori = placement[pos]
        right_idx, right_ori = placement[(pos + 1) % m]
        left_name = fragments[left_idx][0]
        right_name = fragments[right_idx][0]
        # 连接处 = 左侧片段的右端 与 右侧片段的左端（末位回接首位，闭环）
        left_right = oriented[left_idx][left_ori][1]
        right_left = oriented[right_idx][right_ori][0]
        j = pos + 1
        if left_right.kind == "blunt":
            compatibility = "blunt_blunt"
            description = (
                f"连接处 {j}：片段 {left_name} 的右端与片段 {right_name} 的左端"
                "均为平端，平端仅与平端相容，可连接"
            )
        else:
            compatibility = "overhang_reverse_complement"
            description = (
                f"连接处 {j}：片段 {left_name} 右端 5' 突出端 "
                f"{left_right.sequence}（5'→3'）与片段 {right_name} 左端 5' 突出端 "
                f"{right_left.sequence}（5'→3'）互为反向互补（{left_right.sequence} "
                f"的反向互补为 {reverse_complement(left_right.sequence)}），可连接"
            )
        junctions.append(LigationJunctionOut(
            junction_index=j,
            left_fragment=left_name,
            right_fragment=right_name,
            left_end=_end_out(left_right),
            right_end=_end_out(right_left),
            compatibility=compatibility,
            description=description,
        ))
    return LigationSolutionOut(placements=placements, junctions=junctions)


def resolve_ligation(req: LigationRequest) -> LigationResponse:
    """回溯枚举全部可闭环排列（每个等价类恰好一次），判定无解 / 唯一 / 多解。"""
    fragments = [
        (f.name, _to_end(f.left_end), _to_end(f.right_end))
        for f in req.fragments
    ]
    n = len(fragments)
    anchor = min(range(n), key=lambda i: fragments[i][0])
    anchor_name, anchor_left, _ = fragments[anchor]

    # 每个片段两种放置方向的 (左端, 右端)：0 = 正向，1 = 反向（左右端互换）
    oriented = [
        ((left, right), (right, left))
        for _, left, right in fragments
    ]
    # 回溯按片段名字典序进行，结果与提交顺序无关
    others = sorted((i for i in range(n) if i != anchor), key=lambda i: fragments[i][0])

    count = 0
    best_key: tuple | None = None
    best_placement: list[tuple[int, int]] = []
    placed: list[tuple[int, int]] = [(anchor, 0)]
    used = [False] * n
    used[anchor] = True

    def backtrack() -> None:
        nonlocal count, best_key, best_placement
        if len(placed) == n:
            last_idx, last_ori = placed[-1]
            last_right = oriented[last_idx][last_ori][1]
            # 闭环连接处：末位片段右端 vs 锚片段左端
            if not ends_compatible(last_right, anchor_left):
                return
            count += 1
            key = tuple(
                (fragments[i][0], "forward" if o == 0 else "reverse")
                for i, o in placed
            )
            if best_key is None or key < best_key:
                best_key = key
                best_placement = list(placed)
            return
        prev_idx, prev_ori = placed[-1]
        prev_right = oriented[prev_idx][prev_ori][1]
        for i in others:
            if used[i]:
                continue
            for o in (0, 1):
                left_end, _ = oriented[i][o]
                if not ends_compatible(prev_right, left_end):
                    continue
                used[i] = True
                placed.append((i, o))
                backtrack()
                placed.pop()
                used[i] = False

    backtrack()

    if count == 0:
        verdict = "no_solution"
    elif count == 1:
        verdict = "unique"
    else:
        verdict = "multiple"

    representative = (
        _build_solution(fragments, oriented, best_placement) if best_placement else None
    )

    search_note = (
        f"回溯枚举 {n} 个片段的全部正反向圆环排列（名称最小片段 {anchor_name} "
        "固定于首位且正向放置，循环旋转与整体反向的排列视为同一方案）"
    )
    if count == 0:
        reason = search_note + "，未发现任何全部连接处均相容的闭环排列，判定无解。"
    elif count == 1:
        reason = search_note + "，恰有 1 个等价类可闭环，判定唯一解。"
    else:
        reason = (
            search_note + f"，共找到 {count} 个互不等价的可闭环方案，判定多解；"
            "代表方案为全部等价类中字典序最小的规范化排列。"
        )

    return LigationResponse(
        fragment_count=n,
        verdict=verdict,
        solution_count=count,
        representative=representative,
        solution_reason=reason,
    )
