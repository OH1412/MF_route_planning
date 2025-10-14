import heapq
import matplotlib.pyplot as plt
import random
import time  # 导入时间模块用于耗时检测
from typing import Dict, Tuple, List, Optional


# ---------------------------
# 基本（内部）坐标/邻居
# ---------------------------
def node_to_coord_inner(node: int) -> Tuple[int, int]:
    node0 = node - 1
    row = node0 // 3
    col = node0 % 3
    return (row, col)


def coord_to_node_inner(row: int, col: int) -> int:
    return row * 3 + col + 1


def neighbors_inner(node: int) -> List[int]:
    r, c = node_to_coord_inner(node)
    nbrs = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 4 and 0 <= nc < 3:
            nbrs.append(coord_to_node_inner(nr, nc))
    return nbrs


def manhattan_inner(a: int, b: int) -> int:
    ra, ca = node_to_coord_inner(a)
    rb, cb = node_to_coord_inner(b)
    return abs(ra - rb) + abs(ca - cb)


# ---------------------------
# 内->外层坐标映射（外围 +1）
# ---------------------------
def inner_to_outer_coord(node: int) -> Tuple[int, int]:
    r_in, c_in = node_to_coord_inner(node)
    return (r_in + 1, c_in + 1)


def outer_to_inner_node(r_out: int, c_out: int) -> Optional[int]:
    r_in = r_out - 1
    c_in = c_out - 1
    if 0 <= r_in < 4 and 0 <= c_in < 3:
        return coord_to_node_inner(r_in, c_in)
    return None


OUT_ROWS = 6
OUT_COLS = 5


# ---------------------------
# 粉色跑道（外围环）索引/路径辅助
# ---------------------------
def build_pink_ring_positions() -> List[Tuple[int, int]]:
    ring = []
    for c in range(0, OUT_COLS):
        ring.append((0, c))
    for r in range(1, OUT_ROWS):
        ring.append((r, OUT_COLS - 1))
    for c in range(OUT_COLS - 2, -1, -1):
        ring.append((OUT_ROWS - 1, c))
    for r in range(OUT_ROWS - 2, 0, -1):
        ring.append((r, 0))
    return ring


PINK_RING = build_pink_ring_positions()
PINK_INDEX = {pos: i for i, pos in enumerate(PINK_RING)}
PINK_COUNT = len(PINK_RING)


def pink_ring_distance(a_idx: int, b_idx: int) -> int:
    d = abs(a_idx - b_idx)
    return min(d, PINK_COUNT - d)


def ring_shortest_path_indices(a_idx: int, b_idx: int) -> List[int]:
    """返回从 a 到 b 的沿环最短路径索引序列（含 a->...->b），如果相等返回 [a]."""
    if a_idx == b_idx:
        return [a_idx]
    d = (b_idx - a_idx) % PINK_COUNT
    if d <= PINK_COUNT - d:
        # forward direction +1
        path = []
        cur = a_idx
        while True:
            path.append(cur)
            if cur == b_idx:
                break
            cur = (cur + 1) % PINK_COUNT
        return path
    else:
        # backward direction -1
        path = []
        cur = a_idx
        while True:
            path.append(cur)
            if cur == b_idx:
                break
            cur = (cur - 1) % PINK_COUNT
        return path


def pink_positions_adjacent_to_inner(node: int) -> List[int]:
    r_out, c_out = inner_to_outer_coord(node)
    adj = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r_out + dr, c_out + dc
        if 0 <= rr < OUT_ROWS and 0 <= cc < OUT_COLS:
            if (rr, cc) in PINK_INDEX:
                adj.append(PINK_INDEX[(rr, cc)])
    return adj


# ---------------------------
# 垂直代价与高度下界（内部网格）
# ---------------------------
def vertical_time_exact(a: int, b: int, heights_map: Dict[int, int], times: Dict[str, float]) -> float:
    ha = heights_map[a]
    hb = heights_map[b]
    if ha == hb:
        return 0.0
    if ha < hb:
        if (ha, hb) in ((200, 400), (400, 600)):
            return times["time_up_low"]
        elif (ha, hb) == (200, 600):
            return times["time_up_high"]
    else:
        if (ha, hb) in ((600, 400), (400, 200)):
            return times["time_down_low"]
        elif (ha, hb) == (600, 200):
            return times["time_down_high"]
    raise ValueError(f"Unhandled height change {ha}->{hb}")


def build_height_min_cost(times: Dict[str, float]) -> Dict[Tuple[int, int], float]:
    heights = [200, 400, 600]
    edges = {}
    for h in heights:
        edges[(h, h)] = 0.0
    edges[(200, 400)] = times["time_up_low"]
    edges[(400, 200)] = times["time_down_low"]
    edges[(400, 600)] = times["time_up_low"]
    edges[(600, 400)] = times["time_down_low"]
    edges[(200, 600)] = times["time_up_high"]
    edges[(600, 200)] = times["time_down_high"]

    mincost = {}
    for h_start in heights:
        dist = {h: float('inf') for h in heights}
        dist[h_start] = 0.0
        pq = [(0.0, h_start)]
        while pq:
            d, h = heapq.heappop(pq)
            if d > dist[h]:
                continue
            for h2 in heights:
                if h == h2:
                    nd = d
                elif (h, h2) in edges:
                    nd = d + edges[(h, h2)]
                else:
                    continue
                if nd < dist[h2]:
                    dist[h2] = nd
                    heapq.heappush(pq, (nd, h2))
        for h_end in heights:
            mincost[(h_start, h_end)] = dist[h_end]
    return mincost


def ascent_from_zero_time(height: int, times: Dict[str, float]) -> float:
    if height == 200:
        return times["time_up_low"]
    elif height == 400:
        return times["time_up_high"]
    elif height == 600:
        return times.get("time_up_to_600", times["time_up_high"])
    else:
        raise ValueError(f"未知高度用于从0抬升: {height}")


def descent_to_zero_time(height: int, times: Dict[str, float]) -> float:
    if height == 200:
        return times["time_down_low"]
    elif height == 400:
        return times["time_down_high"]
    elif height == 600:
        return times["time_down_to_zero"]
    else:
        raise ValueError(f"未知高度用于降到0: {height}")


# ---------------------------
# A* 主逻辑（包含外部跑道机器人）
# 返回： best_time, node_path, stats, kfs1_pick_events_ordered, ext_path_segments
# ext_path_segments: list of (order, kfs1_node, from_idx, to_idx, arrival_time, completion_time)
# ---------------------------
def a_star_with_external_ring_kfs1(
        heights_map: Dict[int, int],
        times: Dict[str, float],
        ext_move_time: float,
        ext_pick_kfs1_time_default: float,
        kfs_locked: int,
        kfs1_set: set,
        time_pick_kfs1_map: Dict[int, float],
        kfs2_positions: List[int],
        entry_nodes=(1, 2, 3),
        exit_nodes=(10, 11, 12),
        min_kfs2_picked_required: int = 1
):
    # 记录A*算法开始时间
    start_time = time.time()

    time_pick_kfs2 = times["time_pick_kfs2"]
    time_clean_kfs2 = times["time_clean_kfs2"]
    w_time = times.get("w_time", 1.0)
    w_pick = times.get("w_pick", 1.0)

    kfs1_list = sorted(list(kfs1_set))
    kfs1_index = {p: i for i, p in enumerate(kfs1_list)}
    total_kfs1 = len(kfs1_list)

    kfs2_index = {p: i for i, p in enumerate(kfs2_positions)}
    total_kfs2 = len(kfs2_positions)

    def kfs2_is_handled(p_mask: int, c_mask: int, pos: int) -> bool:
        if pos not in kfs2_index:
            return True
        idx = kfs2_index[pos]
        return ((p_mask >> idx) & 1) == 1 or ((c_mask >> idx) & 1) == 1

    def is_passable(node: int, k1_mask: int, p_mask: int, c_mask: int, robot_time: float,
                    k1_completion_times: Tuple[float]):
        if node == kfs_locked:
            return False
        if node in kfs1_index:
            idx = kfs1_index[node]
            if ((k1_mask >> idx) & 1) == 0:
                return False
            if k1_completion_times[idx] > robot_time + 1e-9:
                return False
        if node in kfs2_index and not kfs2_is_handled(p_mask, c_mask, node):
            return False
        return True

    h_min = build_height_min_cost(times)

    def heuristic(node: int, k1_mask: int, p_mask: int, c_mask: int) -> float:
        cur_h = heights_map[node]
        best_t = float('inf')
        for ex in exit_nodes:
            if ex == kfs_locked:
                continue
            ex_h = heights_map[ex]
            t_lower = manhattan_inner(node, ex) * times["time_approach"] + h_min[(cur_h, ex_h)]
            if t_lower < best_t:
                best_t = t_lower
        if best_t == float('inf'):
            best_t = 0.0
        picked_so_far = 0
        return w_time * best_t - w_pick * (picked_so_far + (total_kfs2 - picked_so_far))

    INF = float('inf')
    g_obj = {}
    g_time = {}
    g_picked = {}
    came_from = {}
    open_pq = []
    best_goal = None

    ext_start_idx = PINK_INDEX[(0, 0)]

    # 初始化入口状态
    for s in entry_nodes:
        if s == kfs_locked:
            continue
        ascent = ascent_from_zero_time(heights_map[s], times)
        approach = times["time_approach"]
        ext_time0 = 0.0
        k1_comp0 = tuple([0.0] * total_kfs1)

        if s in kfs1_set:
            idx_k1 = kfs1_index[s]
            candidates = pink_positions_adjacent_to_inner(s)
            if not candidates:
                continue
            # 1. 计算外部机器人拾取该KFS1的完成时间
            best_cand = None
            best_steps = None
            for cand in candidates:
                steps = pink_ring_distance(ext_start_idx, cand)
                if best_cand is None or steps < best_steps:
                    best_cand = cand
                    best_steps = steps
            move_time = best_steps * ext_move_time
            dur = time_pick_kfs1_map.get(s, ext_pick_kfs1_time_default)
            completion = ext_time0 + move_time + dur  # 外部拾取完成时间
            ext_time1 = completion
            ext_pos1 = best_cand
            # 2. 标记该KFS1已安排，并记录完成时间
            k1_mask0 = (1 << idx_k1)
            k1_comp_list = list(k1_comp0)
            k1_comp_list[idx_k1] = completion
            k1_comp0 = tuple(k1_comp_list)
            p_mask0 = 0
            c_mask0 = 0
            options_masks = [(p_mask0, c_mask0, 0.0)]
            if s in kfs2_index:
                bit = 1 << kfs2_index[s]
                options_masks.append((p_mask0 | bit, c_mask0 | bit, time_pick_kfs2))
                options_masks.append((p_mask0, c_mask0 | bit, time_clean_kfs2))
            # 3. 初始化内部机器人状态（关键修复：外部完成后才开始内部动作）
            for (p_mask, c_mask, cost_k2) in options_masks:
                # 修复核心：外部拾取完成后，才开始计算内部的抬升+接近+KFS2处理时间
                robot_time0 = completion + ascent + approach + cost_k2
                picked_cnt = bin(p_mask).count("1")
                obj = w_time * robot_time0 - w_pick * picked_cnt
                state = (s, k1_mask0, p_mask, c_mask, round(ext_time1, 3), k1_comp0, ext_pos1)
                if state not in g_obj or obj < g_obj[state]:
                    g_obj[state] = obj
                    g_time[state] = robot_time0
                    g_picked[state] = picked_cnt
                    heapq.heappush(open_pq,
                                   (obj + heuristic(s, k1_mask0, p_mask, c_mask), obj, robot_time0, picked_cnt, state))

        else:
            k1_mask0 = 0;
            p_mask0 = 0;
            c_mask0 = 0
            options_masks = [(p_mask0, c_mask0, 0.0)]
            if s in kfs2_index:
                bit = 1 << kfs2_index[s]
                options_masks.append((p_mask0 | bit, c_mask0 | bit, time_pick_kfs2))
                options_masks.append((p_mask0, c_mask0 | bit, time_clean_kfs2))
            for (p_mask, c_mask, cost_k2) in options_masks:
                robot_time0 = ascent + approach + cost_k2
                obj = w_time * robot_time0 - w_pick * bin(p_mask).count("1")
                state = (s, k1_mask0, p_mask, c_mask, round(ext_time0, 3), k1_comp0, ext_start_idx)
                if state not in g_obj or obj < g_obj[state]:
                    g_obj[state] = obj;
                    g_time[state] = robot_time0;
                    g_picked[state] = bin(p_mask).count("1")
                    heapq.heappush(open_pq,
                                   (obj + heuristic(s, k1_mask0, p_mask, c_mask), obj, robot_time0, g_picked[state],
                                    state))

    if not open_pq:
        raise RuntimeError("入口被阻塞，无可用起始状态")

    # 搜索
    while open_pq:
        f, obj_g, robot_time_g, picked_g_cnt, state = heapq.heappop(open_pq)
        node, k1_mask, p_mask, c_mask, ext_time, k1_comp_tuple, ext_pos_idx = state

        if best_goal is not None:
            best_obj_val = best_goal[0]
            if f >= best_obj_val - 1e-9:
                break

        if g_obj.get(state, INF) < obj_g - 1e-12:
            continue

        # 外部安排
        for k1_node in kfs1_list:
            idx = kfs1_index[k1_node]
            if ((k1_mask >> idx) & 1) == 0:
                candidates = pink_positions_adjacent_to_inner(k1_node)
                if not candidates:
                    continue
                dur = time_pick_kfs1_map.get(k1_node, ext_pick_kfs1_time_default)
                for cand in candidates:
                    steps = pink_ring_distance(ext_pos_idx, cand)
                    move_time = steps * ext_move_time
                    completion = ext_time + move_time + dur
                    new_ext_time = completion
                    new_ext_pos = cand
                    new_k1_mask = k1_mask | (1 << idx)
                    k1_comp_list = list(k1_comp_tuple);
                    k1_comp_list[idx] = completion
                    new_k1_comp_tuple = tuple(k1_comp_list)
                    new_robot_time = robot_time_g
                    new_p_mask = p_mask;
                    new_c_mask = c_mask;
                    new_picked = picked_g_cnt
                    new_obj = w_time * new_robot_time - w_pick * new_picked
                    new_state = (node, new_k1_mask, new_p_mask, new_c_mask, round(new_ext_time, 3), new_k1_comp_tuple,
                                 new_ext_pos)
                    if new_obj + 1e-12 < g_obj.get(new_state, INF):
                        g_obj[new_state] = new_obj;
                        g_time[new_state] = new_robot_time;
                        g_picked[new_state] = new_picked
                        came_from[new_state] = (state,
                                                ("pick_kfs1_ext", k1_node, dur, ext_time, new_ext_time, ext_pos_idx,
                                                 new_ext_pos))
                        heapq.heappush(open_pq,
                                       (new_obj + heuristic(node, new_k1_mask, new_p_mask, new_c_mask), new_obj,
                                        new_robot_time, new_picked, new_state))

        # pick/clean kfs2
        if node in exit_nodes:
            check_nodes = neighbors_inner(node) + [node]
        else:
            check_nodes = neighbors_inner(node)
        for v in check_nodes:
            if v in kfs2_index and not kfs2_is_handled(p_mask, c_mask, v):
                idx2 = kfs2_index[v];
                bit2 = 1 << idx2
                # pick
                new_p_mask = p_mask | bit2;
                new_c_mask = c_mask | bit2
                new_robot_time = robot_time_g + time_pick_kfs2;
                new_picked = picked_g_cnt + 1
                new_obj = w_time * new_robot_time - w_pick * new_picked
                new_state = (node, k1_mask, new_p_mask, new_c_mask, ext_time, k1_comp_tuple, ext_pos_idx)
                if new_obj + 1e-12 < g_obj.get(new_state, INF):
                    g_obj[new_state] = new_obj;
                    g_time[new_state] = new_robot_time;
                    g_picked[new_state] = new_picked
                    came_from[new_state] = (state, ("pick_kfs2", v, time_pick_kfs2, ext_time, ext_time))
                    heapq.heappush(open_pq,
                                   (new_obj + heuristic(node, k1_mask, new_p_mask, new_c_mask), new_obj, new_robot_time,
                                    new_picked, new_state))
                # clean
                new_p_mask2 = p_mask;
                new_c_mask2 = c_mask | bit2
                new_robot_time2 = robot_time_g + time_clean_kfs2;
                new_picked2 = picked_g_cnt
                new_obj2 = w_time * new_robot_time2 - w_pick * new_picked2
                new_state2 = (node, k1_mask, new_p_mask2, new_c_mask2, ext_time, k1_comp_tuple, ext_pos_idx)
                if new_obj2 + 1e-12 < g_obj.get(new_state2, INF):
                    g_obj[new_state2] = new_obj2;
                    g_time[new_state2] = new_robot_time2;
                    g_picked[new_state2] = new_picked2
                    came_from[new_state2] = (state, ("clean_kfs2", v, time_clean_kfs2, ext_time, ext_time))
                    heapq.heappush(open_pq, (new_obj2 + heuristic(node, k1_mask, new_p_mask2, new_c_mask2), new_obj2,
                                             new_robot_time2, new_picked2, new_state2))

        # move
        for v in neighbors_inner(node):
            move_time = times["time_approach"] + vertical_time_exact(node, v, heights_map, times)
            arrival_time = robot_time_g + move_time
            if v in kfs1_index:
                idxv = kfs1_index[v]
                if ((k1_mask >> idxv) & 1) == 0:
                    continue
                completion_v = k1_comp_tuple[idxv]
                effective_arrival = max(arrival_time, completion_v)
            else:
                effective_arrival = arrival_time
            if v in kfs2_index and not kfs2_is_handled(p_mask, c_mask, v):
                continue
            # 关键：调用 is_passable 验证 v 是否可通行（包含 kfs_locked 检查）
            if not is_passable(v, k1_mask, p_mask, c_mask, effective_arrival, k1_comp_tuple):
                continue  # 若不可通行（如 v 是 kfs_locked），则跳过
            new_robot_time = effective_arrival
            new_picked = picked_g_cnt
            new_obj = w_time * new_robot_time - w_pick * new_picked
            new_state = (v, k1_mask, p_mask, c_mask, ext_time, k1_comp_tuple, ext_pos_idx)
            if new_obj + 1e-12 < g_obj.get(new_state, INF):
                g_obj[new_state] = new_obj;
                g_time[new_state] = new_robot_time;
                g_picked[new_state] = new_picked
                came_from[new_state] = (state, ("move", v, move_time, ext_time, ext_time))
                heapq.heappush(open_pq,
                               (new_obj + heuristic(v, k1_mask, p_mask, c_mask), new_obj, new_robot_time, new_picked,
                                new_state))

        # exit candidate
        if node in exit_nodes and node != kfs_locked and picked_g_cnt >= min_kfs2_picked_required:
            # NOTE: do not finalize final_time here directly — we'll compute final_time by simulating the whole action sequence
            exit_descent = descent_to_zero_time(heights_map[node], times)
            extra_move = times["time_approach"]
            final_time_est = robot_time_g + exit_descent + extra_move
            final_obj = w_time * final_time_est - w_pick * picked_g_cnt
            candidate = (final_obj, final_time_est, picked_g_cnt,
                         (node, k1_mask, p_mask, c_mask, ext_time, k1_comp_tuple, ext_pos_idx))
            if best_goal is None or final_obj < best_goal[0] - 1e-12:
                best_goal = candidate

    if best_goal is None:
        raise RuntimeError("无法到达出口且满足至少拾取 KFS2 的约束（不可达或约束不满足）")

    best_obj_val, best_time_est, best_picked_cnt, best_state = best_goal

    # 回溯 actions（按时间顺序）
    actions = []
    cur = best_state
    while True:
        prev = came_from.get(cur)
        if prev is None:
            break
        prev_state, action = prev
        actions.append((action, cur))
        cur = prev_state
    actions.reverse()

    # 回溯路径状态序列
    path_states = []
    cur = best_state
    while True:
        path_states.append(cur)
        prev = came_from.get(cur)
        if prev is None:
            break
        cur = prev[0]
    path_states.reverse()
    node_path = []
    for st in path_states:
        nd = st[0]
        if not node_path or node_path[-1] != nd:
            node_path.append(nd)

    # --- 关键修复：根据回溯的 actions 逐步模拟 robot_time（包含等待外部 KFS1 完成） ---
    # 找初始状态（path_states[0]）的机器人时间（g_time 存的）
    initial_state = path_states[0]
    robot_time_sim = g_time.get(initial_state, 0.0)  # initial robot time
    # k1 completion dict (we'll update as we see pick_kfs1_ext actions)
    # start from initial state's k1 completion tuple
    initial_k1_comp_tuple = initial_state[5] if len(initial_state) > 5 else tuple([0.0] * total_kfs1)
    k1_completion = list(initial_k1_comp_tuple)

    # 现在按 actions 顺序模拟
    for (action, after_state) in actions:
        aname = action[0]
        if aname == "move":
            # ("move", v, move_time, ext_before, ext_after)
            _, v, move_time, ext_before, ext_after = action
            # arrival
            robot_time_sim += move_time
            # if v has KFS1 -> may need to wait
            if v in kfs1_index:
                idxv = kfs1_index[v]
                idxv = kfs1_index[v]
                comp_v = k1_completion[idxv]
                if comp_v > robot_time_sim:
                    robot_time_sim = comp_v  # wait for external completion
        elif aname == "pick_kfs2":
            _, v, dur, ext_before, ext_after = action
            robot_time_sim += dur
        elif aname == "clean_kfs2":
            _, v, dur, ext_before, ext_after = action
            robot_time_sim += dur
        elif aname == "pick_kfs1_ext":
            # ("pick_kfs1_ext", k1_node, dur, ext_before, new_ext_time, ext_pos_before, ext_pos_after)
            _, k1_node, dur, ext_before, ext_after, ext_pos_before, ext_pos_after = action
            # update k1_completion for that node (external completion)
            if k1_node in kfs1_index:
                k1_completion[kfs1_index[k1_node]] = ext_after
            # robot_time_sim does NOT increase because external timeline is separate
        else:
            # unknown action type — ignore
            pass

    # 最后 robot_time_sim 为机器人到达 best_state.node 时的真实时间（包含所有等待）
    # 计算最终离开时间（降到 0 + 额外走出）
    final_node = best_state[0] if isinstance(best_state, tuple) else best_state[0]
    exit_descent = descent_to_zero_time(heights_map[final_node], times)
    extra_move = times["time_approach"]
    final_time_sim = robot_time_sim + exit_descent + extra_move

    # --- 提取外部 pick 事件并构建外部路径段 ext_path_segments（原样保留） ---
    ext_path_segments = []
    order = 0
    if path_states:
        initial_state = path_states[0]
        initial_ext_time = initial_state[4]
        initial_ext_pos = initial_state[6]
        initial_k1_comp = initial_state[5]
        for idx_k1, k1_node in enumerate(kfs1_list):
            comp_t = initial_k1_comp[idx_k1] if idx_k1 < len(initial_k1_comp) else 0
            if comp_t and all(not (act[0][0] == "pick_kfs1_ext" and act[0][1] == k1_node) for act in actions):
                steps = pink_ring_distance(ext_start_idx, initial_ext_pos)
                move_time = steps * ext_move_time
                dur = time_pick_kfs1_map.get(k1_node, ext_pick_kfs1_time_default)
                arrival_time = 0.0 + move_time
                completion_time = comp_t
                order += 1
                ext_path_segments.append(
                    (order, k1_node, ext_start_idx, initial_ext_pos, arrival_time, completion_time))

    for (action, after_state) in actions:
        if action[0] == "pick_kfs1_ext":
            _, k1_node, dur, ext_before, ext_after, ext_pos_before, ext_pos_after = action
            steps = pink_ring_distance(ext_pos_before, ext_pos_after)
            move_time = steps * ext_move_time
            arrival_time = ext_before + move_time
            completion_time = ext_after
            order += 1
            ext_path_segments.append((order, k1_node, ext_pos_before, ext_pos_after, arrival_time, completion_time))

    # 生成 kfs1_pick_events_ordered (node,dur,completion) 以便原先的图示使用
    kfs1_pick_events_ordered = []
    for idx_seg, (ordn, k1_node, from_idx, to_idx, arr, comp) in enumerate(ext_path_segments, start=1):
        dur = time_pick_kfs1_map.get(k1_node, ext_pick_kfs1_time_default)
        kfs1_pick_events_ordered.append((idx_seg, k1_node, dur, comp))

    stats = {
        'objective': best_obj_val,
        # 用真实模拟得到的 final time （替换原来的 best_time_est）
        'time': final_time_sim,
        'picked_kfs2_count': best_picked_cnt,
        'picked_mask_kfs2': best_state[2],
        'cleaned_mask_kfs2': best_state[3],
        'kfs1_scheduled_mask': best_state[1],
        'kfs1_completion_times': best_state[5],
        'final_state': best_state,
        'actions': actions,
        'computation_time': time.time() - start_time  # 添加A*算法计算时间
    }

    return final_time_sim, node_path, stats, kfs1_pick_events_ordered, ext_path_segments


# ---------------------------
# 绘图：包含粉色跑道、内部/外部路径（alpha=0.6）
# ext_path_segments: list of (order, kfs1_node, from_idx, to_idx, arrival_time, completion_time)
# ---------------------------
def draw_grid_with_pink_ring(ax, heights: Dict[int, int], path: Optional[List[int]],
                             entry_nodes: Tuple[int, int], exit_nodes: Tuple[int, int],
                             kfs_locked: int, kfs1_set: set, kfs2_positions: List[int],
                             picked_mask_kfs2: int, cleaned_mask_kfs2: int,
                             kfs1_pick_events: List[Tuple[int, int, float, float]],
                             ext_path_segments: List[Tuple[int, int, int, int, float, float]],
                             ext_move_time: float):
    plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]
    plt.rcParams["axes.unicode_minus"] = False

    height_colors_inner = {200: '#175300', 400: '#00732f', 600: '#95a642'}
    pink_color = '#ffe6ef'

    # 画外层格子（粉色环 + 内部按高度）
    for r in range(OUT_ROWS):
        for c in range(OUT_COLS):
            if (r, c) in PINK_INDEX:
                face = pink_color
            else:
                inner = outer_to_inner_node(r, c)
                if inner is not None:
                    face = height_colors_inner[heights[inner]]
                else:
                    face = '#dddddd'
            rect = plt.Rectangle((c, OUT_ROWS - 1 - r), 1, 1, facecolor=face, edgecolor='black', linewidth=0.8)
            ax.add_patch(rect)
            inner = outer_to_inner_node(r, c)
            if inner is not None:
                ax.text(c + 0.12, OUT_ROWS - 1 - r + 0.75, f'{inner}', ha='left', va='center', fontsize=9,
                        fontweight='bold', color='white')

    # 绘 KFS1 / KFS2 / KFS_locked
    for node in kfs1_set:
        r_out, c_out = inner_to_outer_coord(node)
        row_disp = OUT_ROWS - 1 - r_out
        ax.add_patch(plt.Rectangle((c_out + 0.2, row_disp + 0.2), 0.6, 0.6, facecolor='#ffbb44', edgecolor='black',
                                   linewidth=1.2))
    for i, node in enumerate(kfs2_positions):
        r_out, c_out = inner_to_outer_coord(node)
        row_disp = OUT_ROWS - 1 - r_out
        bit = 1 << i
        if (picked_mask_kfs2 & bit):
            ax.add_patch(
                plt.Rectangle((c_out + 0.28, row_disp + 0.28), 0.44, 0.44, facecolor='#f0e6ff', edgecolor='purple',
                              linewidth=1.0))
        elif (cleaned_mask_kfs2 & bit):
            ax.add_patch(
                plt.Rectangle((c_out + 0.26, row_disp + 0.26), 0.48, 0.48, facecolor='#e6f2ff', edgecolor='blue',
                              linewidth=1.0))
        else:
            ax.add_patch(plt.Rectangle((c_out + 0.2, row_disp + 0.2), 0.6, 0.6, facecolor='#aa66cc', edgecolor='black',
                                       linewidth=1.2))

    if kfs_locked is not None:
        rr_out, cc_out = inner_to_outer_coord(kfs_locked)
        row_disp = OUT_ROWS - 1 - rr_out
        ax.add_patch(plt.Rectangle((cc_out + 0.2, row_disp + 0.2), 0.6, 0.6, facecolor='#444444', edgecolor='black',
                                   linewidth=1.2))

    # 内部机器人路径（半透明 alpha 0.6）
    if path is not None and len(path) >= 2:
        for i in range(len(path) - 1):
            s = path[i];
            t = path[i + 1]
            sr_out, sc_out = inner_to_outer_coord(s)
            tr_out, tc_out = inner_to_outer_coord(t)
            sx, sy = sc_out + 0.5, OUT_ROWS - 1 - sr_out + 0.5
            tx, ty = tc_out + 0.5, OUT_ROWS - 1 - tr_out + 0.5
            ax.arrow(sx, sy, tx - sx, ty - sy, head_width=0.12, head_length=0.12, length_includes_head=True,
                     fc='blue', ec='blue', linewidth=2, zorder=6, alpha=0.6)

    # 外部机器人路径（沿 ring segments，alpha 0.6）
    for (order, k1_node, from_idx, to_idx, arrival_time, completion_time) in ext_path_segments:
        ring_indices = ring_shortest_path_indices(from_idx, to_idx)
        # draw arrows along ring path (step by step)
        for i in range(len(ring_indices) - 1):
            a_idx = ring_indices[i];
            b_idx = ring_indices[i + 1]
            a_pos = PINK_RING[a_idx];
            b_pos = PINK_RING[b_idx]
            ax_x1 = a_pos[1] + 0.5;
            ax_y1 = OUT_ROWS - 1 - a_pos[0] + 0.5
            ax_x2 = b_pos[1] + 0.5;
            ax_y2 = OUT_ROWS - 1 - b_pos[0] + 0.5
            ax.arrow(ax_x1, ax_y1, ax_x2 - ax_x1, ax_y2 - ax_y1, head_width=0.08, head_length=0.08,
                     length_includes_head=True, fc='magenta', ec='magenta', linewidth=1.2, zorder=7, alpha=0.6)
        # 在目标粉色格附近标注执行序号与完成时间（也在内部 KFS1 那一格处有同样信息）
        to_pos = PINK_RING[to_idx]
        tx = to_pos[1] + 0.5;
        ty = OUT_ROWS - 1 - to_pos[0] + 0.2
        ax.text(tx, ty, f'#{order}\nc={completion_time:.2f}s', ha='center', va='center', fontsize=7,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    # 在 KFS1 被安排事件处添加序号/耗时/完成时间（内部格上）
    for (order, node, dur, comp_time) in kfs1_pick_events:
        r_out, c_out = inner_to_outer_coord(node)
        row_disp = OUT_ROWS - 1 - r_out
        ax.text(c_out + 0.8, row_disp + 0.15, f'#{order}\n{dur:.2f}s\nc={comp_time:.2f}s', ha='center', va='center',
                fontsize=7,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1))

    ax.set_xlim(0, OUT_COLS);
    ax.set_ylim(0, OUT_ROWS);
    ax.set_aspect('equal');
    ax.axis('off')


# ---------------------------
# 主程序示例：随机生成并绘出 9 个子图
# ---------------------------
if __name__ == "__main__":
    # 记录总程序开始时间
    total_start_time = time.time()
    random.seed(None)

    heights = {
        1: 400, 2: 200, 3: 400,
        4: 200, 5: 400, 6: 600,
        7: 400, 8: 600, 9: 400,
        10: 200, 11: 400, 12: 200
    }

    times = {
        "time_approach": 0.5,
        "time_up_low": 1.60,
        "time_up_high": 3.20,
        "time_down_low": 2.00,
        "time_down_high": 4.00,
        "time_down_to_zero": 5.00,
        "time_up_to_600": 4.50,
        "time_pick_kfs1_default": 2.5,
        "time_pick_kfs2": 1.6,
        "time_clean_kfs2": 0.3,
        "w_time": 1.0,
        "w_pick": 100.0
    }

    # 外部机器人（粉色跑道）参数
    ext_move_time = 0.8
    ext_pick_kfs1_time_default = 2.2

    kfs1_candidates = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12]
    kfs_locked_candidates = list(range(4, 13))
    all_cells = list(range(1, 13))

    time_pick_kfs1_map_global = {
        1: 2.0, 2: 3.0, 3: 2.2, 4: 1.8, 6: 2.6,
        7: 2.4, 9: 2.1, 10: 1.9, 11: 2.7, 12: 2.3
    }

    entry_nodes = (1, 2, 3)
    exit_nodes = (10, 11, 12)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    # 存储每个案例的计算时间，用于后续统计
    computation_times = []

    for idx, kfs_locked in enumerate(kfs_locked_candidates):
        ax = axes[idx]

        # 记录当前案例开始时间
        case_start_time = time.time()

        available_kfs1 = [c for c in kfs1_candidates if c != kfs_locked]
        sample_kfs1_count = min(3, len(available_kfs1))
        kfs1_set = set(random.sample(available_kfs1, sample_kfs1_count))

        candidates_for_kfs2 = [c for c in all_cells if c != kfs_locked and c not in kfs1_set]
        kfs2_count = min(4, len(candidates_for_kfs2))
        kfs2_positions = random.sample(candidates_for_kfs2, kfs2_count)

        time_pick_kfs1_map = {node: time_pick_kfs1_map_global.get(node, times["time_pick_kfs1_default"]) for node in
                              kfs1_set}

        try:
            best_time, node_path, stats, kfs1_pick_events, ext_path_segments = a_star_with_external_ring_kfs1(
                heights_map=heights,
                times=times,
                ext_move_time=ext_move_time,
                ext_pick_kfs1_time_default=ext_pick_kfs1_time_default,
                kfs_locked=kfs_locked,
                kfs1_set=kfs1_set,
                time_pick_kfs1_map=time_pick_kfs1_map,
                kfs2_positions=kfs2_positions,
                entry_nodes=entry_nodes,
                exit_nodes=exit_nodes,
                min_kfs2_picked_required=1
            )
            picked_mask_kfs2 = stats['picked_mask_kfs2']
            cleaned_mask_kfs2 = stats['cleaned_mask_kfs2']
            picked_cnt = stats['picked_kfs2_count']
            display_time = stats['time']
            computation_time = stats['computation_time']
            computation_times.append(computation_time)
            print(f"案例 {idx + 1} 计算完成，A*算法耗时: {computation_time:.4f}秒")
        except RuntimeError:
            node_path = None
            picked_mask_kfs2 = 0
            cleaned_mask_kfs2 = 0
            picked_cnt = 0
            display_time = 0.0
            kfs1_pick_events = []
            ext_path_segments = []
            computation_time = time.time() - case_start_time
            computation_times.append(computation_time)
            print(f"案例 {idx + 1} 无法求解，耗时: {computation_time:.4f}秒")

        draw_grid_with_pink_ring(ax, heights, node_path, entry_nodes, exit_nodes,
                                 kfs_locked, kfs1_set, kfs2_positions,
                                 picked_mask_kfs2, cleaned_mask_kfs2,
                                 kfs1_pick_events, ext_path_segments, ext_move_time)

        # 内部路径箭头（如果 node_path 存在则 draw 已画过）；这里我们不重复画
        ax.text(-0.53, 0.02, f'locked={kfs_locked}\nKFS1={sorted(kfs1_set)}\nKFS2={sorted(kfs2_positions)}',
                transform=ax.transAxes, fontsize=7, ha='left', va='bottom',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        if node_path is None:
            ax.text(2.5, 3.0, '不可达', ha='center', va='center', fontsize=12, color='red', fontweight='bold',
                    zorder=10)
        else:
            ax.text(7.0, 5.6, f't={display_time:.2f}s\npicked={picked_cnt}', ha='right', va='top', fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1), zorder=10)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color='w', marker='s', markerfacecolor='#175300', markersize=8, label='200mm'),
        Line2D([0], [0], color='w', marker='s', markerfacecolor='#00732f', markersize=8, label='400mm'),
        Line2D([0], [0], color='w', marker='s', markerfacecolor='#95a642', markersize=8, label='600mm'),
        Line2D([0], [0], color='green', marker='o', markerfacecolor='w', label='入口', markersize=6),
        Line2D([0], [0], color='red', marker='o', markerfacecolor='w', label='出口', markersize=6),
        Line2D([0], [0], color='w', marker='s', markerfacecolor='#ffbb44', markersize=7, label='KFS1（外部跑道拾取）'),
        Line2D([0], [0], color='w', marker='s', markerfacecolor='#aa66cc', markersize=7, label='KFS2（未处理）'),
        Line2D([0], [0], color='w', marker='s', markerfacecolor='#f0e6ff', markeredgecolor='purple', markersize=6,
               label='KFS2（已拾取）'),
        Line2D([0], [0], color='w', marker='s', markerfacecolor='#e6f2ff', markeredgecolor='blue', markersize=6,
               label='KFS2（已清除但未拾取）'),
        Line2D([0], [0], color='w', marker='s', markerfacecolor='#ffe6ef', markersize=7,
               label='粉色跑道（外部机器人活动区域）'),
        Line2D([0], [0], color='w', marker='s', markerfacecolor='#444444', markersize=7, label='KFS_locked（不可拾取）'),
        Line2D([0], [0], color='magenta', marker='>', markerfacecolor='magenta', label='外部机器人路径', alpha=0.6),
        Line2D([0], [0], color='blue', marker='>', markerfacecolor='blue', label='内部机器人路径', alpha=0.6),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=5,
        bbox_to_anchor=(0.5, -0.01),
        prop={'size': 7, 'weight': 'bold'}
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # 计算并显示总耗时和统计信息
    total_time = time.time() - total_start_time
    print(f"\n总程序运行时间: {total_time:.4f}秒")
    if computation_times:
        avg_time = sum(computation_times) / len(computation_times)
        print(f"A*算法平均耗时: {avg_time:.4f}秒")
        print(f"A*算法总耗时: {sum(computation_times):.4f}秒")

    plt.show()
