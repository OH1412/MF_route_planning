import matplotlib.pyplot as plt
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


# ---------------------------
# 内->外层坐标映射（外围 +1）
# 外层尺寸： OUT_ROWS=6, OUT_COLS=5
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
# 绘图函数：展示网格、KFS位置等
# ---------------------------
def draw_custom_kfs(ax,
                    heights: Dict[int, int],
                    entry_nodes: Tuple[int, int],
                    exit_nodes: Tuple[int, int],
                    kfs_locked: int,
                    kfs1_set: set,
                    kfs2_positions: List[int]):
    # 设置中文字体
    plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 高度颜色映射
    height_colors_inner = {200: '#175300', 400: '#00732f', 600: '#95a642'}
    pink_color = '#ffe6ef'

    # 绘制网格
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
            rect = plt.Rectangle((c, OUT_ROWS - 1 - r), 1, 1,
                                 facecolor=face, edgecolor='black', linewidth=0.8)
            ax.add_patch(rect)
            # 标注内部节点编号
            inner = outer_to_inner_node(r, c)
            if inner is not None:
                ax.text(c + 0.12, OUT_ROWS - 1 - r + 0.75, f'{inner}',
                        ha='left', va='center', fontsize=9, fontweight='bold', color='white')

    # 绘制KFS1
    for node in kfs1_set:
        r_out, c_out = inner_to_outer_coord(node)
        row_disp = OUT_ROWS - 1 - r_out
        ax.add_patch(plt.Rectangle((c_out + 0.2, row_disp + 0.2), 0.6, 0.6,
                                   facecolor='#ffbb44', edgecolor='black', linewidth=1.2))
        ax.text(c_out + 0.5, row_disp + 0.5, 'K1',
                ha='center', va='center', fontsize=8, fontweight='bold')

    # 绘制KFS2
    for i, node in enumerate(kfs2_positions):
        r_out, c_out = inner_to_outer_coord(node)
        row_disp = OUT_ROWS - 1 - r_out
        ax.add_patch(plt.Rectangle((c_out + 0.2, row_disp + 0.2), 0.6, 0.6,
                                   facecolor='#aa66cc', edgecolor='black', linewidth=1.2))
        ax.text(c_out + 0.5, row_disp + 0.5, 'K2',
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    # 绘制锁定位置
    if kfs_locked is not None:
        rr_out, cc_out = inner_to_outer_coord(kfs_locked)
        row_disp = OUT_ROWS - 1 - rr_out
        ax.add_patch(plt.Rectangle((cc_out + 0.2, row_disp + 0.2), 0.6, 0.6,
                                   facecolor='#444444', edgecolor='black', linewidth=1.2))
        ax.text(cc_out + 0.5, row_disp + 0.5, '锁',
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    # 标记入口和出口
    for node in entry_nodes:
        r_out, c_out = inner_to_outer_coord(node)
        row_disp = OUT_ROWS - 1 - r_out
        # ax.add_patch(plt.Circle((c_out + 0.5, row_disp + 0.5), 0.15,
        #                         facecolor='none', edgecolor='green', linewidth=2))

    for node in exit_nodes:
        r_out, c_out = inner_to_outer_coord(node)
        row_disp = OUT_ROWS - 1 - r_out
        # ax.add_patch(plt.Circle((c_out + 0.5, row_disp + 0.5), 0.15,
        #                         facecolor='none', edgecolor='red', linewidth=2))

    # 设置坐标轴
    ax.set_xlim(0, OUT_COLS)
    ax.set_ylim(0, OUT_ROWS)
    ax.set_aspect('equal')
    ax.axis('off')


# ---------------------------
# 主程序：自定义参数并绘图
# ---------------------------
if __name__ == "__main__":
    # 高度配置（固定）
    heights = {
        1: 400, 2: 200, 3: 400,
        4: 200, 5: 400, 6: 600,
        7: 400, 8: 600, 9: 400,
        10: 200, 11: 400, 12: 200
    }

    # 入口和出口节点（固定）
    entry_nodes = (1, 2, 3)
    exit_nodes = (10, 11, 12)

    # ---------------------------
    # 若对方没有等待R1的程序
    # ---------------------------
    # 三个KFS1的位置（1-12之间的整数）
    kfs1_set = {1, 9, 11}

    # 四个KFS2的位置（1-12之间的整数）
    kfs2_positions = [3, 4, 7, 10]

    # 锁定位置（1-12之间的整数）
    kfs_locked = 5

    # 创建图形并绘图
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制自定义的KFS配置
    draw_custom_kfs(ax, heights, entry_nodes, exit_nodes,
                    kfs_locked, kfs1_set, kfs2_positions)

    # 添加说明文本
    ax.text(-0.3, 0.02,
            f'KFS1位置: {sorted(kfs1_set)}\n'
            f'KFS2位置: {sorted(kfs2_positions)}\n'
            f'锁定位置: {kfs_locked}',
            transform=ax.transAxes, fontsize=9,
            ha='left', va='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 创建图例
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color='w', marker='s', markerfacecolor='#175300', markersize=8, label='200mm'),
        Line2D([0], [0], color='w', marker='s', markerfacecolor='#00732f', markersize=8, label='400mm'),
        Line2D([0], [0], color='w', marker='s', markerfacecolor='#95a642', markersize=8, label='600mm'),
        Line2D([0], [0], color='green', marker='o', markerfacecolor='w', label='入口', markersize=6),
        Line2D([0], [0], color='red', marker='o', markerfacecolor='w', label='出口', markersize=6),
        Line2D([0], [0], color='w', marker='s', markerfacecolor='#ffbb44', markersize=7, label='KFS1'),
        Line2D([0], [0], color='w', marker='s', markerfacecolor='#aa66cc', markersize=7, label='KFS2'),
        Line2D([0], [0], color='w', marker='s', markerfacecolor='#444444', markersize=7, label='锁定位置'),
        Line2D([0], [0], color='w', marker='s', markerfacecolor='#ffe6ef', markersize=7, label='粉色跑道'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=5, bbox_to_anchor=(0.5, -0.02), prop={'size': 8})

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
#在程序限制下，kfs1、kfs2 和 kfs_locked 的总组合摆放方法为 57,960 种