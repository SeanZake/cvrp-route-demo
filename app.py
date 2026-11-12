import copy
import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# 页面配置
# =========================

st.set_page_config(
    page_title="物流配送车辆路径优化系统",
    page_icon="🚚",
    layout="wide",
)

# =========================
# 数据生成与基础计算
# =========================

def generate_instance(
    customer_count: int,
    vehicle_capacity: int,
    seed: int,
):
    """
    生成一个随机VRP实例。
    0号节点为仓库，其余节点为客户。
    """
    rng = np.random.default_rng(seed)

    coordinates = {
        0: (50.0, 50.0)
    }

    demands = {
        0: 0
    }

    for customer_id in range(1, customer_count + 1):
        coordinates[customer_id] = (
            float(rng.uniform(5, 95)),
            float(rng.uniform(5, 95)),
        )

        demands[customer_id] = int(
            rng.integers(
                low=1,
                high=max(2, min(11, vehicle_capacity)),
            )
        )

    return coordinates, demands

def build_distance_matrix(coordinates):
    """构造距离矩阵。"""
    node_ids = sorted(coordinates.keys())
    matrix = {}

    for i in node_ids:
        matrix[i] = {}

        for j in node_ids:
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]

            matrix[i][j] = math.hypot(x2 - x1, y2 - y1)

    return matrix

def route_distance(route, distance_matrix):
    """计算单条路线距离。"""
    return sum(
        distance_matrix[start][end]
        for start, end in zip(route[:-1], route[1:])
    )

def total_distance(routes, distance_matrix):
    """计算所有路线总距离。"""
    return sum(
        route_distance(route, distance_matrix)
        for route in routes
    )

def route_load(route, demands):
    """计算单条路线的配送量。"""
    return sum(
        demands[node]
        for node in route
        if node != 0
    )

def is_feasible(routes, demands, vehicle_capacity, customer_count):
    """
    检查方案是否满足约束：
    1. 每个客户恰好出现一次
    2. 没有客户遗漏
    3. 每条路线不超载
    """
    visited = []

    for route in routes:
        if not route:
            return False

        if route[0] != 0 or route[-1] != 0:
            return False

        load = route_load(route, demands)

        if load > vehicle_capacity:
            return False

        visited.extend(
            node for node in route
            if node != 0
        )

    expected = list(range(1, customer_count + 1))

    return (
        sorted(visited) == expected
        and len(visited) == len(set(visited))
    )

# =========================
# 最近邻贪心算法
# =========================

def greedy_solution(
    coordinates,
    demands,
    vehicle_capacity,
):
    """
    使用容量约束最近邻方法生成初始方案。
    """
    distance_matrix = build_distance_matrix(coordinates)

    customer_ids = set(
        node for node in coordinates
        if node != 0
    )

    unvisited = set(customer_ids)
    routes = []

    while unvisited:
        route = [0]
        current_node = 0
        current_load = 0

        while True:
            candidates = [
                customer for customer in unvisited
                if current_load + demands[customer] <= vehicle_capacity
            ]

            if not candidates:
                break

            next_customer = min(
                candidates,
                key=lambda customer:
                distance_matrix[current_node][customer],
            )

            route.append(next_customer)
            unvisited.remove(next_customer)

            current_node = next_customer
            current_load += demands[next_customer]

        route.append(0)
        routes.append(route)

    return routes

# =========================
# 模拟退火邻域操作
# =========================

def neighbor_two_opt(routes):
    """
    对一条路线进行2-opt操作。
    只改变访问顺序，不改变载重。
    """
    candidate = copy.deepcopy(routes)

    valid_route_ids = [
        index for index, route in enumerate(candidate)
        if len(route) >= 4
    ]

    if not valid_route_ids:
        return candidate

    route_id = random.choice(valid_route_ids)
    route = candidate[route_id]

    left, right = sorted(
        random.sample(range(1, len(route) - 1), 2)
    )

    route[left:right + 1] = reversed(
        route[left:right + 1]
    )

    return candidate

def neighbor_relocate(
    routes,
    demands,
    vehicle_capacity,
):
    """
    将一个客户从一条路线移动到另一条路线。
    """
    candidate = copy.deepcopy(routes)

    source_ids = [
        index for index, route in enumerate(candidate)
        if len(route) > 2
    ]

    if not source_ids:
        return candidate

    source_id = random.choice(source_ids)
    source_route = candidate[source_id]

    customer_position = random.randint(
        1,
        len(source_route) - 2,
    )

    customer = source_route.pop(customer_position)

    target_ids = list(range(len(candidate)))

    if len(target_ids) > 1:
        target_ids.remove(source_id)

    target_id = random.choice(target_ids)
    target_route = candidate[target_id]

    insert_position = random.randint(
        1,
        len(target_route) - 1,
    )

    target_load = route_load(target_route, demands)

    if target_load + demands[customer] <= vehicle_capacity:
        target_route.insert(insert_position, customer)
    else:
        source_route.insert(customer_position, customer)

    # 删除可能出现的空路线
    candidate = [
        route for route in candidate
        if len(route) > 2
    ]

    return candidate

def neighbor_swap(
    routes,
    demands,
    vehicle_capacity,
):
    """
    交换两条不同路线中的客户。
    """
    candidate = copy.deepcopy(routes)

    valid_route_ids = [
        index for index, route in enumerate(candidate)
        if len(route) > 2
    ]

    if len(valid_route_ids) < 2:
        return candidate

    route_a_id, route_b_id = random.sample(
        valid_route_ids,
        2,
    )

    route_a = candidate[route_a_id]
    route_b = candidate[route_b_id]

    position_a = random.randint(1, len(route_a) - 2)
    position_b = random.randint(1, len(route_b) - 2)

    customer_a = route_a[position_a]
    customer_b = route_b[position_b]

    load_a = route_load(route_a, demands)
    load_b = route_load(route_b, demands)

    new_load_a = (
        load_a
        - demands[customer_a]
        + demands[customer_b]
    )

    new_load_b = (
        load_b
        - demands[customer_b]
        + demands[customer_a]
    )

    if (
        new_load_a <= vehicle_capacity
        and new_load_b <= vehicle_capacity
    ):
        route_a[position_a] = customer_b
        route_b[position_b] = customer_a

    return candidate

def generate_neighbor(
    routes,
    demands,
    vehicle_capacity,
):
    """随机选择一种邻域操作。"""
    operation = random.choice(
        ["two_opt", "relocate", "swap"]
    )

    if operation == "two_opt":
        return neighbor_two_opt(routes)

    if operation == "relocate":
        return neighbor_relocate(
            routes,
            demands,
            vehicle_capacity,
        )

    return neighbor_swap(
        routes,
        demands,
        vehicle_capacity,
    )

# =========================
# 模拟退火算法
# =========================

def simulated_annealing(
    initial_routes,
    demands,
    vehicle_capacity,
    customer_count,
    distance_matrix,
    initial_temperature=100.0,
    cooling_rate=0.995,
    minimum_temperature=0.01,
    iterations_per_temperature=80,
    max_iterations=8000,
    seed=42,
):
    """
    模拟退火算法。
    返回历史最优路线、历史最优距离和收敛数据。
    """
    random.seed(seed)

    current_routes = copy.deepcopy(initial_routes)
    best_routes = copy.deepcopy(initial_routes)

    current_cost = total_distance(
        current_routes,
        distance_matrix,
    )

    best_cost = current_cost
    temperature = initial_temperature

    history = []
    iteration = 0

    while (
        temperature > minimum_temperature
        and iteration < max_iterations
    ):
        for _ in range(iterations_per_temperature):
            if iteration >= max_iterations:
                break

            candidate_routes = generate_neighbor(
                current_routes,
                demands,
                vehicle_capacity,
            )

            if not is_feasible(
                candidate_routes,
                demands,
                vehicle_capacity,
                customer_count,
            ):
                continue

            candidate_cost = total_distance(
                candidate_routes,
                distance_matrix,
            )

            delta = candidate_cost - current_cost

            if (
                delta <= 0
                or random.random()
                < math.exp(-delta / temperature)
            ):
                current_routes = candidate_routes
                current_cost = candidate_cost

            if current_cost < best_cost:
                best_routes = copy.deepcopy(current_routes)
                best_cost = current_cost

            history.append(best_cost)
            iteration += 1

        temperature *= cooling_rate

    return best_routes, best_cost, history

# =========================
# 可视化函数
# =========================

def plot_routes(
    coordinates,
    demands,
    routes,
    distance_matrix,
    title,
):
    """
    绘制配送路线图。
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.tab20(
        np.linspace(0, 1, max(len(routes), 1))
    )

    for route_id, route in enumerate(routes):
        x_values = [
            coordinates[node][0]
            for node in route
        ]

        y_values = [
            coordinates[node][1]
            for node in route
        ]

        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            markersize=5,
            color=colors[route_id % len(colors)],
            label=(
                f"Vehicle {route_id + 1} "
                f"({route_distance(route, distance_matrix):.1f})"
            ),
        )

    # 绘制仓库
    depot_x, depot_y = coordinates[0]

    ax.scatter(
        depot_x,
        depot_y,
        s=280,
        marker="*",
        color="red",
        edgecolors="black",
        linewidths=1,
        zorder=5,
        label="Depot",
    )

    ax.annotate(
        "Depot",
        (depot_x, depot_y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )

    # 绘制客户
    for customer_id in sorted(demands):
        if customer_id == 0:
            continue

        x, y = coordinates[customer_id]

        ax.scatter(
            x,
            y,
            s=45,
            color="black",
            zorder=4,
        )

        ax.annotate(
            f"{customer_id} (q={demands[customer_id]})",
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)

    if len(routes) <= 10:
        ax.legend(
            loc="best",
            fontsize=8,
        )

    fig.tight_layout()

    return fig

def plot_convergence(history):
    """绘制模拟退火收敛曲线。"""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    if history:
        ax.plot(
            range(1, len(history) + 1),
            history,
            color="tab:blue",
            linewidth=1.5,
        )

    ax.set_title("Simulated Annealing Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Distance")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    return fig

def build_route_records(
    routes,
    demands,
    vehicle_capacity,
    distance_matrix,
):
    """生成车辆明细表。"""
    records = []

    for index, route in enumerate(routes, start=1):
        load = route_load(route, demands)
        distance = route_distance(route, distance_matrix)

        records.append(
            {
                "车辆": index,
                "访问顺序": " → ".join(
                    str(node) for node in route
                ),
                "配送量": load,
                "车辆容量": vehicle_capacity,
                "装载率": f"{load / vehicle_capacity:.1%}",
                "行驶距离": round(distance, 2),
                "是否超载": "是"
                if load > vehicle_capacity
                else "否",
            }
        )

    return pd.DataFrame(records)

# =========================
# Streamlit界面
# =========================

st.title("物流配送车辆路径优化系统")

st.caption(
    "基于容量约束车辆路径问题（CVRP）的贪心算法与模拟退火优化系统"
)

st.sidebar.header("实验参数")

customer_count = st.sidebar.slider(
    "客户数量",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
)

vehicle_capacity = st.sidebar.slider(
    "车辆容量",
    min_value=10,
    max_value=100,
    value=30,
    step=5,
)

data_seed = st.sidebar.number_input(
    "数据随机种子",
    min_value=0,
    max_value=999999,
    value=42,
    step=1,
)

algorithm_seed = st.sidebar.number_input(
    "算法随机种子",
    min_value=0,
    max_value=999999,
    value=2024,
    step=1,
)

cooling_rate = st.sidebar.slider(
    "降温系数",
    min_value=0.90,
    max_value=0.999,
    value=0.995,
    step=0.001,
)

max_iterations = st.sidebar.slider(
    "最大迭代次数",
    min_value=1000,
    max_value=30000,
    value=8000,
    step=1000,
)

run_button = st.sidebar.button(
    "开始生成并优化",
    type="primary",
)

# =========================
# 生成或运行实例
# =========================

if (
    "coordinates" not in st.session_state
    or run_button
):
    with st.spinner("正在生成客户数据并运行优化算法..."):
        coordinates, demands = generate_instance(
            customer_count=customer_count,
            vehicle_capacity=vehicle_capacity,
            seed=int(data_seed),
        )

        distance_matrix = build_distance_matrix(
            coordinates
        )

        impossible_customers = [
            customer_id
            for customer_id in demands
            if customer_id != 0
            and demands[customer_id] > vehicle_capacity
        ]

        if impossible_customers:
            st.error(
                "存在客户需求超过车辆容量，请增大车辆容量。"
            )
            st.stop()

        start_time = time.perf_counter()

        initial_routes = greedy_solution(
            coordinates,
            demands,
            vehicle_capacity,
        )

        initial_distance = total_distance(
            initial_routes,
            distance_matrix,
        )

        optimized_routes, optimized_distance, history = (
            simulated_annealing(
                initial_routes=initial_routes,
                demands=demands,
                vehicle_capacity=vehicle_capacity,
                customer_count=customer_count,
                distance_matrix=distance_matrix,
                cooling_rate=cooling_rate,
                max_iterations=max_iterations,
                seed=int(algorithm_seed),
            )
        )

        elapsed_time = time.perf_counter() - start_time

        st.session_state.coordinates = coordinates
        st.session_state.demands = demands
        st.session_state.distance_matrix = distance_matrix
        st.session_state.initial_routes = initial_routes
        st.session_state.optimized_routes = optimized_routes
        st.session_state.initial_distance = initial_distance
        st.session_state.optimized_distance = optimized_distance
        st.session_state.history = history
        st.session_state.elapsed_time = elapsed_time

coordinates = st.session_state.coordinates
demands = st.session_state.demands
distance_matrix = st.session_state.distance_matrix
initial_routes = st.session_state.initial_routes
optimized_routes = st.session_state.optimized_routes
initial_distance = st.session_state.initial_distance
optimized_distance = st.session_state.optimized_distance
history = st.session_state.history
elapsed_time = st.session_state.elapsed_time

# =========================
# 指标区域
# =========================

improvement = 0.0

if initial_distance > 0:
    improvement = (
        (initial_distance - optimized_distance)
        / initial_distance
        * 100
    )

total_demand = sum(demands.values())
average_load_rate = (
    total_demand
    / (len(optimized_routes) * vehicle_capacity)
)

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "初始总距离",
    f"{initial_distance:.2f}",
)

metric2.metric(
    "优化后总距离",
    f"{optimized_distance:.2f}",
    delta=f"-{initial_distance - optimized_distance:.2f}",
)

metric3.metric(
    "距离改善率",
    f"{improvement:.2f}%",
)

metric4.metric(
    "使用车辆数",
    f"{len(optimized_routes)}",
)

st.success(
    "优化方案满足客户覆盖约束和车辆容量约束。"
)

st.info(
    f"平均装载率：{average_load_rate:.1%}　"
    f"算法运行时间：{elapsed_time:.3f} 秒"
)

# =========================
# 路线图
# =========================

st.subheader("配送路线对比")

left_chart, right_chart = st.columns(2)

with left_chart:
    initial_fig = plot_routes(
        coordinates=coordinates,
        demands=demands,
        routes=initial_routes,
        distance_matrix=distance_matrix,
        title=(
            f"Greedy Initial Routes "
            f"(Distance: {initial_distance:.2f})"
        ),
    )

    st.pyplot(
        initial_fig,
        use_container_width=True,
    )

    plt.close(initial_fig)

with right_chart:
    optimized_fig = plot_routes(
        coordinates=coordinates,
        demands=demands,
        routes=optimized_routes,
        distance_matrix=distance_matrix,
        title=(
            f"Optimized Routes "
            f"(Distance: {optimized_distance:.2f})"
        ),
    )

    st.pyplot(
        optimized_fig,
        use_container_width=True,
    )

    plt.close(optimized_fig)

# =========================
# 收敛曲线
# =========================

st.subheader("模拟退火收敛曲线")

convergence_fig = plot_convergence(history)

st.pyplot(
    convergence_fig,
    use_container_width=True,
)

plt.close(convergence_fig)

# =========================
# 车辆配送明细
# =========================

st.subheader("优化后车辆配送明细")

route_df = build_route_records(
    routes=optimized_routes,
    demands=demands,
    vehicle_capacity=vehicle_capacity,
    distance_matrix=distance_matrix,
)

st.dataframe(
    route_df,
    use_container_width=True,
    hide_index=True,
)

# =========================
# 客户数据
# =========================

st.subheader("客户数据")

customer_records = []

for customer_id in sorted(coordinates):
    x, y = coordinates[customer_id]

    customer_records.append(
        {
            "节点编号": customer_id,
            "节点类型": "仓库"
            if customer_id == 0
            else "客户",
            "X坐标": round(x, 2),
            "Y坐标": round(y, 2),
            "需求量": demands[customer_id],
        }
    )

customer_df = pd.DataFrame(customer_records)

st.dataframe(
    customer_df,
    use_container_width=True,
    hide_index=True,
)

csv_data = customer_df.to_csv(
    index=False,
    encoding="utf-8-sig",
)

st.download_button(
    label="下载客户数据CSV",
    data=csv_data,
    file_name="vrp_customer_data.csv",
    mime="text/csv",
)
