import dash
from dash import dcc, html, callback, Output, Input, no_update, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
import threading
import epics
import signal
import os
import logging
from collections import deque
from threading import Lock, Event, RLock
import psutil
import queue
from datetime import datetime
import json

# ==================== 配置部分 ====================
# 日志配置
log_filename = f"waveform_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# EPICS配置
PREFIX = "wf128:"
PREFIX_REC = "wf"
NUM_CHANNELS = 128
BUFFER_SIZE = 2000
CHANNELS_PER_TAB = 16  # 每个标签页16个通道 (4x4)
NUM_TABS = 8  # 共8个标签页 (128/16=8)

# 显示配置
FIGURE_HEIGHT = 1400
FIGURE_WIDTH = 2400
ROWS_PER_TAB = 4
COLS_PER_TAB = 4
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#17becf",
    "#bcbd22",
    "#393b79",
    "#8c6d31",
    "#843c39",
    "#d6616b",
    "#7b4173",
    "#637939",
] * 8

UPDATE_INTERVAL = 500  # 固定更新间隔(ms)
SAVE_DIR = "saved_waveforms"

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)


# ==================== 数据部分 ====================
class ChannelData:
    __slots__ = ["buffer", "lock", "last_value", "update_time", "is_active"]

    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.lock = RLock()
        self.last_value = None
        self.update_time = 0
        self.is_active = False

    def update(self, value):
        if value is None:
            return

        with self.lock:
            try:
                arr = np.asarray(value, dtype=np.float32).flatten()
                self.buffer.extend(arr)
                self.last_value = arr
                self.update_time = time.time()
                self.is_active = True
            except Exception as e:
                logger.error(f"Data update error: {str(e)}")

    def get_data(self):
        with self.lock:
            if self.buffer:
                return list(self.buffer)
            return (
                np.zeros(BUFFER_SIZE)
                if self.last_value is None
                else np.asarray(self.last_value)
            )


# 全局状态
class AppState:
    def __init__(self):
        self.paused = False
        self.lock = RLock()
        self.last_save_time = 0
        self.performance_stats = {
            "avg_update_time": 0,
            "max_update_time": 0,
            "update_count": 0,
        }

    def set_paused(self, paused):
        with self.lock:
            self.paused = paused

    def is_paused(self):
        with self.lock:
            return self.paused

    def update_performance(self, elapsed_time):
        with self.lock:
            self.performance_stats["update_count"] += 1
            total_time = self.performance_stats["avg_update_time"] * (
                self.performance_stats["update_count"] - 1
            )
            self.performance_stats["avg_update_time"] = (
                total_time + elapsed_time
            ) / self.performance_stats["update_count"]
            self.performance_stats["max_update_time"] = max(
                self.performance_stats["max_update_time"], elapsed_time
            )


# 数据队列
tab_data_queues = [queue.Queue(maxsize=2000 * 16 * 2) for _ in range(NUM_TABS)]
shutdown_event = Event()
app_state = AppState()


# ==================== EPICS连接 ====================
def make_callback(channel_idx):
    def callback(pvname=None, value=None, char_value=None, **kw):
        try:
            if app_state.is_paused():
                return

            actual_value = value if value is not None else char_value
            if actual_value is None or (
                isinstance(actual_value, (str, bytes)) and not actual_value.strip()
            ):
                return

            tab_idx = channel_idx // CHANNELS_PER_TAB

            if tab_idx < NUM_TABS:
                try:
                    tab_data_queues[tab_idx].put_nowait((channel_idx, actual_value))
                except queue.Full:
                    try:
                        tab_data_queues[tab_idx].get_nowait()
                        tab_data_queues[tab_idx].put_nowait((channel_idx, actual_value))
                    except queue.Empty:
                        pass
        except Exception as e:
            logger.error(f"Callback error for channel {channel_idx}: {str(e)}")

    return callback


def init_pvs():
    logger.info("Initializing PV connections...")
    connections = []

    for i in range(NUM_CHANNELS):
        pv_name = f"{PREFIX}{PREFIX_REC}{i}"
        try:
            pv = epics.PV(
                pv_name,
                callback=make_callback(i),
                auto_monitor=epics.dbr.DBE_VALUE,
                connection_timeout=1.0,
                form="time",
            )
            pv.get(use_monitor=True)
            time.sleep(0.01)
            connections.append(pv)
        except Exception as e:
            logger.error(f"Failed to initialize PV {i}: {str(e)}")
            connections.append(None)

    return connections


# ==================== 标签页处理器 ====================
class TabProcessor(threading.Thread):
    def __init__(self, tab_idx, channels, shutdown_event):
        super().__init__(daemon=True)
        self.tab_idx = tab_idx
        self.channels = channels
        self.shutdown_event = shutdown_event
        self.start_ch = tab_idx * CHANNELS_PER_TAB
        self.end_ch = (tab_idx + 1) * CHANNELS_PER_TAB
        self.last_update = time.time()
        self.batch_size = 10
        self.batch_timeout = 0.05

    def run(self):
        logger.info(f"TabProcessor {self.tab_idx} started")
        while not self.shutdown_event.is_set():
            if app_state.is_paused():
                time.sleep(0.1)
                continue

            batch = []
            start_time = time.time()

            while (
                len(batch) < self.batch_size
                and (time.time() - start_time) < self.batch_timeout
            ):
                try:
                    data = tab_data_queues[self.tab_idx].get_nowait()
                    batch.append(data)
                except queue.Empty:
                    time.sleep(0.001)
                    continue

            if batch:
                try:
                    with channels[self.start_ch].lock:
                        for channel_idx, value in batch:
                            if self.start_ch <= channel_idx < self.end_ch:
                                channels[channel_idx].update(value)
                    self.last_update = time.time()
                except Exception as e:
                    logger.error(f"TabProcessor {self.tab_idx} batch error: {str(e)}")

            if len(batch) == self.batch_size:
                self.batch_size = min(50, self.batch_size + 2)
            else:
                self.batch_size = max(10, self.batch_size - 1)


# ==================== 波形保存功能 ====================
def save_waveforms():
    """保存所有通道的波形数据到TXT文件(128列×2000行)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAVE_DIR, f"waveforms_{timestamp}.txt")

    try:
        data_matrix = []
        for i in range(NUM_CHANNELS):
            channel_data = channels[i].get_data()
            if len(channel_data) < BUFFER_SIZE:
                padded_data = np.pad(
                    channel_data, (0, BUFFER_SIZE - len(channel_data)), "constant"
                )
                data_matrix.append(padded_data[:BUFFER_SIZE])
            else:
                data_matrix.append(channel_data[:BUFFER_SIZE])

        data_matrix = np.array(data_matrix).T
        np.savetxt(filename, data_matrix, fmt="%.6f", delimiter="\t")

        metadata = {
            "timestamp": timestamp,
            "shape": "2000 rows × 128 columns",
            "description": "Each column represents one channel waveform",
        }
        with open(os.path.join(SAVE_DIR, f"metadata_{timestamp}.json"), "w") as f:
            json.dump(metadata, f)

        logger.info(f"Waveforms saved to {filename} (shape: {data_matrix.shape})")
        app_state.last_save_time = time.time()
        return True, filename
    except Exception as e:
        logger.error(f"Failed to save waveforms: {str(e)}")
        return False, str(e)


# ==================== Dash应用 ====================
app = dash.Dash(__name__, compress=True, update_title=None)
server = app.server


def create_tab_figure(tab_idx):
    fig = make_subplots(
        rows=ROWS_PER_TAB,
        cols=COLS_PER_TAB,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        shared_xaxes=False,
        shared_yaxes=False,
    )

    start_ch = tab_idx * CHANNELS_PER_TAB
    end_ch = (tab_idx + 1) * CHANNELS_PER_TAB

    for i in range(start_ch, end_ch):
        row, col = divmod(i - start_ch, COLS_PER_TAB)
        fig.add_trace(
            go.Scatter(
                x=np.arange(100),
                y=np.zeros(100),
                line=dict(width=1, color=COLORS[i % len(COLORS)]),
                name=f"CH{i}",
            ),
            row=row + 1,
            col=col + 1,
        )

    fig.update_layout(
        height=FIGURE_HEIGHT,
        width=FIGURE_WIDTH,
        margin=dict(l=20, r=20, t=50, b=20),
        uirevision=str(tab_idx),
    )
    return fig


# 创建控制面板
controls = html.Div(
    [
        html.Div(
            [
                html.Button(
                    "⏸️ Pause",
                    id="pause-button",
                    n_clicks=0,
                    style={"margin-right": "10px", "padding": "10px 15px"},
                ),
                html.Button(
                    "▶️ Resume",
                    id="resume-button",
                    n_clicks=0,
                    style={"margin-right": "10px", "padding": "10px 15px"},
                ),
                html.Button(
                    "💾 Save Waveforms",
                    id="save-button",
                    n_clicks=0,
                    style={"margin-right": "10px", "padding": "10px 15px"},
                ),
                html.Div(
                    id="save-status", style={"margin-left": "20px", "color": "green"}
                ),
                html.Div(
                    id="performance-stats",
                    style={"margin-top": "10px", "font-size": "14px"},
                ),
            ],
            style={
                "padding": "20px",
                "background-color": "#f8f9fa",
                "border-radius": "5px",
            },
        )
    ]
)

# 创建标签页
tabs = []
for i in range(NUM_TABS):
    start_ch = i * CHANNELS_PER_TAB
    end_ch = (i + 1) * CHANNELS_PER_TAB
    tab_label = f"Ch {start_ch}-{end_ch-1}"

    tabs.append(
        dcc.Tab(
            label=tab_label,
            children=[
                html.Div(
                    [
                        dcc.Graph(
                            id=f"waveform-tab-{i}",
                            figure=create_tab_figure(i),
                            config={"displayModeBar": False},
                            style={"height": "85vh", "width": "100%"},
                        ),
                        dcc.Interval(
                            id=f"update-interval-{i}",
                            interval=UPDATE_INTERVAL,
                            n_intervals=0,
                        ),
                        html.Div(id=f"status-bar-{i}"),
                    ]
                )
            ],
            className="custom-tab",
            selected_className="custom-tab--selected",
        )
    )

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("128-Channel Waveform Monitor (8 Tabs)"),
                controls,
            ],
            style={"margin-bottom": "20px"},
        ),
        dcc.Tabs(
            id="tabs-container",
            children=tabs,
            persistence=True,
            persistence_type="memory",
        ),
    ]
)


# ==================== 回调函数 ====================
@app.callback(
    [Output(f"update-interval-{i}", "interval") for i in range(NUM_TABS)]
    + [Output("save-status", "children"), Output("performance-stats", "children")],
    [
        Input("pause-button", "n_clicks"),
        Input("resume-button", "n_clicks"),
        Input("save-button", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def handle_controls(pause_clicks, resume_clicks, save_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return [no_update] * NUM_TABS + [no_update, no_update]

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "pause-button":
        app_state.set_paused(True)
        logger.info("Monitoring paused")
        return [no_update] * NUM_TABS + ["Monitoring paused", no_update]

    if trigger_id == "resume-button":
        app_state.set_paused(False)
        logger.info("Monitoring resumed")
        return [no_update] * NUM_TABS + ["Monitoring resumed", no_update]

    if trigger_id == "save-button":
        success, filename = save_waveforms()
        if success:
            return [no_update] * NUM_TABS + [
                f"Waveforms saved to {filename}",
                no_update,
            ]
        else:
            return [no_update] * NUM_TABS + [f"Save failed: {filename}", no_update]

    return [no_update] * NUM_TABS + [no_update, no_update]


for i in range(NUM_TABS):

    @app.callback(
        [Output(f"waveform-tab-{i}", "figure"), Output(f"status-bar-{i}", "children")],
        [Input(f"update-interval-{i}", "n_intervals")],
        [State(f"waveform-tab-{i}", "figure")],
        prevent_initial_call=True,
    )
    def update_tab_figure(n_intervals, current_figure, tab_idx=i):
        start_time = time.time()

        if app_state.is_paused():
            return [no_update, "Paused"]

        try:
            start_ch = tab_idx * CHANNELS_PER_TAB
            end_ch = (tab_idx + 1) * CHANNELS_PER_TAB

            if current_figure is None:
                fig = make_subplots(
                    rows=ROWS_PER_TAB,
                    cols=COLS_PER_TAB,
                    vertical_spacing=0.05,
                    horizontal_spacing=0.05,
                )
            else:
                fig = go.Figure(current_figure)

            active_channels = 0

            for ch_idx in range(start_ch, end_ch):
                y_data = channels[ch_idx].get_data()
                if not isinstance(y_data, (list, np.ndarray)):
                    y_data = [y_data] if isinstance(y_data, (int, float)) else []

                x_data = np.arange(len(y_data))

                if len(y_data) > 0 and not all(v == 0 for v in y_data[:10]):
                    active_channels += 1

                row, col = divmod(ch_idx - start_ch, COLS_PER_TAB)
                trace_idx = ch_idx - start_ch

                if trace_idx < len(fig.data):
                    fig.data[trace_idx].x = x_data
                    fig.data[trace_idx].y = y_data
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=x_data,
                            y=y_data,
                            line=dict(width=1, color=COLORS[ch_idx % len(COLORS)]),
                            name=f"CH{ch_idx}",
                        ),
                        row=row + 1,
                        col=col + 1,
                    )

            # 保持原有布局
            if current_figure and "layout" in current_figure:
                fig.update_layout(current_figure["layout"])

            elapsed_time = time.time() - start_time
            app_state.update_performance(elapsed_time)

            stats = app_state.performance_stats
            perf_info = (
                f"Avg: {stats['avg_update_time']*1000:.1f}ms | "
                f"Max: {stats['max_update_time']*1000:.1f}ms | "
                f"Count: {stats['update_count']}"
            )

            status = f"Active: {active_channels}/{CHANNELS_PER_TAB} | Last: {time.strftime('%H:%M:%S')}"
            return [fig, status]

        except Exception as e:
            logger.error(f"Tab {tab_idx} update error: {str(e)}", exc_info=True)
            return [no_update, f"Error: {str(e)}"]


# ==================== 主程序 ====================
def shutdown_handler(sig, frame):
    logger.info("Shutting down...")
    shutdown_event.set()
    os._exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # 初始化通道数据
    channels = [ChannelData() for _ in range(NUM_CHANNELS)]

    # 初始化PV连接
    pv_connections = init_pvs()

    # 启动标签页处理器线程
    tab_processors = []
    for i in range(NUM_TABS):
        processor = TabProcessor(i, channels, shutdown_event)
        processor.start()
        tab_processors.append(processor)

    # 启动资源监控线程
    def monitor_resources():
        while not shutdown_event.is_set():
            time.sleep(10)
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            logger.info(
                f"Memory: {mem.used/1024/1024:.1f}MB used, "
                f"{mem.available/1024/1024:.1f}MB available | "
                f"CPU: {cpu}%"
            )

    threading.Thread(target=monitor_resources, daemon=True).start()

    # 启动Dash应用
    app.run(host="0.0.0.0", port=8050, debug=False, threaded=True)
