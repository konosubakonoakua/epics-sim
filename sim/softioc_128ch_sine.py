from softioc import softioc, builder, asyncio_dispatcher
import numpy as np
import asyncio
import time
import sys
import logging
from collections import defaultdict
from datetime import datetime

# 配置参数
NUM_CHANNELS = 128  # 通道数量
SAMPLE_RATE = 128000  # 采样率(Hz)
BUFFER_SIZE = 2000  # 每个波形点数
BASE_FREQ = 2  # 基础频率(Hz)
UPDATE_INTERVAL = 0.5  # 更新间隔(秒)
NOISE_AMPLITUDE = 0.01  # 高斯噪声幅度
LOG_FILE = "waveform_generator.log"  # 日志文件路径

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("WaveformGenerator")


class WaveformGenerator:
    def __init__(self):
        self.pv_objects = []
        self.t_elapsed = 0
        self.pv_freq_map = defaultdict(float)
        self._running = True
        self.start_time = datetime.now()

        logger.info("波形生成器初始化完成")
        logger.info(f"配置参数: 通道数={NUM_CHANNELS}, 采样率={SAMPLE_RATE}Hz")
        logger.info(f"基础频率={BASE_FREQ}Hz, 噪声幅度={NOISE_AMPLITUDE}")

    def create_pvs(self):
        """创建所有PV（必须在SetDeviceName之后调用）"""
        logger.info("开始创建PV...")
        try:
            for i in range(NUM_CHANNELS):
                pv_name = f"wf{i}"
                pv = builder.WaveformIn(pv_name, length=BUFFER_SIZE)
                self.pv_objects.append(pv)
                freq = BASE_FREQ * (i + 1)
                self.pv_freq_map[pv_name] = freq
                logger.debug(f"创建PV: {pv_name}, 频率={freq}Hz")

            logger.info(f"成功创建{NUM_CHANNELS}个波形PV")
        except Exception as e:
            logger.error(f"创建PV时出错: {str(e)}")
            raise

    def generate_noisy_sine_wave(self, freq, n_samples):
        """生成带高斯噪声的正弦波"""
        try:
            t = np.linspace(
                self.t_elapsed,
                self.t_elapsed + n_samples / SAMPLE_RATE,
                n_samples,
                endpoint=False,
            )
            clean_wave = np.sin(2 * np.pi * freq * t)
            noise = np.random.normal(0, NOISE_AMPLITUDE, n_samples)

            # 记录噪声统计信息
            noise_stats = {
                "mean": float(np.mean(noise)),
                "std": float(np.std(noise)),
                "min": float(np.min(noise)),
                "max": float(np.max(noise)),
            }
            logger.debug(f"生成波形: 频率={freq}Hz, 噪声统计={noise_stats}")

            return clean_wave + noise
        except Exception as e:
            logger.error(f"生成波形时出错: {str(e)}")
            raise

    async def update_all_waveforms(self):
        """持续更新所有PV波形"""
        update_count = 0
        logger.info("开始波形更新循环...")

        while self._running:
            try:
                start_time = time.perf_counter()
                timestamp = time.time()
                update_count += 1

                for pv in self.pv_objects:
                    pv_base_name = pv.name.split(":")[-1] if ":" in pv.name else pv.name
                    freq = self.pv_freq_map[pv_base_name]
                    waveform = self.generate_noisy_sine_wave(freq, BUFFER_SIZE)
                    pv.set(waveform, timestamp=timestamp)

                self.t_elapsed += UPDATE_INTERVAL
                elapsed = time.perf_counter() - start_time

                # 每10次更新记录一次性能
                if update_count % 10 == 0:
                    logger.info(
                        f"更新周期 #{update_count}, "
                        f"耗时={elapsed*1000:.2f}ms, "
                        f"累计运行时间={(datetime.now()-self.start_time).total_seconds():.1f}s"
                    )

                await asyncio.sleep(max(0, UPDATE_INTERVAL - elapsed))

            except Exception as e:
                logger.error(f"波形更新时出错: {str(e)}", exc_info=True)
                await asyncio.sleep(1)  # 出错时暂停1秒

    def stop(self):
        """停止波形更新"""
        self._running = False
        run_time = (datetime.now() - self.start_time).total_seconds()
        logger.info(
            f"停止波形生成器，总运行时间={run_time:.2f}秒，"
            f"累计更新次数={int(run_time/UPDATE_INTERVAL)}"
        )


async def run_ioc():
    """运行IOC的主协程"""
    # 设置设备名称
    builder.SetDeviceName("wf128")

    # 初始化波形生成器
    wf_gen = WaveformGenerator()
    wf_gen.create_pvs()

    # 初始化IOC
    builder.LoadDatabase()
    dispatcher = asyncio_dispatcher.AsyncioDispatcher()
    softioc.iocInit(dispatcher)

    # 启动波形更新任务
    update_task = asyncio.create_task(wf_gen.update_all_waveforms())

    try:
        # 替代interactive_ioc的解决方案
        print("IOC已启动，按Ctrl+C停止...")
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        wf_gen.stop()
        await update_task


def main():
    # Windows事件循环策略
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 创建新的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(run_ioc())
    except KeyboardInterrupt:
        print("\n程序正常终止")
    finally:
        loop.close()
        asyncio.set_event_loop(None)


if __name__ == "__main__":
    try:
        logger.info("=" * 50)
        logger.info("启动波形生成器 IOC")
        logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 50)

        main()
    except Exception as e:
        logger.critical(f"程序致命错误: {str(e)}", exc_info=True)
    finally:
        logger.info("=" * 50)
        logger.info("波形生成器 IOC 已关闭")
        logger.info("=" * 50)
