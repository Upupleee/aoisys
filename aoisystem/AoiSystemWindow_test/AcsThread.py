from PyQt5.QtCore import QThread, pyqtSignal
import time
from Acs.acs_test.acs_load_buffer_file import run_buffer_from_file

class AcsThread(QThread):
    signal_received_reset = pyqtSignal(str)
    # signal_received_loop = pyqtSignal(int)
    signal_received_finish = pyqtSignal(str)
    signal_received_path = pyqtSignal(str)

    def __init__(self, acs_comm):
        super().__init__()
        self.conn = acs_comm
        self.loop_iterations = None
        # mode 为 'homing' 表示回零操作，'moving' 表示运动操作
        self.operations = []
        self._is_running = True
        self.file_queue = None

    def add_iteration(self, value):
        self.loop_iterations=value

    def add_queue(self, queue):
        self.file_queue = queue        

    def add_operation(self, mode):
        self.operations.append(mode)

    def run(self):
        if self.conn == -1:
            print("无法连接到控制器")
            return
        try:
            for mode in self.operations:
                if self._is_running:
                    if mode == 'homing':
                        run_buffer_from_file(self.conn, 1, "/home/aoi/pretest/acs_test/Buffer_single_new_aoi/x_home_buffer_1.txt")
                        run_buffer_from_file(self.conn, 2, "/home/aoi/pretest/acs_test/Buffer_single_new_aoi/y_home_buffer_2.txt")                                          
                        run_buffer_from_file(self.conn, 3, "/home/aoi/pretest/acs_test/Buffer_single_new_aoi/z_home_buffer_3.txt")
                        run_buffer_from_file(self.conn, 5, "/home/aoi/pretest/acs_test/Buffer_single_new_aoi/k2_home_buffer_5.txt")
                        run_buffer_from_file(self.conn, 4, "/home/aoi/pretest/acs_test/Buffer_single_new_aoi/init_buffer_4.txt")
                        self.signal_received_reset.emit('RESET')
                    elif mode == 'moving':
                        for i in range(self.loop_iterations):
                            print(f"循环第 {i+1} 次")
                            run_buffer_from_file(self.conn, 0, "/home/aoi/pretest/acs_test/Buffer_single_new_aoi/forward_buffer_0.txt")
                            run_buffer_from_file(self.conn, 7, "/home/aoi/pretest/acs_test/Buffer_single_new_aoi/backward_buffer_7.txt")
                            base_path = "/home/aoi/pretest/images/"
                            prefix = "test_imgs"
                            suffix = ".tif"
                            file = f"{base_path}{prefix}{i+1}{suffix}"
                            self.file_queue.put(file)
                            # self.signal_received_loop.emit(i+1)
                            self.signal_received_path.emit(file)
                        run_buffer_from_file(self.conn, 9, "/home/aoi/pretest/acs_test/Buffer_single_new_aoi/reset_k2_buffer_9.txt")
                        self.signal_received_finish.emit('FINISH')
                    
        except Exception as e:
            print(f"线程执行出错：{e}")
        finally:
            self.operations.clear()
            

    def stop(self):
        self._is_running = False
        self.wait()