from pymodbus.client import ModbusTcpClient
from pymodbus.payload import BinaryPayloadDecoder, BinaryPayloadBuilder
from pymodbus.constants import Endian
import time
import struct
# PLC 连接类
class PLCConnection:
    def __init__(self, ip, port=502):
        self.ip = ip
        self.port = port
        self.client = ModbusTcpClient(self.ip, port=self.port)
        self.connect()

    def connect(self):
        while not self.client.connect():
            print(f"Connection error to {self.ip}:{self.port}. Retrying in 5 seconds...")
            time.sleep(5)
        print(f"Connected to PLC at {self.ip}:{self.port}")
        
    def disconnect(self):
        self.client.close()

    def reconnect(self):
        self.client.close()
        self.connect()

    def read_bits(self, address, count):
        try:
            result = self.client.read_coils(address, count)
            if result.isError():
                print(f"Read bits error: {result}")
                self.reconnect()
                return self.read_bits(address, count)
            return result.bits
        except Exception as e:
            print(f"Read bits error: {e}. Reconnecting...")
            self.reconnect()
            return self.read_bits(address, count)

    def write_bit(self, address, value):
        try:
            result = self.client.write_coil(address, value)
            if result.isError():
                print(f"Write bit error: {result}")
                self.reconnect()
                self.write_bit(address, value)
        except Exception as e:
            print(f"Write bit error: {e}. Reconnecting...")
            self.reconnect()
            self.write_bit(address, value)

    def read_registers(self, address, count):
        try:
            result = self.client.read_holding_registers(address, count, slave= 1)
            if result.isError():
                print(f"Read registers error: {result}")
                self.reconnect()
                return self.read_registers(address, count)
            return result.registers
        except Exception as e:
            print(f"Read registers error: {e}. Reconnecting...")
            # self.reconnect()
            # return self.read_registers(address, count)

    def write_register(self, address, value):
        try:
            result = self.client.write_register(address, value)
            if result.isError():
                print(f"Write register error: {result}")
                self.reconnect()
                self.write_register(address, value)
        except Exception as e:
            print(f"Write register error: {e}. Reconnecting...")
            # self.reconnect()
            # self.write_register(address, value)
    
    def read_float(self, address):
        registers = self.read_registers(address, 2)
        decoder = BinaryPayloadDecoder.fromRegisters(registers, byteorder=Endian.BIG, wordorder=Endian.LITTLE)
        return decoder.decode_32bit_float()

    def write_float(self, address, value):
        builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.LITTLE)
        builder.add_32bit_float(value)
        registers = builder.to_registers()
        try:
            result = self.client.write_registers(address, registers)
            if result.isError():
                print(f"Write float error: {result}")
                self.reconnect()
                self.write_float(address, value)
        except Exception as e:
            print(f"Write float error: {e}. Reconnecting...")
            self.reconnect()
            self.write_float(address, value)

# # # 主程序
if __name__ == "__main__":
    plc = PLCConnection('20.0.0.200')
    ready_status = plc.read_bits(151, 1)[0]
    print(ready_status)
#     plc.write_bit(150,0)
#     plc.write_bit(151,1)
    ready_status = plc.read_bits(151, 1)[0]
#     print(ready_status)
# #     print(plc.read_bits(151,1)[0])
#     # plc.write_bit(200,1)
#     # print(plc.read_bits(200,1)[0])
#     plc.write_bit(200,0)
#     print(plc.read_bits(200,1)[0])
#     print(plc.read_bits(202,1)[0])
    # plc.write_bit(256,1)
    # plc.write_bit(255,0) 
#     plc.write_bit(256,0)
#     plc.write_bit(255,1)#吸真空打开
# # plc.write_bit(254, 0)  #CLOSE
# #     # 读取位信号示例：整机上电就绪
# #     ready_status = plc.read_bits(100, 1)
# #     print(f"整机上电就绪状态: {ready_status}")

# # #     # 下发位信号示例：主控软件启动完成
# # #     plc.write_bit(150, True)
# # #     print("已下发主控软件启动完成信号")

# #     # 下发整数信号示例：光源增亮模块 A 通道亮度设置
#     # plc.write_register(151, 10)
# #     print("已下发光源增亮模块 A 通道亮度设置信号")
# #     # 读取整数信号示例：光源增亮模块 A 通道实际亮度
#     time.sleep(2)
#     print(plc.read_float(202))
#     print(plc.read_float(204))
#     a_brightness = plc.read_float(200)
#     print(f"压力表: {a_brightness}")
#     print(plc.read_float(206))
#     plc.write_float(1002,-5) 
#     time.sleep(3)
#     print(plc.read_float(206))
#     time.sleep(3)
#     plc.disconnect()


#     # 读取浮点型信号示例：负压表实时值
#     pressure_value = plc.read_float(200)
#     print(f"负压表实时值: {pressure_value}")

#     # 下发浮点型信号示例：负压表模式
#     plc.write_float(1000, 1.0)
#     print("已下发负压表模式信号")