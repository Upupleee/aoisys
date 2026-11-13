from pymodbus.client import ModbusTcpClient
from pymodbus.payload import BinaryPayloadDecoder, BinaryPayloadBuilder
from pymodbus.constants import Endian
import time
import ipdb

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
            result = self.client.read_holding_registers(address, count,slave=1)
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
        decoder = BinaryPayloadDecoder.fromRegisters(registers, byteorder=Endian.BIG, wordorder=Endian.BIG)
        return decoder.decode_32bit_float()

    def write_float(self, address, value):
        builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
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

# # 主程序
if __name__ == "__main__":
    #  
    plc = PLCConnection('20.0.0.200')

    # light=plc.read_bits(200,3)
    # print(light)
    # stop = plc.read_bits(202,2)
    # print(stop)

    # ##################################
    print(plc.read_bits(200,1))
    # plc.write_bit(150,0)
    # print(plc.read_bits(150,1))
    plc.write_bit(151,1)
    maincontrol_reset = plc.read_bits(151,1)
    print(maincontrol_reset)
    ##################################

    ##################################
    #
    # a=plc.read_registers(150,2)
    # print(a)
    # plc.write_register(150, 0)
    # b=plc.read_registers(150,2)
    # print(b)
    # plc.write_register(151, 1)
    # c=plc.read_registers(151,2)
    # print(c)
    ##################################

    ##################################
    ##################################
    # lightcomm = plc.read_bits(111,2)
    # print(lightcomm)
    # lightcomm = plc.read_bits(110,2)
    # print(lightcomm)
    ##################################
    ##################################


    ##################################
    #LIGHT CURTAIN TRIGGER, CALL THIS AFTERWARDS.
    #plc.write_bit(151,1) #DOES NOT WORK YET
    ##################################

    # 读取位信号示例：整机上电就绪
    # ready_status = plc.read_bits(100, 1)[0]
    # print(f"整机上电就绪状态: {ready_status}")

#     # 下发位信号示例：主控软件启动完成
#     plc.write_bit(150, True)
#     print("已下发主控软件启动完成信号")

    # 下发整数信号示例：光源增亮模块 A 通道亮度设置
    #ipdb.set_trace
    #plc.write_register(150, 999)
    #plc.write_register(150, 1)
    #print("已下发光源增亮模块 A 通道亮度设置信号")
#     # 读取整数信号示例：光源增亮模块 A 通道实际亮度
    #a_brightness = plc.read_registers(101, 3)
    # a_brightness = plc.read_registers(100, 3)
    # print(f"光源增亮模块 A 通道实际亮度: {a_brightness}")

    # #plc.write_bit(258, 1)   #  on power for platform

    ##########################################
    ##########################################
    #FIRST OPEN BLOWER,THEN TAKE VACCUM EFFECT,AFTER FINISHING VACCUM EFFECT,SWITCH OFF VACCUM AND CLOSE BLOWER 
    #plc.write_register(254, 1)
    # __import__('ipdb').set_trace()
    # plc.write_bit(254, 1)
    # #plc.write_bit(254, 0)
    # #plc.write_register(254, 1)
    # plc.write_bit(255, 1)  #VACUUM
    # plc.write_bit(255, 0)  #BREAK VACUUM
    # plc.write_bit(254, 0)  #CLOSE
    ###########################################
    ###########################################


    # ###########################################
    # #plc.write_register(254, 1)
    # plc.write_bit(250, 1)   #GREEN LIGHT
    # plc.write_bit(250, 0)
    # ###########################################

    # ###########################################
    # #plc.write_register(254, 1)
    # plc.write_bit(251, 1)   #YELLOW LIGHT
    # plc.write_bit(251, 0)
    # ###########################################

    # ###########################################
    # #plc.write_register(254, 1)
    # plc.write_bit(252, 1)   #RED LIGHT
    # plc.write_bit(252, 0)
    # ###########################################

    # ###########################################
    # #plc.write_register(254, 1)
    # plc.write_bit(253, 1)   #BUZZ
    # plc.write_bit(253, 0)
    # ###########################################


    # ###########################################
    # #plc.write_register(254, 1)
    # plc.write_bit(256, 1)   # BREAK VACCUM
    # plc.write_bit(256, 0)
    # ###########################################

    # ###########################################
    # #plc.write_register(254, 1)
    # plc.write_bit(258, 1)   # ON POWER FOR PLATFORM 
    # plc.write_bit(258, 0)   # POWER OFF FOR PLATFORM
    # ###########################################


#     # 读取浮点型信号示例：负压表实时值
#     pressure_value = plc.read_float(200)
#     print(f"负压表实时值: {pressure_value}")

#     # 下发浮点型信号示例：负压表模式
#     plc.write_float(1000, 1.0)
#     print("已下发负压表模式信号")
