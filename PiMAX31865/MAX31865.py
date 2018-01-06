# coding:utf-8
"""
Implements the Maxim 31865 resistance to digital converter.
The chip uses SPI to communicate. In this code a rpi is used for SPI by just bit-banging, a better solution would rather
use spidev, but a first try was not successful.

numpy is only required if using the full Callendar–Van Dusen equation.

This code (especially the bit-banging part is based on steve71's work https://github.com/steve71/MAX31865
"""

# system packages
import time
import logging

# rpi specific imports
try:
    import RPi.GPIO as GPIO
except ImportError:
    print("could not import RPi.GPIO")

# coefficients (IEC 751)
A = 3.908e-3
B = -5.775e-7
C = -4.183e-12


# resistance to temperature conversion methods
def c_v_d(r, r_0):
    """
    Callendar–Van Dusen equation.
    Requires numpy.
    :param r: resistance [Ohm]
    :param r_0: resistance at 0 degC [Ohm]
    :return: temperature [degC]
    """
    try:
        import numpy as np
    except ImportError:
        print("could not import numpy")

    coefficients = np.array([r_0 - r,  # T**0
                             r_0 * A,
                             r_0 * B,
                             -100.0 * r_0 * C,
                             r_0 * C])  # T**4

    roots = np.roots(coefficients[::-1])
    theta = roots[-1]  # we need the last one

    # getting rid of the imaginary part (which is zero)
    return abs(theta)


def c_v_d_quad(r, r_0):
    """
    Quadratic approximation of the Callendar–Van Dusen equation.
    The formula is derived by neglecting all parts of the polynomial higher than 2.
    Class A PT100 sensors have a absolute accuracy of 0.2 degC, in which case the quadratic formula is sufficient.
    :param r: resistance [Ohm]
    :param r_0: resistance at 0 degC [Ohm]
    :return: temperature [degC]
    """
    import math

    # only the + part of the quadratic formula
    theta = (-r_0 * A + math.sqrt((r_0 * A) ** 2 - 4.0 * r_0 * B * (r_0 - r))) / (2 * r_0 * B)

    return theta


class FaultError(Exception):
    """
    some custom exception
    """
    pass


class MAX31865(object):
    """
    Reading Temperature from the MAX31865 with GPIO using
    the Raspberry Pi.  Any pins can be used.
    Numpy can be used to completely solve the Callendar-Van Dusen equation
    but it slows the temp reading down.  I commented it out in the code.
    Both the quadratic formula using Callendar-Van Dusen equation (ignoring the
    3rd and 4th degree parts of the polynomial) and the straight line approx.
    temperature is calculated with the quadratic formula one being the most accurate.

    Parameters:
          cs_pin, miso_pin, mosi_pin, clk_pin for the SPI connection, the defaults correspondent to a rpi 3's default SPI pins
          r_0 resistance of the PT at 0 degC (100 for PT100, 1000 for PT1000)
          r_ref is the resistance for the reference resistor in the circuit, r_ref must be greater than r_0 !!
          three_wire when using a 3 wire connection (4 wire = 2 sense leads, 3 wire = 1 sense lead, 2 wire = 0 sense leads)
                    the configuration bits have to be changed. Choose False for 2 wire and 4 wire!
          log is the logger instance, if you don't know what you are doing, leave this alone
    """

    def __init__(self, cs_pin=8, miso_pin=9, mosi_pin=10, clk_pin=11,
                 r_0=1e3, r_ref=4e2, three_wire=False,
                 log=None):

        # logger
        if not log:
            logging.basicConfig(level=logging.DEBUG)
            log = logging.getLogger(__name__)
        self.log = log

        # SPI specific
        self.cs_pin = cs_pin
        self.miso_pin = miso_pin
        self.mosi_pin = mosi_pin
        self.clk_pin = clk_pin
        self.log.debug(
            "CS: {}\tMISO: {}\tMOSI: {}\tCLK: {}".format(self.cs_pin, self.miso_pin, self.mosi_pin, self.clk_pin))

        # setup gpio stuff
        self._setup_gpio()

        # circuit specific
        # value of the reference resistor, defaults to 400 Ohm
        self.r_ref = r_ref
        # PT value at 0 degC, defaults to 1000 Ohm
        self.r_0 = r_0
        self.log.debug("PT: {} Ohm\tR_REF: {} Ohm".format(self.r_0, self.r_ref))

        # chip config
        #
        # b10000000 = 0x80
        # 0x8x to specify 'write register value'
        # 0xx0 to specify 'configuration register'
        #
        # Config Register
        # ---------------
        # bit 7: Vbias -> 1 (ON)
        # bit 6: Conversion Mode -> 0 (MANUAL)
        # bit5: 1-shot ->1 (ON)
        # bit4: 3-wire select -> 0 (3 wire config off by default)
        # bits 3-2: fault detection cycle -> 0 (none)
        # bit 1: fault status clear -> 1 (clear any fault)
        # bit 0: 50/60 Hz filter select -> 0 (60Hz)
        #
        # 0b11000010 or 0xD2 for continuous auto conversion
        # at 60Hz (faster conversion)
        #
        # 0b10100010 is what we use here for one-shot
        self.config = 0b10100010 if not three_wire else 0b10110010
        self.log.debug("3 wire: {}".format(three_wire))

    def _setup_gpio(self):
        self.log.debug("setting up GPIO")
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        self.log.debug("setting up pins")
        GPIO.setup(self.cs_pin, GPIO.OUT)
        GPIO.setup(self.miso_pin, GPIO.IN)
        GPIO.setup(self.mosi_pin, GPIO.OUT)
        GPIO.setup(self.clk_pin, GPIO.OUT)

        GPIO.output(self.cs_pin, GPIO.HIGH)
        GPIO.output(self.clk_pin, GPIO.LOW)
        GPIO.output(self.mosi_pin, GPIO.LOW)

    def _raise_for_fault(self, out):
        """
        10 Mohm resistor is on breakout board to help
        detect cable faults
        bit 7: RTD High Threshold / cable fault open
        bit 6: RTD Low Threshold / cable fault short
        bit 5: REFIN- > 0.85 x VBias -> must be requested
        bit 4: REFIN- < 0.85 x VBias (FORCE- open) -> must be requested
        bit 3: RTDIN- < 0.85 x VBias (FORCE- open) -> must be requested
        bit 2: Overvoltage / undervoltage fault
        bits 1,0 don't care

        :param status:
        :return:
        """

        # get fault thresholds
        [hft_msb, hft_lsb] = [out[3], out[4]]
        hft = ((hft_msb << 8) | hft_lsb) >> 1
        self.log.debug("high fault threshold: {:d}".format(hft))

        [lft_msb, lft_lsb] = [out[5], out[6]]
        lft = ((lft_msb << 8) | lft_lsb) >> 1
        self.log.debug("low fault threshold: {:d}".format(lft))

        # raise if a fault has occured
        status = out[7]
        self.log.debug("Status byte: {:x}".format(status))

        if (status & 0x80) == 1:
            raise FaultError("High threshold limit (Cable fault/open)")

        if (status & 0x40) == 1:
            raise FaultError("Low threshold limit (Cable fault/short)")

        if (status & 0x04) == 1:
            raise FaultError("Overvoltage or Undervoltage Error")

    def read_resistance(self):
        """
        Reads a single resistance value.
        :return: PT resistance [Ohm]
        """
        # write config to register to start a readout
        self.write_register(0, self.config)

        # conversion time is less than 100ms
        time.sleep(.1)  # give it 100ms for conversion

        # read all registers
        out = self.read_register(0, 8)

        conf_reg = out[0]
        self.log.debug("config register byte: {:x}".format(conf_reg))

        # extract MSB and LSB
        [rtd_msb, rtd_lsb] = [out[1], out[2]]

        # calculate ADC code
        rtd_ADC_Code = ((rtd_msb << 8) | rtd_lsb) >> 1
        self.log.debug("RTD ADC Code: {:d}".format(rtd_ADC_Code))

        # calculate resistance
        res_RTD = (rtd_ADC_Code * self.r_ref) / 2 ** 15  # formula see datasheet
        self.log.debug("PT Resistance: {:f} Ohm".format(res_RTD))

        # raise if a fault has occured
        self._raise_for_fault(out)

        return res_RTD

    def read_temp(self, convert=c_v_d_quad):
        """
        Read resistance, convert to temperature with call and return only the temperature
        :param convert: a function which accepts the resistance and the resistance at 0 degC
        :return: temperature [degC]
        """
        return self.read_all(convert)["temperature"]

    def read_all(self, convert=c_v_d_quad):
        """
        Read resistance, convert to temperature with call and return both
        :param convert: a function which accepts the resistance and the resistance at 0 degC
        :return: {"resistance": (resistance, "Ohm"), "temperature": (temperature, "degC")}
        """
        self.log.debug("using conversion function: {}".format(repr(convert)))
        r = self.read_resistance()
        theta = convert(r, self.r_0)

        return {"resistance": (r, "Ohm"), "temperature": (theta, "degC")}

    def write_register(self, regNum, dataByte):
        GPIO.output(self.cs_pin, GPIO.LOW)

        # 0x8x to specify 'write register value'
        addressByte = 0x80 | regNum;

        # first byte is address byte
        self.send_byte(addressByte)
        # the rest are data bytes
        self.send_byte(dataByte)

        GPIO.output(self.cs_pin, GPIO.HIGH)

    def read_register(self, regNumStart, numRegisters):
        out = []
        GPIO.output(self.cs_pin, GPIO.LOW)

        # 0x to specify 'read register value'
        self.send_byte(regNumStart)

        for byte in range(numRegisters):
            data = self.recv_byte()
            out.append(data)

        GPIO.output(self.cs_pin, GPIO.HIGH)
        return out

    def send_byte(self, byte):
        for bit in range(8):
            GPIO.output(self.clk_pin, GPIO.HIGH)
            if (byte & 0x80):
                GPIO.output(self.mosi_pin, GPIO.HIGH)
            else:
                GPIO.output(self.mosi_pin, GPIO.LOW)
            byte <<= 1
            GPIO.output(self.clk_pin, GPIO.LOW)

    def recv_byte(self):
        byte = 0x00
        for bit in range(8):
            GPIO.output(self.clk_pin, GPIO.HIGH)
            byte <<= 1
            if GPIO.input(self.miso_pin):
                byte |= 0x1
            GPIO.output(self.clk_pin, GPIO.LOW)
        return byte


if __name__ == "__main__":
    csPin = [8, 12]
    misoPin = 9
    mosiPin = 10
    clkPin = 11
    sensor = [MAX31865(cs, misoPin, mosiPin, clkPin, 1e2) for cs in csPin]
    try:
        while True:
            for s in sensor:
                print(s.read_all(c_v_d_quad))
    except KeyboardInterrupt:
        print("shutting down")
    finally:
        GPIO.cleanup()
