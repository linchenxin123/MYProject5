from microbit import *
import utime

UART_BAUDRATE = 9600
TIMEOUT = 300

NUM_LED_PATTERNS = {
    0: [
        (0,1,1,1,0),
        (1,0,0,0,1),
        (1,0,0,0,1),
        (1,0,0,0,1),
        (0,1,1,1,0)
    ],
    1: [
        (0,0,1,0,0),
        (0,1,1,0,0),
        (0,0,1,0,0),
        (0,0,1,0,0),
        (0,1,1,1,0)
    ],
    2: [
        (0,1,1,1,0),
        (1,0,0,0,1),
        (0,0,1,0,0),
        (0,1,0,0,0),
        (1,1,1,1,0)
    ],
    3: [
        (0,1,1,1,0),
        (1,0,0,0,1),
        (0,0,1,0,0),
        (1,0,0,0,1),
        (0,1,1,1,0)
    ],
    4: [
        (0,0,0,1,0),
        (0,0,1,1,0),
        (0,1,0,1,0),
        (1,1,1,1,1),
        (0,0,0,1,0)
    ]
}

def init_uart():
    uart.init(baudrate=UART_BAUDRATE, bits=8, parity=None, stop=1, timeout=TIMEOUT)
    display.show("R")
    utime.sleep(1000)
    display.clear()
    return uart

def display_led_number(num):
    pattern = NUM_LED_PATTERNS.get(num, NUM_LED_PATTERNS[0])
    display.clear()
    for y in range(5):
        for x in range(5):
            if pattern[y][x] == 1:
                display.set_pixel(x, y, 9)
            else:
                display.set_pixel(x, y, 0)
    utime.sleep_ms(100)

def main():
    uart = init_uart()
    while True:
        try:
            if uart.any():
                data = uart.readline().decode("utf-8").strip()
                if data.isdigit():
                    num = int(data)
                    display_led_number(num)
                else:
                    display_led_number(0)
            else:
                display_led_number(0)
            utime.sleep_ms(200)
        except Exception as e:
            print(f"Error: {e}")
            display_led_number(0)
            utime.sleep_ms(500)

if __name__ == "__main__":
main()
