# Reference: Yahboom Smart official website K210 visual recognition module - AI vision experiment routines (face recognition part)
# My original parts: 1. Modular function encapsulation; 2. UART serial communication (sending recognition results to micro:bit); 3. Face feature hash protection; 4. Optimization of resource release logic
from maix import GPIO, utils, KPU, image, UART
from fpioa_manager import fm
from board import board_info
import time
import gc
import hashlib

FACE_PIC_SIZE = 64
THRESHOLD = 80.5
BOUNCE_DELAY = 50
UART_BAUDRATE = 9600

MODEL_PATHS = {
    "yolo_face": "/sd/KPU/yolo_face_detect/face_detect_320x240.kmodel",
    "ld5": "/sd/KPU/face_recognization/ld5.kmodel",
    "feature_extract": "/sd/KPU/face_recognization/feature_extraction.kmodel"
}

FACE_ID_TO_NUM = {0: 1, 1: 2, 2: 3, 3: 4}
UNREGISTERED_NUM = 0

dst_point = [
    (int(38.2946 * FACE_PIC_SIZE / 112), int(51.6963 * FACE_PIC_SIZE / 112)),
    (int(73.5318 * FACE_PIC_SIZE / 112), int(51.5014 * FACE_PIC_SIZE / 112)),
    (int(56.0252 * FACE_PIC_SIZE / 112), int(71.7366 * FACE_PIC_SIZE / 112)),
    (int(41.5493 * FACE_PIC_SIZE / 112), int(92.3655 * FACE_PIC_SIZE / 112)),
    (int(70.7299 * FACE_PIC_SIZE / 112), int(92.2041 * FACE_PIC_SIZE / 112))
]

anchor = (
    0.1075, 0.126875, 0.126875, 0.175, 0.1465625, 0.2246875, 0.1953125, 0.25375,
    0.2440625, 0.351875, 0.341875, 0.4721875, 0.5078125, 0.6696875, 0.8984375, 1.099687,
    2.129062, 2.425937
)

registered_face_features = []
start_registration = False
recog_flag = False
current_face_id = -1

def init_hardware():
    lcd.init()
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.skip_frames(time=100)
    clock = time.clock()
    
    fm.register(board_info.UART1_TX, fm.fpioa.UART1_TX)
    fm.register(board_info.UART1_RX, fm.fpioa.UART1_RX)
    uart = UART(UART.UART1, UART_BAUDRATE, 8, 0, 1, timeout=1000)
    
    print("Hardware initialization completed")
    return lcd, sensor, clock, uart

def init_boot_key():
    fm.register(board_info.BOOT_KEY, fm.fpioa.GPIOHS0)
    key_gpio = GPIO(GPIO.GPIOHS0, GPIO.IN)
    
    def key_callback(*_):
        global start_registration
        start_registration = True
        time.sleep_ms(BOUNCE_DELAY)
    
    key_gpio.irq(key_callback, GPIO.IRQ_RISING, GPIO.WAKEUP_NOT_SUPPORT)
    return key_gpio

def load_ai_models():
    yolo_kpu = KPU()
    yolo_kpu.load_kmodel(MODEL_PATHS["yolo_face"])
    yolo_kpu.init_yolo2(
        anchor, anchor_num=9, img_w=320, img_h=240, net_w=320, net_h=240,
        layer_w=10, layer_h=8, threshold=0.7, nms_value=0.2, classes=1
    )
    
    ld5_kpu = KPU()
    print("Preparing to load model")
    ld5_kpu.load_kmodel(MODEL_PATHS["ld5"])
    
    fea_kpu = KPU()
    print("Preparing to load model")
    fea_kpu.load_kmodel(MODEL_PATHS["feature_extract"])
    
    return yolo_kpu, ld5_kpu, fea_kpu

def expand_face_region(x, y, w, h, scale):
    x1_t = x - scale * w
    x2_t = x + w + scale * w
    y1_t = y - scale * h
    y2_t = y + h + scale * h
    x1 = int(x1_t) if x1_t > 1 else 1
    x2 = int(x2_t) if x2_t < 320 else 319
    y1 = int(y1_t) if y1_t > 1 else 1
    y2 = int(y2_t) if y2_t < 240 else 239
    cut_img_w = x2 - x1 + 1
    cut_img_h = y2 - y1 + 1
    return x1, y1, cut_img_w, cut_img_h

def hash_face_feature(feature):
    return hashlib.sha256(bytes(feature)).hexdigest()

def map_face_id_to_num(face_id):
    return FACE_ID_TO_NUM.get(face_id, UNREGISTERED_NUM)

def send_num_to_microbit(uart, num):
    try:
        uart.write(bytes(f"{num}\n", "utf-8"))
    except Exception as e:
        print(f"UART send failed: {e}")

def main():
    lcd, sensor, clock, uart = init_hardware()
    key_gpio = init_boot_key()
    yolo_kpu, ld5_kpu, fea_kpu = load_ai_models()
    
    feature_img = image.Image(size=(FACE_PIC_SIZE, FACE_PIC_SIZE), copy_to_fb=False)
    feature_img.pix_to_ai()

    print("System started, waiting for face detection...")

    while True:
        gc.collect()
        clock.tick()
        img = sensor.snapshot()
        
        yolo_kpu.run_with_output(img)
        dect = yolo_kpu.regionlayer_yolo2()
        fps = clock.fps()

        if len(dect) > 0:
            for l in dect:
                x1, y1, cut_img_w, cut_img_h = expand_face_region(l[0], l[1], l[2], l[3], scale=0)
                face_cut = img.cut(x1, y1, cut_img_w, cut_img_h)
                face_cut_128 = face_cut.resize(128, 128)
                face_cut_128.pix_to_ai()

                out = ld5_kpu.run_with_output(face_cut_128, getlist=True)
                face_key_point = []
                for j in range(5):
                    x = int(KPU.sigmoid(out[2 * j]) * cut_img_w + x1)
                    y = int(KPU.sigmoid(out[2 * j + 1]) * cut_img_h + y1)
                    face_key_point.append((x, y))
                
                T = image.get_affine_transform(face_key_point, dst_point)
                image.warp_affine_ai(img, feature_img, T)
                feature = fea_kpu.run_with_output(feature_img, get_feature=True)
                del face_key_point

                scores = []
                for j in range(len(registered_face_features)):
                    score = yolo_kpu.feature_compare(registered_face_features[j][1], feature)
                    scores.append(score)
                
                if len(scores):
                    max_score = max(scores)
                    index = scores.index(max_score)
                    if max_score > THRESHOLD:
                        img.draw_string(0, 195, "person:%d,score:%2.1f" % (index, max_score), color=(0, 255, 0), scale=2)
                        recog_flag = True
                        current_face_id = index
                    else:
                        img.draw_string(0, 195, "unregistered,score:%2.1f" % (max_score), color=(255, 0, 0), scale=2)
                        recog_flag = False
                        current_face_id = -1
                else:
                    img.draw_string(0, 195, "unregistered,score:0.0", color=(255, 0, 0), scale=2)
                    recog_flag = False
                    current_face_id = -1
                
                del scores

                if start_registration:
                    feature_hash = hash_face_feature(feature)
                    registered_face_features.append((feature_hash, feature))
                    print(f"Registration successful! Current number of faces: {len(registered_face_features)}, Feature hash: {feature_hash[:10]}...")
                    start_registration = False

                if recog_flag:
                    img.draw_rectangle(l[0], l[1], l[2], l[3], color=(0, 255, 0))
                    recog_flag = False
                else:
                    img.draw_rectangle(l[0], l[1], l[2], l[3], color=(255, 255, 255))

                del face_cut_128
                del face_cut

        img.draw_string(0, 0, "%2.1ffps" % (fps), color=(0, 60, 255), scale=2.0)
        img.draw_string(0, 215, "press boot key to register face", color=(255, 100, 0), scale=2.0)
        lcd.display(img)

        current_num = map_face_id_to_num(current_face_id)
        send_num_to_microbit(uart, current_num)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Program exception: {e}")
    finally:
        yolo_kpu.deinit()
        ld5_kpu.deinit()
        fea_kpu.deinit()
        lcd.clear()
        uart.deinit()
        print("Resources released")

