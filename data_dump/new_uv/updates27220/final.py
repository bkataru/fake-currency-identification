from tflite_runtime.interpreter import Interpreter
from PIL import Image
from pygame import mixer
import numpy as np
import os, ast, re
import cv2
import RPi.GPIO as GPIO
import time

AUDIO_DIR = 'lang_audio/'
LANG = 'English'

mixer.init()

print("Loading GPIO configuration")
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
bl = 20
el = 21
fl = 2
ts = 3
GPIO.setup(bl, GPIO.OUT)
GPIO.setup(el, GPIO.OUT)
GPIO.setup(fl, GPIO.OUT)
# GPIO.output(el,1)
# GPIO.output(bl,1)
# GPIO.output(fl,0)
GPIO.setup(ts, GPIO.IN, pull_up_down = GPIO.PUD_DOWN)
print("GPIO configuration loaded")

def map(x, in_min, in_max, out_min, out_max):
    return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)

def apply_brightness_contrast(input_img):
    bright = 185
    contr = 187
    brightness = map(bright, 0, 510, -255, 255)
    contrast = map(contr, 0, 254, -127, 127)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf

def get_labels(path):	
	f = open(path, "r")
	labels = f.read()
	labels = ast.literal_eval(labels)

	final_labels = {v: k for k, v in labels.items()}
	
	return final_labels
  
def predict_image(frame, final_labels, interpreter, input_details, output_details):
    test_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(test_image, (224, 224), cv2.INTER_AREA)
    test_image = np.expand_dims(test_image, axis=0)
    
    test_image = (2.0 / 255.0) * test_image - 1.0
    test_image = test_image.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], test_image)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])

    result_dict = dict()
    for key in list(final_labels.keys()):
        result_dict[final_labels[key]] = result[0][key]
    sorted_results = {k: v for k, v in sorted(result_dict.items(), key=lambda item: item[1], reverse=True)}

    for label in sorted_results.keys():
        print("{}: {}%".format(label, sorted_results[label] * 100))

    final_result = dict()
    final_result[list(sorted_results.keys())[0]] = sorted_results[list(sorted_results.keys())[0]] * 100

    return final_result

print("Loading classification interpreter")
classify_interpreter = Interpreter(model_path="converted_currency_mobilenetmodel.tflite")
print("Loaded classification interpreter")
print("Loading watermark interpreter")
watermark_interpreter = Interpreter(model_path="converted_watermark_mobilenetmodel.tflite")
print("Loaded watermark interpreter")
print("Loading UV interpreter")
uv_interpreter = Interpreter(model_path="converted_uv_mobilenetmodel.tflite")
print("Loaded UV interpreter")

classify_input_details = classify_interpreter.get_input_details()
classify_output_details = classify_interpreter.get_output_details()
watermark_input_details = watermark_interpreter.get_input_details()
watermark_output_details = watermark_interpreter.get_output_details()
uv_input_details = uv_interpreter.get_input_details()
uv_output_details = uv_interpreter.get_output_details()

print("Allocating classification model tensors...")
classify_interpreter.allocate_tensors()
print("Allocated classification model tensors...")
print("Allocating watermark model tensors...")
watermark_interpreter.allocate_tensors()
print("Allocated watermark model tensors...")
print("Allocating UV model tensors...")
uv_interpreter.allocate_tensors()
print("Allocated UV model tensors...")
print("All models loaded")

print("Loading labels...")
currency_labels = get_labels('mobilenet_currency_class_indices.txt')
uv_labels = get_labels('mobilenet_uv_class_indices.txt')
watermark_labels = get_labels('mobilenet_watermark_class_indices.txt')
print("Labels loaded")

def main():
    GPIO.output(fl, 1)
    time.sleep(0.3)

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    effect = apply_brightness_contrast(frame)
    cv2.imwrite("latestframe_classify.jpg", effect)
    
    res = predict_image(effect, currency_labels, classify_interpreter,
    classify_input_details, classify_output_details)
    
    cap.release()
    time.sleep(0.1)
    print("Frontlight Frame read")
    GPIO.output(fl, 0)
    
    print("=" * 50)
    print(res)
    
    note = list(res.keys())[0].split('_')[:2]
    note = "_".join(note)
    
    audio_file = AUDIO_DIR + LANG + '/' + note + '.mp3'
    print(audio_file)
    mixer.music.load(audio_file)
    mixer.music.play()
    
    if note == 'invalid':
        return
    
    cap.release()
    
    print('=' * 50)
    cap = cv2.VideoCapture(0)
    GPIO.output(bl,0)
    GPIO.output(el,0)
    time.sleep(1.2)
    
    ret, frame = cap.read()
    
    time.sleep(0.5)
    print("Watermark frame read")
    GPIO.output(el,1)
    
    effect_wm = apply_brightness_contrast(frame)
    cv2.imwrite("latestframe_watermark.jpg", effect_wm)
    
    # import random
    # cv2.imwrite("watermark_dataset/{}.jpg".format(random.randint(0, 10000000)), effect_wm)
    
    res_wm = predict_image(effect_wm, watermark_labels, watermark_interpreter,
	watermark_input_details, watermark_output_details)
    print(res_wm)
	
    wm = list(res_wm.keys())[0]
    if wm == 'no_watermark':
        beep_file = AUDIO_DIR + 'fakebeep.mp3'
        mixer.music.load(beep_file)
        mixer.music.play()
	
        while mixer.music.get_busy() == True:
            continue
	
        audio_file = AUDIO_DIR + LANG + '/fake.mp3'
        mixer.music.load(audio_file)
        mixer.music.play()
	
        return
	
    beep_file = AUDIO_DIR + 'realbeep.mp3'
    mixer.music.load(beep_file)
    mixer.music.play()
    
    # while mixer.music.get_busy() == True:
    #     continue
	    
    notename = list(res.keys())[0]
    target_strip = ''
    print("random", notename)
    if 'old' in notename:
        if 'back' in notename:
            if int(notename.split('_')[0]) in [100]:
                target_strip = 'continuous'
            else:
                audio_file = AUDIO_DIR + LANG + '/flip.mp3'
                mixer.music.load(audio_file)
                mixer.music.play()
	    
                return
        elif 'front' in notename:
            target_strip = 'dashed'
	    
    elif 'new' in notename:
        if 'front' in notename:
            if int(notename.split('_')[0]) in [10, 20, 50]:
                target_strip = 'dashed'
            if int(notename.split('_')[0]) in [100, 200, 500, 2000]:
                audio_file = AUDIO_DIR + LANG + '/flip.mp3'
                mixer.music.load(audio_file)
                mixer.music.play()
	
                return
        elif 'back' in notename:
            if int(notename.split('_')[0]) in [10, 20]:
                audio_file = AUDIO_DIR + LANG + '/flip.mp3'
                mixer.music.load(audio_file)
                mixer.music.play()
	
                return	
            if int(notename.split('_')[0]) in [50, 100, 200, 500, 2000]:
                target_strip = 'continuous'
		
    cap.release()
    print('=' * 50)
    
    cap = cv2.VideoCapture(0)
    GPIO.output(bl,1)
    GPIO.output(el,0)
    time.sleep(1.2)
    cap.release()
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    time.sleep(0.3)
    print("UV frame read")
    GPIO.output(el,1)
    
    # cv2.imwrite("latestframe_uv.jpg", frame)
    effect_uv = apply_brightness_contrast(frame)
    cv2.imwrite("latestframe_uv.jpg", effect_uv)
    
    import random
    cv2.imwrite("uv_dataset/{}.jpg".format(random.randint(0, 10000000)), effect_uv)
    
    
    res_uv = predict_image(effect_uv, uv_labels, uv_interpreter,
	uv_input_details, uv_output_details)
    print(res_uv)
	
    uv_strip = list(res_uv.keys())[0]
    
    if uv_strip == target_strip:
        beep_file = AUDIO_DIR + 'realbeep.mp3'
        mixer.music.load(beep_file)
        mixer.music.play()

        while mixer.music.get_busy() == True:
            continue
	
        audio_file = AUDIO_DIR + LANG + '/real.mp3'
        mixer.music.load(audio_file)
        mixer.music.play()
    else:
        if note == '50_new':
            audio_file = AUDIO_DIR + LANG + '/flip.mp3'
            mixer.music.load(audio_file)
            mixer.music.play()
        else:
            beep_file = AUDIO_DIR + 'fakebeep.mp3'
            mixer.music.load(beep_file)
            mixer.music.play()
	    
            while mixer.music.get_busy() == True:
                continue

            audio_file = AUDIO_DIR + LANG + '/fake.mp3'
            mixer.music.load(audio_file)
            mixer.music.play()
	
	    	    
print("=" * 50)
if __name__ == "__main__":
    print("System ready")
    while True:
        GPIO.wait_for_edge(ts, GPIO.RISING)
        main()
    
    GPIO.cleanup()
