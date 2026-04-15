# Copyright (c) Daniel Fusaro. All rights reserved.

from motor import velocity
from utils.model_utils import read_model
from model import SimpleNN
import runloop

from hub import port, motion_sensor
import color_sensor
import motor_pair
motor_pair.pair(motor_pair.PAIR_1, port.C, port.D)

######### PARAMETERS #########
INPUT_SIZE= 3
NUM_CLASSES = 4
MODELNAME= "/flash/simple_nn_model_new.ckpt"
##############################

###### global variables ######
turning = predicting = False
last_color_seen = "background"
vel = 100
##############################

def get_color():
    (r, g, b, i) = color_sensor.rgbi(port.B)
    i /= 1023
    r = int(r*i)
    g = int(g*i)
    b = int(b*i)
    rgb = [r, g, b]
    return rgb

def turn_done():
    return abs(motion_sensor.tilt_angles()[0] * -0.1) > 90

def color_seen():
    global last_color_seen
    return color_valid() and not(last_color_seen=="background")

def color_valid():
    global last_color_seen
    return last_color_seen is not None

async def go_straight():
    global vel
    motor_pair.move(motor_pair.PAIR_1, 0, velocity=vel)
    await runloop.until(color_seen)
    motor_pair.stop(motor_pair.PAIR_1)

async def turn_around(degrees=90):
    global vel
    vel *= -1 ## toggle velocity
    await motor_pair.move_for_degrees(motor_pair.PAIR_1, degrees, 0, velocity=vel)

async def turn_right():
    motion_sensor.reset_yaw(0)
    motor_pair.move(motor_pair.PAIR_1, 100, velocity=50)
    await runloop.until(turn_done)
    motor_pair.stop(motor_pair.PAIR_1)

async def turn_left():
    motion_sensor.reset_yaw(0)
    motor_pair.move(motor_pair.PAIR_1, 100, velocity=-50)
    await runloop.until(turn_done)
    motor_pair.stop(motor_pair.PAIR_1)

async def move():
    global last_color_seen, turning, predicting, velocity    
    
    while True:
        await runloop.until(color_valid)

        if last_color_seen == "background":
            await go_straight()
        
        if last_color_seen is not None and last_color_seen!="background":
            turning=True
            if last_color_seen=="green":
                await turn_around()
            elif last_color_seen=="red":
                await turn_right()
            elif last_color_seen=="blue":
                await turn_left()
            turning = False

            while predicting: # we must wait for the model to predict the color
                runloop.sleep_ms(10)
            last_color_seen=None


async def see():
    # load the model
    model = SimpleNN([INPUT_SIZE, 8, 16, NUM_CLASSES], queue_size=3)
    colors = read_model(model, MODELNAME)

    global last_color_seen, turning, predicting

    while True:
        await runloop.sleep_ms(5)

        if not turning:
            # we start a non-interruptable sequence (predicting=True)
            predicting= True
            last_color_seen = None

            # read from the color sensor
            rgb = get_color()
            
            # use the model to predict the color
            color_idx, confidence = model.predict(rgb)

            if confidence > 0:
                last_color_seen = colors[color_idx]
                print("I saw colow", last_color_seen, \
                    "(with confidence {:.2f}%)".format(confidence))
            predicting=False

#runloop.run(see())
runloop.run(move(), see())

