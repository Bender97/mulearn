# Copyright (c) Daniel Fusaro 2026. All rights reserved.

import runloop
import color_sensor
from hub import port
import motor_pair
motor_pair.pair(motor_pair.PAIR_1, port.C, port.D)


######### PARAMETERS #########
# example colors: background, green, orange, blue
color = "background"
N_samples= 50
move_robot = True
##############################


def get_color():
    (r, g, b, i) = color_sensor.rgbi(port.B)
    i /= 1023
    r = int(r*i)
    g = int(g*i)
    b = int(b*i)
    return r, g, b
    

async def create_dataset(color):
    global N_samples, move_robot
    filename = "/flash/{:s}.csv".format(color)
    f_out = open(filename, "w")
    for i in range(N_samples):
        r, g, b = get_color()
        s = "{:d} {:d} {:d}\n".format(r, g, b)
        f_out.write(s)
        if i % 10 ==0:
            print("{:.2f}% ".format(i/N_samples*100), end="")
        await runloop.sleep_ms(100)
    print("100.00 %")
    print("Done! I stored the dataset for color", color, "here:", filename)
    move_robot = False

async def move():
    global move_robot
    degrees = 180
    while move_robot:
        await motor_pair.move_for_degrees(motor_pair.PAIR_1, degrees, 0)
        await runloop.sleep_ms(100)
        degrees*=-1

runloop.run(move(), create_dataset(color))