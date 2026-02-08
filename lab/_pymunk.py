"""
Created on Fri Feb 06 23:41:01 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path

import sys
import pygame
import pymunk
import pymunk.pygame_util
from pygame.color import THECOLORS

def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("joints. just wait, and the L will tip over")
    clock = pygame.time.Clock()

    space = pymunk.Space()
    space.gravity = 0, 980

    lines = add_L(space)
    balls = []
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    ticks_to_next_ball = 10
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit(0)

        space.step(1/50)

        ticks_to_next_ball -= 1
        if ticks_to_next_ball <= 0:
            ticks_to_next_ball = 20
            ball_shape = add_ball(space)
            balls.append(ball_shape)

        screen.fill((255, 255, 255))

        balls_to_remove = []
        for ball in balls:
            if ball.body.position.y > 550:
                balls_to_remove.append(ball)
        for ball in balls_to_remove:
            space.remove(ball, ball.body)
            balls.remove(ball)

        space.debug_draw(draw_options)
        pygame.display.flip()
        clock.tick(50)

def add_ball(space):
    mass = 3
    radius = 25
    body = pymunk.Body()
    import random
    x = random.randint(20, 500)
    body.position = x, 50
    shape = pymunk.Circle(body, radius)
    shape.mass = mass
    shape.friction = 1
    space.add(body, shape)
    return shape

def draw_ball(screen, ball):
    p = int(ball.body.position.x), int(ball.body.position.y)
    pygame.draw.circle(screen, (0, 0, 255), p, int(ball.radius), 2)

def add_static_L(space):
    body = pymunk.Body(body_type = pymunk.Body.STATIC)
    body.position = (300, 300)
    l1 = pymunk.Segment(body, (-200, 0), (200, 0), 2)
    l2 = pymunk.Segment(body, (-200, 0), (-200, -50), 2)
    l1.friction = 1
    l2.friction = 2
    space.add(body, l1, l2)
    return l1, l2

def add_L(space):
    rotation_center_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    rotation_center_body.position = (300, 300)

    rotation_limit_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    rotation_limit_body.position = (200, 300)

    body = pymunk.Body()
    body.position = (300, 300)
    l1 = pymunk.Segment(body, (-150, 0), (255, 0), 2)
    l2 = pymunk.Segment(body, (-150, 0), (-150, -50), 2)
    l1.friction = 1
    l2.friction = 2
    l1.mass = 8
    l2.mass = 1
    rotation_center_joint = pymunk.PinJoint(body, rotation_center_body, (0, 0), (0, 0))
    joint_limit = 20
    rotation_limit_joint = pymunk.SlideJoint(body, rotation_limit_body, (-150, 0), (-50, 0), 0, joint_limit)
    space.add(body, l1, l2, rotation_center_joint, rotation_limit_joint)
    return l1, l2

def draw_lines(screen, lines):
    for line in lines:
        body = line.body
        pv1 = body.position + line.a.rotated(body.angle)
        pv2 = body.position + line.b.rotated(body.angle)
        p1 = to_pygame(pv1)
        p2 = to_pygame(pv2)
        pygame.draw.lines(screen, THECOLORS["lightgray"], False, [p1, p2])

def to_pygame(p):
    return round(p.x), round(p.y)




if __name__ == "__main__":
    print("---- run ----")
    sys.exit(main())

