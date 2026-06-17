#!/usr/bin/env 


# Utils.py has various utility functions that will be reused both
# in recording the data to train with and the playing of the game of 
# our bot.
# Takes screenshots, has the input mapping, etc.
# For playing, see play.py. For recording (i.e getting the data) see record.py


import sys
import array
import pygame
import wx
from PIL import ImageGrab
from PIL import Image
import numpy as np

from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread
from skimage.util import img_as_float

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

wx.App()


# Class to define our screenshots taken both for training and for playing.
# We have a src dimensions and destination IMG dimensions. This lets us
# more easily decide if we need to resize to help in training/data gathering.
# For now, I will have them be the same as an easy start but it is an option to change
# due to either hardware or software limitations.
class Screenshot:
	SCR_W = 640
	SCR_H = 480
	SRC_D = 4

	OFFSET_X = 0
	OFFSET_Y = 0

	IMG_W = 640
	IMG_H = 480
	IMG_D = 3

	image_array = array.array('B', [0] * (SRC_W * SRC_H * SRC_D));


# Our controller class. Our own custom controller to more easily adapt to whatever
# we have connected and let us set our own tolerances and mapping from this controller
# to one Dolphin will play nicer with (Dolphin can use most controller, but this helps us
# orgainze it for later pipeing easily)

# Depending on the type of controller we use, we have to map our internal
# button variables to different axis, buttons, and hats. 
# see https://www.pygame.org/docs/ref/joystick.html for uses

# For my personal use case, it'll be easiest to use Dual Shock 4/PS4

class Controller:
	def __init__(self):
		try:
			pygame.init()
			self.joystick = pygame.joystick.joystick(1)
			self.joystick.init()
		except:
			print 'unable to connect to Controller'

	# Read will return the current controller state as an array of the buttons and stick values.
	# Return order = [mainStickX, mainStickY, cStickX, cStickY, b, a, y, x, start, d-pad Up, Down, Left, Right, Z, Left_trigger, right_trigger]
	def read(self):
		pygame.event.pump()
		mainX = self.joystick.get_axis(0); # main stick
		mainY = self.joystick.get_axis(1); # main stick

		# right stick which we will treat as the C-stick
		cX = self.joystick.get_axis(2);
		cY = self.joystick.get_axis(3);

		# Mapping will be:
		# b == Square
		# a == X/Cross
		# y == Triangle
		# x == Circle
		# Start == Options
		# D-pad == Dpad
		# left trigger == Left trigger 
		# right trigger == Right Trigger
		# z == Right Bumper

		b = self.joystick.get_button(2);
		a = self.joystick.get_button(0);
		y = self.joystick.get_button(3);
		x = self.joystick.get_button(1);

		start = self.joystick.get_button(6);
		d_pad_up = self.joystick.get_button(11);
		d_pad_down = self.joystick.get_button(12);
		d_pad_left = self.joystick.get_button(13);
		d_pad_right = self.joystick.get_button(14);

		z = self.joystick.get_button(10);

		# Triggers are axises on ps4 controller for pygame, so we'll say maybe 3/4 press counts?
		left_trigger = self.joystick.get_axis(4)
		right_trigger = self.joystick.get_axis(5)



		# There are more buttons on ps4 than on gamecube, oh well we leave it alone.


		# we have some normalization to do if we want to translate the above from pygame
		# to our options for Dolphin.
		# Dolphin Joysticks are [0, 1] with middle being 0.5, whereas pygame is [-1, 1] with 0 being center.
		# Trigger is a simple button press in Dolphin so convert it to a button press value

		# pygame get button are all Bools, the axis return floats from [-1, 1], and if
		# we need to use hats, they are -1, 0, 1 (but thankfully not needed for my controller setup).

		# this should change from [-1, 1] to [0, 1] and center from 0 to 0.5
		x = 0.5 + (x/2.0)
        y = 0.5 + (y/2.0)

		# Trigger axis is out -> in so -1 out, 1 is in with 0 being halfway pressed. We will test to confirm this 
		# but that is what to documentation implies.

		# Output the triggers as booleans by just changing them to Bools based on current values.
		left_trigger = (left_trigger > 0.5)
		right_trigger = (right_trigger > 0.5)

		# we have it, time to output it as a list. Output as our sticks first, then all the buttons in order above as they are declared and assigned.

		return [mainX, mainY, cX, cY, b, a, y, x, start, d_pad_up, d_pad_down, d_pad_left, d_pad_right, z, left_trigger, right_trigger];

	def manual_override(self):
        pygame.event.pump()
        return self.joystick.get_button(4) == 1


# Data class is used for transforming our images and controller inputs into a Data set to train the model on.
# For now, I've copied what I've used before but I want to revist this methodolgy for PyTorch to make sure
# it's more correct and just in generally more workable.
class Data:
  	def __init__(self):
     		self._X = np.load("data/X.npy")
    	    self._y = np.load("data/y.npy")
    		self._epochs_completed = 0
     		self._index_in_epoch = 0
     		self._num_examples = self._X.shape[0]

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]
	def load_sample(sample):
   		image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,))
   		joystick_values = np.loadtxt(sample + '/data.csv', delimiter=',', usecols=(1,2,3,4,5))
   		return image_files, joystick_values
		# training data viewer
	def viewer(sample):
   		image_files, joystick_values = load_sample(sample)

    	plotData = []

    	plt.ion()
    	plt.figure('viewer', figsize=(16, 6))

   		for i in range(len(image_files)):

        	# joystick
        	print i, " ", joystick_values[i,:]

        	# format data
    		plotData.append( joystick_values[i,:] )
        	if len(plotData) > 30:
        	    plotData.pop(0)
       		x = np.asarray(plotData)

        # image (every 3rd)
        if (i % 3 == 0):
            plt.subplot(121)
            image_file = image_files[i]
            img = mpimg.imread(image_file)
            plt.imshow(img)

        # plot
        plt.subplot(122)
        plt.plot(range(i,i+len(plotData)), x[:,0], 'r')
        plt.hold(True)
        plt.plot(range(i,i+len(plotData)), x[:,1], 'b')
        plt.plot(range(i,i+len(plotData)), x[:,2], 'g')
        plt.plot(range(i,i+len(plotData)), x[:,3], 'k')
        plt.plot(range(i,i+len(plotData)), x[:,4], 'y')
        plt.draw()
        plt.hold(False)

        plt.pause(0.0001) # seconds
        i += 1

	# prepare training data
	def prepare(samples):
    	print "Preparing data"

    	X = []
    	y = []

    	for sample in samples:
        	print sample

        	# load sample
        	image_files, joystick_values = load_sample(sample)

        	# add joystick values to y
        	y.append(joystick_values)

        	# load, prepare and add images to X
        	for image_file in image_files:
        	    image = imread(image_file)
         	   vec = prepare_image(image)
        	   X.append(vec)

    	print "Saving to file..."
    	X = np.asarray(X)
    	y = np.concatenate(y)

    	np.save("data/X", X)
    	np.save("data/y", y)

    	print "Done!"
    	return

	if __name__ == '__main__':
    if sys.argv[1] == 'viewer':
        viewer(sys.argv[2])
    elif sys.argv[1] == 'prepare':
        prepare(sys.argv[2:])
	


