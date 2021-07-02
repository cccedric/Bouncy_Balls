#!/usr/bin/env python

import numpy as np


def accel_fun(t, balls):
    """
    Returns the rate-of-change to apply to the ball data array (radius, x and y, xvel and yvel) as a function of the current simulation time and current ball data array.

    This method is passed to `ballsim.animate` to determine the dynamics of the system.

    E.g. if there are two balls, then returning the following array: np.array([[0, 1, 0, 0, 0],
                                                                                [2, 0, 0, 0, 0]])
    will result in the time rate-of-change of the first ball's x position to be 1 (it moves at 1 meter per second) and the 2nd ball will not move but its radius will increase by 2 meters per second.

    :param t:           Current time of simulation (seconds).
    :param balls:       A np array of `dtype` `np.float32` and shape `(N,5)` where `N` is the number of balls.
                        `balls[:,0]` is a (N,) array of the ball radii.
                        `balls[:,1:3]` is a (N,2) array of the ball x/y positions.
                        `balls[:,3:5]` is a (N,2) array of the ball x/y velocities.
    :returns:           An array indicating the rate of change of the ball radii, positions and velocities. Same shape as the `balls` input array.
    :rtype:             A `np.ndarray` of shape `(N,5)`.
    """

    # RoT = rate of change
    RoT = np.zeros_like(balls)

    # Edit RoT here...
    RoT[:, 1:3] = balls[:, 3:5]
    RoT[:, 4] = -9.81
    return RoT


def handle_wall_collision(boxdims, cor, balls):
    """
    Return a new ball data array describing balls after collisions with walls are handled.

    :param boxdims: Enumerable like ((xmin, xmax), (ymin, ymax)) containing the bounds of the bounding box for the simulation.
    :param cor:     Float describing the coefficient of restitution (from 0 to 1).
    :param balls:   numpy.ndarray of ball objects, shape (N,5). 0th axis indexes the ball. 1st axis indexes: ball radius; x,y position; x,y velocity.
    :returns:       New balls array.

    :type balls:    numpy.ndarray of shape (N,5). 
    :rtype:         numpy.ndarray of shape (N,5).
    """

    # Extracting the min/max x/y coordinates of the box from the input variable `boxdims`
    ((minx, maxx), (miny, maxy)) = boxdims

    # Defining a (N,) array of ball radii
    rads = balls[:, 0]

    ## bbl = balls_bounce_left
    ## bbr = balls_bounce_right
    ## bbt = balls_bounce_top
    ## bbb = balls_bounce_botton
    # Handle collisions for all four walls using reflection
    bbl_mask = balls[:, 1] < (minx + rads)
    balls[bbl_mask, 1] += 2 * (minx + balls[bbl_mask, 0] - balls[bbl_mask, 1])
    balls[bbl_mask, 3] *= -cor

    bbr_mask = balls[:, 1] > (maxx - rads)
    balls[bbr_mask, 1] -= 2 * \
        (balls[bbr_mask, 1] - (maxx - balls[bbr_mask, 0]))
    balls[bbr_mask, 3] *= -cor

    bbt_mask = balls[:, 2] > (maxy - rads)
    balls[bbt_mask, 2] -= 2 * \
        (balls[bbt_mask, 2] - (maxy - balls[bbt_mask, 0]))
    balls[bbt_mask, 4] *= -cor

    bbb_mask = balls[:, 2] < (miny + rads)
    balls[bbb_mask, 2] += 2 * (miny + balls[bbb_mask, 0] - balls[bbb_mask, 2])
    balls[bbb_mask, 4] *= -cor

    return balls


def collision_matrix_fun(balls):
    """
    Produce a square boolean matrix describing which balls collide with which others.

    For the resulting matrix M, `handle_ball_collision(balls[i],balls[j])` will be called if and only if M_ij is True.
    Consequently, the matrix can be upper/lower triangular if the `handle_ball_collision` method is implemented accordingly.

    :param balls:       An (N,5) NumPy array containing the ball data.
    :returns:           An (N,N) boolean array corresponding to the ball_i, ball_j pairs on which `handle_ball_collision` will be called.

    :type balls:        np.ndarray
    :rtype:             np.ndarray
    """

    radii = balls[:, 0]   # Array of shape (N,)
    relradii = radii.reshape(1, radii.shape[0]).repeat(radii.shape[0], axis=0)
    relradii += relradii.T
    posarray = balls[:, 1:3]  # Array of shape (N,2)
    # Array of shape (N,N,2). relposarray[i,j,:] is an array of shape (2,) containing the relative x/y position of balls i and j.
    relposarray = posarray - posarray[:, np.newaxis]

    col_matrix = np.zeros((balls.shape[0],)*2, dtype=bool)

    # Check for collisions between every pair of balls and modify `col_matrix` ...
    col_matrix = np.linalg.norm(
        relposarray[:, :, :], axis=2, keepdims=False) < relradii
    col_matrix_self = np.eye(radii.shape[0], dtype=bool)
    col_matrix ^= col_matrix_self
    # If col_matrix[i,j] is True, then col_matrix[j,i] should be false. This is to avoid duplicate collision checks.
    col_matrix = np.triu(col_matrix)

    return col_matrix


def handle_ball_collision(cor, b1, b2):
    """
    Apply corrections to ball objects to handle collisions between them. 

    This method is called according to the output of `collision_matrix` which determines (quickly!) which balls collide.
    This method will be applied to ball data every so often as part of the `ballsim.animate` method, according to the `tcol` argument.
    Collision handling can be time consuming, so you need to make sure you implement a solution that isn't highly inefficient.

    This method should return a tuple (b1, b2) of the new ball arrays, each of shape (5,), describing the new state of the two balls after interacting.

    :param b1:      NumPy array of shape (5,) describing the first  ball (radius, x, y, xvel, yvel)
    :param b2:      NumPy array of shape (5,) describing the second ball (radius, x, y, xvel, yvel)
    :type b1:       np.ndarray
    :type b2:       np.ndarray
    :returns:       Tuple (b1, b2) of new ball arrays after the two balls collide.
    :rtype:         np.ndarray
    """

    # Array of shape (2,) containing relative x/y position.
    relp = b2[1:3] - b1[1:3]
    sumrad = b1[0] + b2[0]
    rpnorm = np.linalg.norm(relp)   # The length of the vector `relp`.

    # Handle collisions between b1 and b2...
    m1 = np.pi * b1[0] ** 2
    m2 = np.pi * b2[0] ** 2
    Delta = (1 + cor) / 2 * np.dot(2 * (b1[3:5] - b2[3:5]),
                                   (relp / rpnorm)) * (relp / rpnorm) / (1 / m1 + 1 / m2)

    # Change b1 and b2 (each a (5,) NumPy array)...
    b1[1:3] -= b1[0] / sumrad * (sumrad - rpnorm) / 2 * (relp / rpnorm)
    b2[1:3] += b2[0] / sumrad * (sumrad - rpnorm) / 2 * (relp / rpnorm)
    b1[3:5] -= Delta / m1
    b2[3:5] += Delta / m2

    return b1, b2
