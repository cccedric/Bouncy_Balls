#!/usr/bin/env python

import sys
import time
import datetime
import multiprocessing
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as pltan
from dataclasses import dataclass
from typing import List
import math
import functools
import scipy.integrate
import imageio


def start_plot_pyqtgraph(boxdims):

    ((x_min, x_max), (y_min, y_max)) = boxdims

    from PyQt5 import QtWidgets
    from pyqt_graph import MainWindow, BallDataWorker
    qtApp = QtWidgets.QApplication(sys.argv)
    qtWin = MainWindow()
    worker = BallDataWorker()
    qtWin.connect_ball_worker_thread(worker)
    worker.start()

    return qtApp, qtWin, worker


def start_plot_matplotlib(boxdims, fig_larger_dim):
    ((x_min, x_max), (y_min, y_max)) = boxdims
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range > y_range:
        figsize = (fig_larger_dim, fig_larger_dim * y_range / x_range)
    else:
        figsize = (fig_larger_dim * x_range / y_range, fig_larger_dim)

    plt.figure(figsize=figsize)
    ax = plt.axes(label='basic_plot')
    return ax


def randomstate(boxdims, radmin=0.2, radmax=1, num=30):

    ((x_min, x_max), (y_min, y_max)) = boxdims
    balls = np.zeros((num, 5), dtype=np.float32)

    # Randomise radii
    balls[:, 0] = np.random.rand(num)*(radmax - radmin) + radmin
    # Randomise x
    balls[:, 1] = x_min + balls[:, 0] + \
        (np.random.rand(num) * (x_max - balls[:, 0] - (x_min + balls[:, 0])))
    # Randomise y
    balls[:, 2] = y_min + balls[:, 0] + \
        (np.random.rand(num) * (y_max - balls[:, 0] - (y_min + balls[:, 0])))
    # Randomise x,y vel
    balls[:, 3:5] = np.random.randn(num, 2) * 5

    return balls


def draw_bg(boxdims):
    ((x_min, x_max), (y_min, y_max)) = boxdims
    plt.axhline(y=y_min, xmin=x_min, xmax=x_max, color='black')
    plt.axhline(y=y_max, xmin=x_min, xmax=x_max, color='black')
    plt.axvline(x=x_min, ymin=y_min, ymax=y_max, color='black')
    plt.axvline(x=x_max, ymin=y_min, ymax=y_max, color='black')


def draw_balls(balls, ax, ball_color):
    for ball in balls:
        c = ball_color  # RGBA from 0 to 1
        cp = patches.Circle((ball[1], ball[2]), radius=ball[0], color=c)
        ax.add_artist(cp)


def update_balls(balls, plottitle, ax=None, qtWin=None, worker=None, use_pyqtgraph=False):

    if use_pyqtgraph:
        #qtWin.update_plot(balls, plottitle)
        worker.update_ball_data(balls, plottitle)
    else:
        ax.set_title(plottitle)
        for ind in range(len(ax.artists)):
            b = balls[ind]
            ax.artists[ind].center = b[1:3]
            ax.artists[ind].radius = b[0]
        fig = plt.gcf()
        fig.canvas.draw()
        fig.canvas.flush_events()
        # plt.pause(0.00000000001)


def draw_everything(balls, boxdims, show, ball_color, ax=None, qtWin=None, use_pyqtgraph=False):
    ((x_min, x_max), (y_min, y_max)) = boxdims
    if use_pyqtgraph:
        if show:
            qtWin.setVisible(True)
    else:
        ax.clear()
        plt.axis('off')
        ax.set_aspect(1)
        ax.set_xlim([x_min-0.5, x_max+0.5])
        ax.set_ylim([y_min-0.5, y_max+0.5])

        draw_bg(boxdims)
        draw_balls(balls, ax, ball_color)
        if show:
            plt.show(block=False)


def decimated_collision_matrix_fun(boxdims, collision_matrix_fun, balls):

    ((x_min, x_max), (y_min, y_max)) = boxdims

    numballs = len(balls)
    posarray = balls[:, 1:3]
    radii = balls[:, 0]
    assert numballs == posarray.shape[0], "Array of radii and posarray do not have a matching shape."

    if numballs < 500:
        return collision_matrix_fun(balls)

    # Decimating region
    maxrad = max(radii)
    w = x_max - x_min
    h = y_max - y_min
    wmid = x_min + (w/2)
    hmid = y_min + (h/2)

    num_diameters_across = 7
    mxnum = int(w / (maxrad*2*num_diameters_across)) + 1
    mynum = int(h / (maxrad*2*num_diameters_across)) + 1
    mw = w / mxnum
    mh = h / mynum

    ret = np.zeros((numballs, numballs), dtype=bool)
    for i, j in itertools.product(range(mxnum), range(mynum)):
        ball_idx_in_region = np.argwhere(
            np.logical_and(
                np.abs(posarray[:, 0] - (x_min + (i+1/2)*mw)
                       ) <= (mw/2 + maxrad),
                np.abs(posarray[:, 1] - (y_min + (j+1/2)*mh)
                       ) <= (mh/2 + maxrad)
            )
        ).flatten()
        m = ball_idx_in_region  # Shorthand 'm' = mask
        scol = collision_matrix_fun(balls[m])
        l = scol.shape[0]

        for bi in range(l):
            for bj in range(bi+1, l):
                if scol[bi, bj]:
                    ret[m[bi], m[bj]] = True
    return ret


def advance_shard(rk45_accel_fun, sim_time, ms, balls_shard):
    shap = balls_shard.shape
    # RK4 integration
    obj = scipy.integrate.RK45(rk45_accel_fun, (sim_time)/1000,
                               balls_shard.flatten(), (sim_time + ms)/1000, atol=0.0001)
    while obj.status != "finished":
        obj.step()
    balls_shard = np.reshape(obj.y, shap)
    return balls_shard


def flattened_accel_fun(accel_fun, t, balls_flat):
    shap = (balls_flat.size//5, 5)
    return accel_fun(t, balls_flat.reshape(shap)).flatten()


def advance(balls, process_pool_numproc, sim_time, ms, accel_fun, collision_matrix_fun, auto_decimate_boxdims=(False, None), accel_fun_separable=False):

    # Unpack process pool and number of processes
    process_pool, numproc = process_pool_numproc

    rk45_accel_fun = functools.partial(flattened_accel_fun, accel_fun)
    if (numproc == 1) or not accel_fun_separable:
        balls = advance_shard(rk45_accel_fun, sim_time, ms, balls)
    else:
        ball_shards = np.array_split(balls, numproc)
        mapfun = functools.partial(advance_shard, rk45_accel_fun, sim_time, ms)
        balls = np.concatenate(
            process_pool.map(mapfun, ball_shards)
        )

    # Recomputing the possible collision matrix
    auto_decimate, boxdims = auto_decimate_boxdims
    if auto_decimate:
        balls_possible_col = decimated_collision_matrix_fun(
            boxdims, collision_matrix_fun, balls)
    else:
        balls_possible_col = collision_matrix_fun(balls)

    sim_time += ms

    return sim_time, balls, balls_possible_col


def animate(balls, duration_ms, accel_fun, handle_wall_collision, collision_matrix_fun, handle_ball_collision, boxdims=[[0, 30], [0, 70]], tcol=30, fps=25, save_animation=False, fig_larger_dim=10, liveplot=True, pillowWriter=True, use_pyqtgraph=True, ball_color='orange', auto_decimate=True, accel_fun_separable=False):

    ((x_min, x_max), (y_min, y_max)) = boxdims

    pillowWriter = pillowWriter and (not use_pyqtgraph)
    ax = None                                   # Set if not use_pyqtgraph
    qtApp, qtWin, worker = None, None, None     # Set if use_pyqtgraph

    if liveplot or save_animation:
        if use_pyqtgraph:
            qtApp, qtWin, worker = start_plot_pyqtgraph(boxdims)
        else:
            ax = start_plot_matplotlib(boxdims, fig_larger_dim)
        if use_pyqtgraph:
            qtWin.setup_plot(100*fig_larger_dim, x_min, x_max,
                             y_min, y_max, ball_color=ball_color)
        draw_everything(balls, boxdims, liveplot, ball_color,
                        ax=ax, qtWin=qtWin, use_pyqtgraph=use_pyqtgraph)
        update_balls(balls, 'blank_title', ax=ax, qtWin=qtWin,
                     worker=worker, use_pyqtgraph=use_pyqtgraph)
    #ms_between_draws = 1000./fps if fps > 20 else 25
    ms_between_draws = 1000./25       # REALTIME framerate for plotting.
    if pillowWriter:
        fps = 25  # matplotlib.animate.MovieWriter is faulty. Will only export at 25fps
    ms_between_framesaves = 1000./fps  # SIMTIME framerate for saving frames.
    ms_between_collchecks = tcol

    min_advance = tcol/100  # milliseconds
    simtime = 0
    start_time = datetime.datetime.now()
    def realtime(): return (datetime.datetime.now()-start_time).total_seconds()*1000

    #ms_since_last_draw = lambda : realtime() - ms_last_draw

    ms_last_draw = 0
    ms_last_framesave = 0
    ms_last_collcheck = 0

    if save_animation:
        # Initialise movie saving object
        if pillowWriter:
            moviewriter = pltan.PillowWriter(fps=fps, bitrate=-1)
            moviewriter.setup(plt.gcf(), 'anim.gif', dpi=100)
            moviewriter.grab_frame()
        else:
            if use_pyqtgraph:
                writer = imageio.get_writer('anim.mp4', fps=fps)
                writer.append_data(qtWin.export())
            else:
                #moviewriter = pltan.ImageMagickWriter(fps=fps, bitrate=-1)
                moviewriter = pltan.FFMpegWriter(fps=fps, bitrate=-1)
                moviewriter.setup(plt.gcf(), 'anim.mp4', dpi=100)
                moviewriter.grab_frame()

    def time_to_collcheck(): return ms_between_collchecks - \
        (simtime - ms_last_collcheck)
    def time_to_saveframe(): return ms_between_framesaves - \
        (simtime - ms_last_framesave)
    #time_to_drawframe = lambda : ms_between_draws - (rt - ms_last_draw)
    def time_to_drawframe(): return ms_between_draws - (simtime - ms_last_draw)

    # Create process pool for RK45 integration
    # Creating new threads is expensive. Number of processes should scale slowly with number of balls.
    # When numproc = 1, other parts of the program will see this and not spawn a separate process but use the main thread.
    numballs = len(balls)
    numproc = max(1, multiprocessing.cpu_count() - 1)
    numproc = min(numproc, int((numballs*ms_between_collchecks)/1000)+1)
    process_pool = multiprocessing.Pool(processes=numproc)

    if numproc > 1:
        print("Using multiple processes (produces a lot of overhead): ", numproc)
        print()

    rt = realtime()
    while ((rt < duration_ms) and liveplot) or (simtime < duration_ms):
        rt = realtime()

        saveframe = time_to_saveframe() <= 0 and save_animation
        drawframe = time_to_drawframe() <= 0 and liveplot
        if drawframe or (saveframe and save_animation):
            # Debugging
            if False:
                print(rt)
                print("Drawframe: ", drawframe)
                print("Saveframe: ", saveframe)
                print("Save_anim: ", save_animation)
                print()
            update_balls(balls,
                         "Sim time: %.2fs; Real time: %.2fs" % (
                             simtime/1000, rt/1000),
                         ax=ax, qtWin=qtWin, worker=worker,
                         use_pyqtgraph=use_pyqtgraph
                         )
            if drawframe:
                ms_last_draw = rt
            if saveframe:
                ms_last_framesave = simtime
                if use_pyqtgraph:
                    writer.append_data(qtWin.export())
                else:
                    moviewriter.grab_frame()

        sim_leadtime = simtime - rt
        if sim_leadtime < 0 or not liveplot:
            nextsimduration = time_to_collcheck()
            if liveplot:
                nextsimduration = np.minimum(nextsimduration, -sim_leadtime)
                nextsimduration = np.minimum(
                    nextsimduration, time_to_drawframe())
            if save_animation:
                nextsimduration = np.minimum(
                    nextsimduration, time_to_saveframe())
            nextsimduration = np.maximum(nextsimduration, min_advance)
            nextsimduration = np.minimum(
                nextsimduration, duration_ms - simtime)

            #old_balls = balls
            # For debugging
            if False:
                print("Simtime:  ", simtime)
                print("Position: ", balls[0, 2])
                print("Velocity: ", balls[0, 4])
            simtime, balls, balls_possible_col = advance(
                balls,
                (process_pool, numproc),
                simtime,
                nextsimduration,
                accel_fun,
                collision_matrix_fun,
                auto_decimate_boxdims=(auto_decimate, boxdims),
                accel_fun_separable=accel_fun_separable
            )
            if not liveplot:
                sys.stdout.write(
                    "Simtime: {0:.1f}ms, realtime: {1:.1f}ms        \r".format(simtime, rt))

            collcheck = time_to_collcheck() <= 0
            if collcheck:
                handle_wall_collision(balls)
                numballs = len(balls)
                # Loop over all balls and call collision method.
                col_idxs = np.argwhere(balls_possible_col)
                numcol_handles = len(col_idxs)
                if numproc > 1 and numcol_handles > 1000:  # Number of collision handlings
                    ball_pairs = [[balls[i1], balls[i2]]
                                  for i1, i2 in col_idxs]
                    process_pool.starmap(handle_ball_collision, ball_pairs)
                else:
                    # Single-threaded approach:
                    for ind1, ind2 in col_idxs:
                        handle_ball_collision(balls[ind1], balls[ind2])

                ###
                ms_last_collcheck = simtime
    process_pool.close()
    end_time = datetime.datetime.now()

    if save_animation:
        # This takes a little while
        if use_pyqtgraph:
            writer.close()
        else:
            moviewriter.finish()
            # moviewriter.cleanup()

    if use_pyqtgraph and (liveplot or save_animation):
        worker.exit(0)

    ms_duration = (end_time - start_time).total_seconds()*1000
    print("Realtime duration of animation (ms): ", ms_duration)
    print("Time passed in simulation (ms): ", simtime)
    #print("First ball position: ", balls[0].state.pos)


def test():
    return cProfile.run('basketball.animate(5)', 'cumtime')
