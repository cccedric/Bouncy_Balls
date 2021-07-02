#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import functools
import argparse
import importlib
import sys
import os

import config
import ballsim


if __name__ == "__main__":

    if config.profile:
        import cProfile
    import matplotlib.pyplot as plt

    ## Parse arguments to script
    parser = argparse.ArgumentParser(description="Run a ball simulation.")
    parser.add_argument("--numballs", type=int,   metavar="50",  help="Number of balls.",       default=config.numballs, required=False)
    parser.add_argument("--simdur",   type=float, metavar="simulation_duration", help="Duration of simulation (milliseconds).", default=config.simdur, required=False)
    parser.add_argument("--tcol",     type=float, metavar="1.0", help="Float value for `tcol` (time between collision checks).", default=config.tcol, required=False)
    parser.add_argument("--cor",      type=float, metavar="CoR", help="Coefficient of restitution. (~sqrt(energy loss factor); 1 = elastic).", default=config.cor, required=False)
    parser.add_argument("--seed",     type=int,   metavar="123", help="Random seed for NumPy.", default=None,            required=False)
    parser.add_argument("--physfile", type=str,   metavar=".../physics.py", help="Path to `physics.py` file to import.", default=None, required=False)
    parser.add_argument("--rmin",     type=float, default=config.ballradmin, required=False)
    parser.add_argument("--rmax",     type=float, default=config.ballradmax, required=False)
    parser.add_argument("-x",         action="store_true")

    args = parser.parse_args()
    numballs        = args.numballs
    simdur          = args.simdur
    tcol            = args.tcol
    cor             = args.cor
    random_seed     = args.seed
    physfile        = args.physfile
    rmin            = args.rmin
    rmax            = args.rmax
    suppress_output = args.x
    ############################

    # Import physics.py
    if physfile is None:
        import physics
    else:
        dirname = os.path.dirname(physfile)
        fnname  = os.path.basename(physfile)
        if fnname[-3:] == ".py":
            fnname = fnname[:-3]
        sys.path.insert(0, dirname)
        physics = importlib.import_module(fnname)
        sys.path.pop(0)

    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)

    # Configure output mode
    liveplot = config.liveplot
    saveanim = config.saveanim
    if suppress_output:
        liveplot=False
        saveanim=False

    assert rmin <= rmax, "Minimum ball radius is greater than maximum ball radius. Check settings and command-line options."

    # Create box dimensions and randomise ball states
    boxdims = [[config.x_min, config.x_max], [config.y_min, config.y_max]]
    balls   = ballsim.randomstate(boxdims, num=numballs, radmin=rmin, radmax=rmax)

    # Run the simulation
    def go():
        ballsim.animate(
                balls, 
                simdur,
                physics.accel_fun,
                functools.partial(physics.handle_wall_collision, boxdims, cor),
                physics.collision_matrix_fun,
                functools.partial(physics.handle_ball_collision, cor),
                boxdims=boxdims,
                tcol=tcol,
                liveplot=liveplot, 
                save_animation=saveanim,
                fig_larger_dim=config.fig_larger_dim, 
                pillowWriter=config.pillowWriter, 
                use_pyqtgraph=config.use_pyqtgraph, 
                fps=config.fps, 
                ball_color=(0.5,0.9,0.5), 
                auto_decimate=config.auto_decimate, 
                accel_fun_separable=config.accel_fun_separable
        )
    if config.profile:
        if config.try_use_yappi:
            try:
                import yappi
                using_yappi = True
            except ModuleNotFoundError:
                print("The `yappi` module was not found. Falling back to `cProfile`.")
                using_yappi = False
        else:
            using_yappi = False
        if using_yappi:
            yappi.start()
            go()
            func_stats = yappi.get_func_stats()
            func_stats.save('callgrind.out', 'CALLGRIND')
            yappi.stop()
            yappi.clear_stats()
        else:
            import cProfile
            cProfile.run('go()',sort='cumtime')
    else:
        go()
    #plt.show()
