
import numpy as np

# STUDENT TODO
# ADJUST THESE SETTINGS AS DESIRED

### CONFIG ###
# Simulation conditions
# Coefficient of Restitution. Roughly the square root of energy loss per collision.
cor = 0.5
numballs = 30  # \
ballradmin = 0.2      # |--Randomly generate `numballs` with radii betwween `ballradmin` and `ballradmax`.
ballradmax = 0.5  # /
simdur = 2000    # Simulation duration in milliseconds.
# Milliseconds between collision checks. Physics inaccurate above ~1ms.
tcol = 1

# Box dimensions
x_min = 0
x_max = 10
y_min = 0
y_max = 5
#############

### ADVANCED CONFIG ###

# Profiling options
profile = False
try_use_yappi = True     # Only applies if profile is True.

# Optimisations
auto_decimate = False
accel_fun_separable = False

# Visualisation controls
# Visual drawing size of the longer dimension of the box. Multiply by 100 for pixel width of the larger edge.
fig_larger_dim = 10
# Turn off if you experience problems with image export or complaints about requiring FFMPEG.
use_pyqtgraph = True
# May as well be True if `saveanim` is True; liveplotting happens at 25fps in parallel with framesaving.
liveplot = True
# Writes the resulting animation into ./anim.mp4 at `fps` frames per second if pillowWriter is False, otherwise ./anim.gif at 25 frames per second.
saveanim = True
# Framerate of the video saved (if saveanim and (use_pyqtgraph or not pillowWriter)).
fps = 60
# Whether to use PillowWriter in order to save the animation (option only has an effect if saveanim is True and not use_pyqtgraph).
pillowWriter = True
# Leave this option as True unless you have FFMPEG installed. If True, then PillowWriter will be used
# which requires no external dependencies but is quite slow and produces a GIF (yuck).
# PillowWriter can only save the GIF at 25 fps; the fps option above will be ignored.
