#!/usr/bin/env python
from PyQt5 import QtWidgets
#from pyqtgraph import PlotWidget, plot
#from PQG_ImageExporter import PQG_ImageExporter
import pyqtgraph as pg
import pyqtgraph.exporters

from pyqtgraph.Qt import QtGui, QtCore, QtSvg, QT_LIB
from pyqtgraph import functions as fn

import sys  # We need sys so that we can pass argv to QApplication
import os
import numpy as np

class BallDataWorker(QtCore.QThread):
    signal = QtCore.pyqtSignal(object, object, bool)

    def __init__(self):
        super().__init__()

    def update_ball_data(self, balls, plottitle, export=False):
        self.signal.emit(balls, plottitle, export)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.invisiblePen = pg.mkPen(color=(255,255,255,0), width=0)
        self.penWall = pg.mkPen(color=(0,0,0), width=5)

        self.graphWidget = pg.PlotWidget()
        self.pi = self.graphWidget.getPlotItem()
        self.setCentralWidget(self.graphWidget)

        self.exporter = pg.exporters.ImageExporter(self.pi)
        # Doesn't solve the issue
        #self.exporter = PQG_ImageExporter(self.pi)
        self.graphWidget.setAntialiasing(True)

        # 1:1 aspect ratio
        self.graphWidget.setAspectLocked(self.pi)
        # White background (default is black)
        self.graphWidget.setBackground('w')
        # Prevent manipulation of the plot with the mouse
        self.graphWidget.setMouseEnabled(False,False)
        # Disable autoranging of plot
        self.graphWidget.disableAutoRange(axis=None)
        # Hide "A" button in bottom-left
        self.pi.hideButtons()
        # Hide axes
        [self.pi.hideAxis(s) for s in ['left','right','top','bottom']]
        # Draw only those points whose center is strictly inside the plotting range
        # Do not use! It seems to hide everything anyway, but inconsistently!
        #self.pi.setClipToView(True)
        # Set plot labels : 'title', 'left', 'bottom', 'right', 'top'
        # Disable context menu
        self.pi.setMenuEnabled(enableMenu=False, enableViewBoxMenu='same')

    def setup_plot(self,fig_larger_dim_pix,xmin,xmax,ymin,ymax,ball_color=(1,0.8,0.5)):
        self.symbColor = tuple(int(n*255) for n in ball_color)
        self.penBall   = pg.mkPen(color=ball_color, width=0)
        xr = xmax - xmin
        yr = ymax - ymin
        if yr > xr: 
            h = fig_larger_dim_pix 
            w = xr / yr * h
        else:
            w = fig_larger_dim_pix
            h = yr / xr * w
        h += 50
        w, h = int(w), int(h)
        w = int(np.ceil(w/16)*16)
        h = int(np.ceil(h/16)*16)
        self.setFixedSize(w,h)

        # Fixes another bug in pyqtgraph 0.10.0 where the ImageExporter doesn't export properly
        # until resize events are processed. Or something. This forces it to do that.
        self.setVisible(not self.isVisible())
        self.setVisible(not self.isVisible())

        self.exporter = pg.exporters.ImageExporter(self.pi)
        # 
        self.exporter.params.param('width').setValue(0, blockSignal=self.exporter.widthChanged)
        self.exporter.params.param('height').setValue(0, blockSignal=self.exporter.heightChanged)
        self.exporter.params.param('width').setValue(w, blockSignal=self.exporter.widthChanged)
        self.exporter.params.param('height').setValue(h, blockSignal=self.exporter.heightChanged)

        # From a GitHub bug thread: https://github.com/pyqtgraph/pyqtgraph/issues/538
        # Affects pyqtgraph version 0.10.0
        #I use the following contruct to circumvent the problem:
        #exporter.params.param('width').setValue(1920, blockSignal=exporter.widthChanged)
        #exporter.params.param('height').setValue(1080, blockSignal=exporter.heightChanged)
        #This way I don't have to modify internal code. Beware, if you set the values to the default values (640, 480) the variables will not update their type to int!

        self.graphWidget.setRange(padding=0,update=True,xRange=(xmin,xmax),yRange=(ymin,ymax))
        # Messes with aspect ratio
        #self.graphWidget.setLimits(xMin=xmin,xMax=xmax,yMin=ymin,yMax=ymax)

    def connect_ball_worker_thread(self, worker : BallDataWorker):
        worker.signal.connect(self.update_plot_async)

    @QtCore.pyqtSlot(object,object,bool)
    def update_plot_async(self, balls, plottitle, export):
        self.update_plot(balls, plottitle, export)

    def update_plot(self, balls, plottitle, export=False):

        ballradarray = balls[:,0]
        ballposarray = balls[:,1:3]
        x,y = np.transpose(ballposarray)
        self.pi.clear()
        self.pi.setLabels(title=plottitle)

        xmin, xmax = self.pi.getAxis('bottom').range
        rmult = 2 * self.width() / (xmax - xmin)

        self.pi.plot(x,y, pen=self.invisiblePen, symbolBrush=self.symbColor, symbolPen=self.symbColor,
                symbolSize=[r*rmult for r in ballradarray]
        )

        pg.QtGui.QApplication.processEvents()
        #self.exporter.export(png_name,toBytes=True)
        #b = self.exporter.export(toBytes=True)
        if export : return self.export()
        return

    def export(self):
        w = int(self.exporter.params['width'])
        h = int(self.exporter.params['height'])
        if w == 0 or h == 0:
            raise Exception("Cannot export image with size=0 (requested "
                            "export size is %dx%d)" % (w, h))
        
        targetRect = QtCore.QRect(0, 0, w, h)
        sourceRect = self.exporter.getSourceRect()

        bg = np.empty((h, w, 4), dtype=np.ubyte)
        color = self.exporter.params['background']
        bg[:,:,0] = color.red()
        bg[:,:,1] = color.green()
        bg[:,:,2] = color.blue()
        bg[:,:,3] = color.alpha()

        self.exporter.png = fn.makeQImage(bg, alpha=True, copy=False, transpose=False)
        self.exporter.bg = bg

        ## set resolution of image:
        origTargetRect = self.exporter.getTargetRect()
        resolutionScale = targetRect.width() / origTargetRect.width()

        painter = QtGui.QPainter(self.exporter.png)
        try:
            self.exporter.setExportMode(True, {'antialias': self.exporter.params['antialias'], 'background': self.exporter.params['background'], 'painter': painter, 'resolutionScale': resolutionScale})
            painter.setRenderHint(QtGui.QPainter.Antialiasing, self.exporter.params['antialias'])
            self.exporter.getScene().render(painter, QtCore.QRectF(targetRect), QtCore.QRectF(sourceRect))
        finally:
            self.exporter.setExportMode(False)
        painter.end()

        self.exporter.png = self.exporter.png.convertToFormat(QtGui.QImage.Format_ARGB32)
        width = self.exporter.png.width()
        height = self.exporter.png.height()

        # self.exporter.png is a QImage object.
        ptr = self.exporter.png.constBits()
        # ptr is a sip.voidptr type and needs to have the size set before we can convert it to a list or array
        ptr.setsize(width * height * 4)
        arr = np.array(ptr).reshape(height, width, 4)  #  Copies the data
        return arr

        return self.exporter.png

