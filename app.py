from msilib import datasizemask
import pyqtgraph as pg, matplotlib, numpy as np, pathlib
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QSlider
from pyqtgraph import *
import numpy as np
import pathlib
from sympy import S, symbols, printing
from mainapp import Ui_MainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import threading
import pandas as pd

matplotlib.use('QT5Agg')


class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self,parent=None, dpi = 120):
        self.fig, self.axes = plt.subplots()
        super(MatplotlibCanvas,self).__init__(self.fig)
        self.axes.set_facecolor((242/255,243/255,245/255))
        self.fig.set_facecolor((242/255,243/255,245/255))


class mainApp(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(mainApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.SliderOfOrder.setMinimum(0)
        self.ui.SliderOfOrder.setMaximum(10)
        self.ui.SliderOfOrder.setTickInterval(1)
        self.ui.SliderOfOrder.setTickPosition(QSlider.TicksBelow)

        self.ui.SliderOfChunks.setMinimum(0)
        self.ui.SliderOfChunks.setMaximum(10)
        self.ui.SliderOfChunks.setTickInterval(1)
        self.ui.SliderOfChunks.setTickPosition(QSlider.TicksBelow)



        self.mainSignalDataPlotter = self.ui.mainSignal.plot([], [], pen = "k")
        self.verticalLinePlotter = self.ui.mainSignal.plot([],[], pen = "g")
        self.extraPolationPlotter = self.ui.mainSignal.plot([],[], pen = "r")


        self.chunkplotter = self.ui.mainSignal
        self.ui.actionopen.triggered.connect(lambda :self.read_file())
        self.ui.SliderOfOrder.valueChanged.connect(self.chunks)
        self.ui.SliderOfChunks.valueChanged.connect(self.chunks)

        self.errorMapCanv = MatplotlibCanvas(self)
        self.errorMapCanv.setMaximumWidth(427)
        self.errorMapCanv.setMaximumHeight(260)
        self.ui.errorMapLayout.addWidget(self.errorMapCanv)
        self.errorMapCanv.setVisible(False)

        self.latexWriter = MatplotlibCanvas(self)
        self.latexWriter.setMaximumHeight(260)
        self.latexWriter.axes.set_axis_off()
        self.ui.latexLayout.addWidget(self.latexWriter)

        self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal = [[],[]]

        self.ui.showErrorMapCheckBox.stateChanged.connect(self.showHideErrorMap)
        self.ui.extrapolationCheckBox.stateChanged.connect(self.extraPolationToggle)
        # self.ui.extrapolationCheckBox.stateChanged.connect()
        self.ui.generateErrorMapBtn.clicked.connect(self.errorMapStart)
        self.ui.cancelBtn.clicked.connect(self.errorMapCancel)
        self.ui.xAxesComboBox.currentIndexChanged['int'].connect(self.showErrorMap)
        self.ui.yAxesComboBox.currentIndexChanged['int'].connect(self.showErrorMap)
        self.errorMapGenerated = False
        divider = make_axes_locatable(self.errorMapCanv.axes)
        self.cax = divider.append_axes('right', size='5%', pad=0.05)
        self.cax.set_facecolor((242/255,243/255,245/255))
        

    def errorMapStart(self):
        self.errorMapProcessor = threading.Thread(target=self.generateErrorMap)
        self.stopFlag = False
        self.errorMapProcessor.start()

    def errorMapCancel(self):
        self.stopFlag = True


    def showHideErrorMap(self, state):
        if (state == QtCore.Qt.Checked):
            self.errorMapCanv.setVisible(True)
        else:
            self.errorMapCanv.setVisible(False)

    def extraPolationToggle(self, state):

        if (state == QtCore.Qt.Checked):
            self.ui.SliderOfOrder.valueChanged.disconnect(self.chunks)
            self.ui.SliderOfChunks.valueChanged.disconnect(self.chunks)
            self.chunkplotter.clear()
            self.ui.chunksLabel.setText("extrapolation")
            self.ui.SliderOfChunks.valueChanged.connect(self.updateExtraPolation)
            self.ui.SliderOfOrder.valueChanged.connect(self.updateExtraPolation)
            self.mainSignalDataPlotter = self.ui.mainSignal.plot(self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal, pen = "k")
            self.verticalLinePlotter = self.ui.mainSignal.plot([],[], pen = "g")
            self.extraPolationPlotter = self.ui.mainSignal.plot([],[], pen = "r")
            self.updateExtraPolation()

        else:
            self.ui.SliderOfOrder.valueChanged.disconnect(self.updateExtraPolation)
            self.ui.SliderOfChunks.valueChanged.disconnect(self.updateExtraPolation)
            self.ui.chunksLabel.setText("Chunks")
            self.ui.SliderOfChunks.valueChanged.connect(self.chunks)
            self.ui.SliderOfOrder.valueChanged.connect(self.chunks)
            self.verticalLinePlotter.setData([],[])
            self.extraPolationPlotter.setData([],[])
            self.chunks()

    def updateExtraPolation(self):


        extraPolationEndIndex = self.ui.SliderOfChunks.value()*100
        verticalLineXIndex = max(self.xAxisValuesOfOriginalSignal)*extraPolationEndIndex/1000
        polynomialOrder = self.ui.SliderOfOrder.value()
        polynomial = np.polyfit(self.xAxisValuesOfOriginalSignal[:extraPolationEndIndex], self.yAxisValuesOfOriginalSignal[:extraPolationEndIndex], polynomialOrder)
        polynomialVals = np.poly1d(polynomial)
        self.verticalLinePlotter.setData([verticalLineXIndex,verticalLineXIndex],[min(self.yAxisValuesOfOriginalSignal),max(self.yAxisValuesOfOriginalSignal)])
        self.extraPolationPlotter.setData(self.xAxisValuesOfOriginalSignal,polynomialVals(self.xAxisValuesOfOriginalSignal))
    
    def showErrorMap(self):
        if self.errorMapGenerated == False:
            return

        self.errorMapCanv.axes.clear()
        self.cax.clear()
        
        xAxesComboBoxIndex = self.ui.xAxesComboBox.currentIndex()
        yAxesComboBoxIndex = self.ui.yAxesComboBox.currentIndex()
        
        if (xAxesComboBoxIndex == 0 and yAxesComboBoxIndex == 1):
            image = self.errorMapCanv.axes.imshow(self.polynomialOrderToNumberOfChunksError,cmap = "inferno",extent=[0,self.maxNumberOfChunks,0,self.maxPolyNomialOrder])
            self.errorMapCanv.axes.set_xlabel("number of chunks")
            self.errorMapCanv.axes.set_ylabel("polynomial order")
        elif (xAxesComboBoxIndex == 1 and yAxesComboBoxIndex == 0):
            image = self.errorMapCanv.axes.imshow(self.numberOfChunksToPolynomialOrderError,cmap = "inferno",extent=[0,self.maxPolyNomialOrder,0,self.maxNumberOfChunks])
            self.errorMapCanv.axes.set_xlabel("polynomial order")
            self.errorMapCanv.axes.set_ylabel("number of chunks")

        elif (xAxesComboBoxIndex == 0 and yAxesComboBoxIndex == 2):
            image = self.errorMapCanv.axes.imshow(self.overLappingPercentageToNumberOfChunksError,cmap = "inferno",extent=[0,self.maxNumberOfChunks,0,25])
            self.errorMapCanv.axes.set_xlabel("number of chunks")
            self.errorMapCanv.axes.set_ylabel("overlapping percentage")
        elif (xAxesComboBoxIndex == 2 and yAxesComboBoxIndex == 0):
            image = self.errorMapCanv.axes.imshow(self.numberOfChunksToOverLappingPercentageError,cmap = "inferno",extent=[0,25,0,self.maxNumberOfChunks])
            self.errorMapCanv.axes.set_xlabel("overlapping percentage")
            self.errorMapCanv.axes.set_ylabel("number of chunks")
        
        elif (xAxesComboBoxIndex == 1 and yAxesComboBoxIndex == 2):
            image = self.errorMapCanv.axes.imshow(self.overLappingPercentageToPolynomialOrderError,cmap = "inferno",extent=[0,self.maxPolyNomialOrder,0,25])
            self.errorMapCanv.axes.set_xlabel("polynomial order")
            self.errorMapCanv.axes.set_ylabel("overlapping percentage")
        else:
            image = self.errorMapCanv.axes.imshow(self.polynomialOrderToOverlappingPercentageError,cmap = "inferno",extent=[0,25,0,self.maxPolyNomialOrder])
            self.errorMapCanv.axes.set_xlabel("overlapping percentage")
            self.errorMapCanv.axes.set_ylabel("polynomial order")


        self.errorMapCanv.fig.colorbar(image, cax=self.cax, orientation='vertical')
        self.errorMapCanv.fig.tight_layout()
        self.errorMapCanv.draw()

    def get2DErrorArray(self, polynomialOrderVals = [5], numberOfChunksVals = [5], overlappingSizePercentageVals = [0]):

        if (len(polynomialOrderVals) != 1 and len(numberOfChunksVals) != 1):
            errorMap = np.zeros((len(polynomialOrderVals),len(numberOfChunksVals)))
        elif (len(polynomialOrderVals) != 1):
            errorMap = np.zeros((len(polynomialOrderVals),len(overlappingSizePercentageVals)))
        else:
            errorMap = np.zeros((len(overlappingSizePercentageVals),len(numberOfChunksVals)))

        for polynomialOrder in polynomialOrderVals:
            for overlappingSizePercentage in overlappingSizePercentageVals:
                for numberOfChunks in numberOfChunksVals:
                    if (self.stopFlag):
                        return

                    xAxesChunksValues = np.array_split(self.xAxisValuesOfOriginalSignal,numberOfChunks)
                    yAxesChunksValues = np.array_split(self.yAxisValuesOfOriginalSignal,numberOfChunks)

                    chunkSize = int(len(self.xAxisValuesOfOriginalSignal) / numberOfChunks)
                    overlappedPointsNumber = int(chunkSize * overlappingSizePercentage /100)
                    resError = 0

                    for chunkIndex in range(numberOfChunks):

                        chunkStartIndex = (chunkIndex)*len(xAxesChunksValues[chunkIndex])
                        chunkEndIndex = chunkStartIndex + len(xAxesChunksValues[chunkIndex])

                        if (numberOfChunks > 1):
                            
                            # case 1: we want append to the first chunk
                            if (chunkIndex == 0):
                                chunkEndIndex += overlappedPointsNumber
                                
                            # case 2: we want to append to the last chunk
                            elif (chunkIndex == (numberOfChunks - 1)):
                                chunkStartIndex -= overlappedPointsNumber
                                
                            # case 3: we want to append to a mid chunk
                            else:
                                chunkStartIndex -= int(overlappedPointsNumber/2)
                                chunkEndIndex += int(np.ceil(overlappedPointsNumber/2))

                        xAxesChunksValues[chunkIndex] = self.xAxisValuesOfOriginalSignal[chunkStartIndex:chunkEndIndex]
                        yAxesChunksValues[chunkIndex] = self.yAxisValuesOfOriginalSignal[chunkStartIndex:chunkEndIndex]
                        polynomial = np.polyfit(xAxesChunksValues[chunkIndex], yAxesChunksValues[chunkIndex], polynomialOrder)  
                        resError += np.sum((np.polyval(polynomial, xAxesChunksValues[chunkIndex]) - yAxesChunksValues[chunkIndex])**2)
                    
                    resError /= numberOfChunks
                    error = np.sqrt(resError/(len(self.xAxisValuesOfOriginalSignal)-2))
                    
                    if (len(polynomialOrderVals) != 1 and len(numberOfChunksVals) != 1):
                        errorMap[polynomialOrder - 1][numberOfChunks - 1] = error
                    elif (len(polynomialOrderVals) != 1):
                        errorMap[polynomialOrder - 1][int(overlappingSizePercentage/5)] = error
                    else:
                        errorMap[int(overlappingSizePercentage/5)][numberOfChunks - 1] = error

        return errorMap

    def normalizeAndPrepData(self,data):
        normalizedData = self.normalizeData(data)
        swapedAxesNormalizedData_errorMapModified = np.flip(np.transpose(data),0)
        normalizedData_errorMapModified = np.flip(normalizedData,0)
        return [normalizedData_errorMapModified, swapedAxesNormalizedData_errorMapModified]

    def generateErrorMap(self):

        self.maxPolyNomialOrder = self.ui.orderOfFittingPolynomialErrorMapSlider.value()
        self.maxNumberOfChunks = self.ui.numberOfChunksErrorMapSlider.value()
        

        error2DArray = self.get2DErrorArray(polynomialOrderVals=range(1,self.maxPolyNomialOrder+1),numberOfChunksVals=range(1,self.maxNumberOfChunks+1))        
        [self.polynomialOrderToNumberOfChunksError, self.numberOfChunksToPolynomialOrderError] = self.normalizeAndPrepData(error2DArray)


        error2DArray = self.get2DErrorArray(overlappingSizePercentageVals=range(0,26,5),numberOfChunksVals=range(1,self.maxNumberOfChunks+1))
        [self.overLappingPercentageToNumberOfChunksError, self.numberOfChunksToOverLappingPercentageError] = self.normalizeAndPrepData(error2DArray)


        error2DArray = self.get2DErrorArray(polynomialOrderVals=range(1,self.maxPolyNomialOrder),overlappingSizePercentageVals=range(0,26,5))
        [self.polynomialOrderToOverlappingPercentageError, self.overLappingPercentageToPolynomialOrderError] = self.normalizeAndPrepData(error2DArray)

        self.errorMapGenerated = True
        self.showErrorMap()
                                       
    def normalizeData(self,Data):
        maxOfData = Data.max()
        for raw in range(len(Data)):
            for col in range(len(Data[0])):
                Data[raw][col] = (Data[raw][col])/(maxOfData) * 100

        return Data

    def read_file(self):

        path = QFileDialog.getOpenFileName()[0]
        self.FileName= os.path.basename(path)

        if pathlib.Path(path).suffix == ".csv":
            self.data = np.genfromtxt(path, delimiter=',')
            self.xAxisValuesOfOriginalSignal = self.data[:,0][:1000]
            self.yAxisValuesOfOriginalSignal = self.data[:,1][:1000]
            self.mainSignalDataPlotter.setData(self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal)

        self.ui.mainSignal.setLimits(yMin = min(self.yAxisValuesOfOriginalSignal), yMax = max(self.yAxisValuesOfOriginalSignal))

    def chunks(self):
        # self.errorMapCanv.setVisible(True)
        # else:
        #     self.errorMapCanv.setVisible(False)
        tableData = {'Chunk': [], 'equation': [], 'error%': []}
        self.numberofChunkies = self.ui.SliderOfChunks.value()
        self.chunkplotter.clear()
        self.chunkplotter.plot(self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal,pen = "k")
        self.i=0
        self.xAxisValuesofsplit=np.array_split(self.xAxisValuesOfOriginalSignal, self.numberofChunkies)
        self.yAxisValuesofsplit=np.array_split(self.yAxisValuesOfOriginalSignal, self.numberofChunkies)
        # self.latexWriter.axes.clear()
        # self.latexWriter.axes.set_axis_off()
        # self.ui.equationWriter.clear()
        if(self.numberofChunkies > 0):

            for chunkidx in range(self.numberofChunkies):
                order = self.ui.SliderOfOrder.value()
                self.polynomial = numpy.polyfit(self.xAxisValuesofsplit[chunkidx], self.yAxisValuesofsplit[chunkidx], order)
                self.xAxischunks = np.linspace(min(self.xAxisValuesofsplit[chunkidx]),max(self.xAxisValuesofsplit[chunkidx]))
                self.yAxischunks = np.polyval(self.polynomial, self.xAxischunks)
                # print(self.yAxischunks)

                self.res = numpy.sum((numpy.polyval(
                    numpy.polyfit(self.xAxisValuesofsplit[chunkidx], self.yAxisValuesofsplit[chunkidx], order),
                    self.xAxisValuesofsplit[chunkidx]) - self.yAxisValuesofsplit[chunkidx]) ** 2)

                self.res /= self.numberofChunkies
                self.error1 = np.sqrt(self.res / (len(self.xAxisValuesOfOriginalSignal) - 2))

                self.error = self.error1 * 100



                self.xAxisValuesofsplit[chunkidx] = symbols("x")
                self.poly = sum(S("{:6.2f}".format(v)) * self.xAxisValuesofsplit[chunkidx] ** i
                                                for i, v in enumerate(self.polynomial [::-1]))
                self.eq_latex = printing.latex(self.poly)


                self.chunkplotter.plot(self.xAxischunks, self.yAxischunks, symbol='o')
                tableData['error%'].append(round(self.error, 3))
                tableData['equation'].append(f'${self.eq_latex}$')
                tableData['Chunk'].append(f'eq of chunk {(chunkidx+1)}')

                # self.ui.equationWriter.append(f'eq of chunk {(chunkidx+1)}: ' + self.eq_latex)
                # self.ui.equationWriter.append('percentage error: ' + str(self.error))

            # TableData_frame = pd.DataFrame(tableData) ## framing the table with pandas library
            self.latexWriter.axes.clear()

            if (order > 0): ## plot the table only if the spin box has value
                TableData_frame = pd.DataFrame(tableData) ## forming the table with pandas library
                table=self.latexWriter.axes.table(cellText=TableData_frame.values, colLabels=TableData_frame.columns, loc='center')
                table.auto_set_column_width(col=list(range(len(TableData_frame.columns)))) ## orientation
                table.scale(1, 1.5)
                table.auto_set_font_size(True)
            
            # table = self.latexWriter.axes.table(cellText=TableData_frame.values, colLabels=TableData_frame.columns, loc='center')
            # table.scale(1,2)
            self.latexWriter.fig.tight_layout()
            self.latexWriter.axes.set_axis_off()
            self.latexWriter.draw()










if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = mainApp()
    main.show()
    sys.exit(app.exec_())

