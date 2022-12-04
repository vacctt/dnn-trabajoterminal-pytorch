import matplotlib.pyplot as plot

class Plot:
    def __init__(self, title=""):
        self.title = title

    def makeGraph(self, axis_items=[], x_axis_title="", y_axis_title=""):
        for i in range(0,len(axis_items)):
            plot.plot(axis_items[i]['items-x'], axis_items[i]['items-y'], label=axis_items[i]['title'])

        plot.xlabel(x_axis_title)
        plot.ylabel(y_axis_title)
        plot.title(self.title)
        plot.legend()
        plot.show()

    def changeGraphTitle(self,newTitle):
        self.title = newTitle