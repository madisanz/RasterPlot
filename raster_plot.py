
import numpy
import matplotlib.pyplot as pylab
from pylab import show, savefig

def make_plot(ts, ts1, gids, neurons, hist=True, hist_binwidth=5.0,
               grayscale=False, title=None, xlabel=None, skipT=1, subtract=0):
    
    pylab.figure()

    if grayscale:
        color_marker = "|k"
        color_bar = "gray"
    else:
        color_marker = "."
        color_bar = "blue"

    color_edge = "black"

    if xlabel is None:
        xlabel = "Time (s)"

    ylabel = "Neuron ID"

    
    if hist:

        ax1 = pylab.axes([0.1, 0.3, 0.85, 0.6])
        plotid = pylab.plot(ts1[::skipT], numpy.array(gids[::skipT]) - subtract, color_marker)
        pylab.ylabel(ylabel)
        pylab.xticks([])
        xlim = pylab.xlim()
        
        pylab.axes([0.1, 0.1, 0.85, 0.17])
        t_bins = numpy.arange(
            numpy.amin(ts), numpy.amax(ts),
            float(hist_binwidth)
        )
        n, bins = _histogram(ts, bins=t_bins)
        num_neurons = len(numpy.unique(neurons))
        heights = 1000 * n / (hist_binwidth * num_neurons)

        pylab.bar(t_bins, heights, width=hist_binwidth, color=color_bar,
                  edgecolor=color_edge)
        pylab.yticks([
            int(x) for x in
            numpy.linspace(0.0, int(max(heights) * 1.1) + 5, 4)
        ])
        pylab.ylabel("Rate (Hz)")
        pylab.xlabel(xlabel)
        pylab.xlim(xlim)
        
        maxt = int(max(t_bins)/1000.) + 1
        pylab.xticks(range(0,1000*maxt+1,2000), range(0,maxt+1,2))
        pylab.axes(ax1)
    else:
        plotid = pylab.plot(ts1, gids, color_marker)
        pylab.xlabel(xlabel)
        pylab.ylabel(ylabel)

    if title is None:
        pylab.title("Raster plot")
    else:
        pylab.title(title)

    pylab.draw()

    return plotid


def _histogram(a, bins=10, bin_range=None, normed=False):
   
    from numpy import asarray, iterable, linspace, sort, concatenate

    a = asarray(a).ravel()

    if bin_range is not None:
        mn, mx = bin_range
        if mn > mx:
            raise ValueError("max must be larger than min in range parameter")

    if not iterable(bins):
        if bin_range is None:
            bin_range = (a.min(), a.max())
        mn, mx = [mi + 0.0 for mi in bin_range]
        if mn == mx:
            mn -= 0.5
            mx += 0.5
        bins = linspace(mn, mx, bins, endpoint=False)
    else:
        if (bins[1:] - bins[:-1] < 0).any():
            raise ValueError("bins must increase monotonically")

    block = 65536
    n = sort(a[:block]).searchsorted(bins)
    for i in range(block, a.size, block):
        n += sort(a[i:i + block]).searchsorted(bins)
    n = concatenate([n, [len(a)]])
    n = n[1:] - n[:-1]

    if normed:
        db = bins[1] - bins[0]
        return 1.0 / (a.size * db) * n, bins
    else:
        return n, bins
